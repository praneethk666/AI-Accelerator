"""Resolution-driven tiling for large-format / diagram pages (CAD, circuit schematics,
E-size engineering sheets).

Problem: a VLM downsamples a whole large sheet to ~1.5-2k px, so tiny reference
designators (R12, C4, U3), net labels and title-block text turn to mush. Sending the
page whole therefore loses exactly the detail these documents are about.

Two strategies (extraction.large_format.strategy):

  grid (transcribe_large_page / transcribe_large_page_blocks) — render the page at
    high DPI and split it into OVERLAPPING tiles small enough to stay legible,
    transcribe each tile at full detail, then merge + deduplicate. Tile COUNT falls
    out of the actual page size vs the VLM's legible resolution — a normal A4 is 1
    tile, an E-size sheet becomes e.g. a 2x3 grid automatically. Simple and robust,
    but sends every tile through the VLM regardless of content density, and a tile
    boundary can cut a component/table in half.

  agentic (transcribe_large_page_regions), DEFAULT — locate distinct regions first
    (one cheap coarse-resolution call: locate_regions), then zoom into just each
    real region with a targeted high-res crop (transcribe_regions_blocks), instead
    of mechanically slicing the whole sheet into a fixed grid. "Coarse-to-fine" /
    "localized zoom", the real 2025-2026 pattern (Zoom-Refine, ZoomEye). Falls back
    to grid tiling if the locate pass finds nothing (never worse than before).
    Live-tested finding (4-Aug): bbox GROUNDING reliability varies a lot by model —
    a free-tier vision model can identify regions well but return bboxes that mix
    normalized and raw-pixel values in the same array; see locate_vision below.

Config (extraction.large_format):
  enabled       : master switch (default True)
  strategy      : "agentic" (default) or "grid"
  locate_dpi    : coarse locate-pass render DPI, agentic only (default 100)
  locate_max_px : cap so even a huge sheet fits the locate call in ONE image,
                  agentic only (default 2000)
  locate_vision : optional vision config dict used ONLY for the one coarse locate
                  call per oversized page (agentic only) — falls back to the
                  general vision_ocr config if unset. Split out because this is a
                  bbox-GROUNDING task, and empirically not every vision model that
                  transcribes content well also grounds boxes reliably; this lets
                  a customer point the (cheap, rare) locate call at a stronger
                  model while the many per-region zoom/detail calls keep using
                  whatever provider is otherwise configured.
  dpi           : render DPI for the zoom-pass crops (agentic) / tiling (grid)
                  (default 200)
  vlm_max_px    : grid only — target max tile edge in px the VLM reads cleanly
                  (default 2600 — keeps A4/Letter at 1 tile, tiles only genuinely
                  oversized sheets A3/D/E)
  overlap       : grid only — fractional tile overlap so components on a seam
                  aren't cut (default 0.12)
  merge         : grid only — "llm" (dedup-merge via the text LLM) or "concat"
                  (default "llm")
  max_tiles     : grid only — safety cap on tiles per page (default 24)

No per-document constants, no "CAD = aspect > X" hardcoding — every threshold here
is a config value so behavior is tunable per customer/document type, not baked in.
"""
from __future__ import annotations

import io
import logging
import math

from backend.core import prompts
from backend.core.vision_client import describe_image

logger = logging.getLogger(__name__)


def _cfg(config: dict) -> dict:
    return (config.get("extraction") or {}).get("large_format") or {}


def tile_grid(page_w_pt: float, page_h_pt: float, dpi: int, vlm_max_px: int) -> tuple[int, int]:
    """How many tiles per axis so each tile's rendered edge <= vlm_max_px. (1,1) means
    the page is small enough to send whole — no tiling needed."""
    w_px = page_w_pt * dpi / 72.0
    h_px = page_h_pt * dpi / 72.0
    tx = max(1, math.ceil(w_px / max(vlm_max_px, 256)))
    ty = max(1, math.ceil(h_px / max(vlm_max_px, 256)))
    return tx, ty


def needs_tiling(page, config: dict) -> bool:
    """True when this page rendered at the tiling DPI would exceed the VLM's legible
    resolution in either axis — i.e. it's a large-format sheet that must be tiled."""
    c = _cfg(config)
    if not c.get("enabled", True):
        return False
    tx, ty = tile_grid(page.rect.width, page.rect.height,
                       int(c.get("dpi", 200)), int(c.get("vlm_max_px", 2600)))
    return tx * ty > 1


def _render(page, dpi: int):
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _tiles(img, tx: int, ty: int, overlap: float):
    """Yield (row, col, PIL tile) for a tx*ty grid with fractional overlap."""
    W, H = img.size
    tw, th = W / tx, H / ty
    ox, oy = tw * overlap, th * overlap
    for r in range(ty):
        for c in range(tx):
            l = max(0, int(c * tw - ox))
            t = max(0, int(r * th - oy))
            rgt = min(W, int((c + 1) * tw + ox))
            bot = min(H, int((r + 1) * th + oy))
            yield r, c, img.crop((l, t, rgt, bot))


def _merge(parts: list[str], config: dict) -> str:
    """Merge per-tile transcriptions into one deduplicated description. Prefer an LLM
    dedup-merge (overlapping tiles repeat content); fall back to plain concatenation."""
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return ""
    if len(parts) == 1 or (_cfg(config).get("merge") or "llm").lower() != "llm":
        return "\n\n".join(parts)
    try:
        from backend.core.llm_client import get_llm
        from backend.core import usage
        body = "\n\n".join(f"[tile {i + 1}]\n{p}" for i, p in enumerate(parts))
        full_prompt = prompts.SCHEMATIC_MERGE + body
        reply = get_llm(config).invoke(full_prompt)
        usage.record_from_message("large_format_merge", reply, prompt=full_prompt, model=config.get("llm", {}).get("model"), provider=config.get("llm", {}).get("provider"))
        out = (getattr(reply, "content", str(reply)) or "").strip()
        return out or "\n\n".join(parts)
    except Exception as e:
        logger.warning("large_format: LLM merge failed (%s); concatenating tiles", e)
        return "\n\n".join(parts)


def transcribe_large_page(page, config: dict, prompt: str | None = None) -> str:
    """Tile a large-format page, transcribe each tile, and return the merged Markdown.
    Falls back to a single whole-page transcription if tiling isn't needed or fails."""
    c = _cfg(config)
    dpi = int(c.get("dpi", 200))
    vlm_max_px = int(c.get("vlm_max_px", 2600))
    overlap = float(c.get("overlap", 0.12))
    max_tiles = int(c.get("max_tiles", 24))
    tile_prompt = prompt or prompts.SCHEMATIC_TILE
    vcfg = {"vision": config.get("vision_ocr") or {}}

    tx, ty = tile_grid(page.rect.width, page.rect.height, dpi, vlm_max_px)
    if tx * ty <= 1:
        return describe_image(page.get_pixmap(dpi=dpi).tobytes("png"), tile_prompt, vcfg).strip()
    if tx * ty > max_tiles:                       # clamp the grid to the tile budget
        scale = math.sqrt(max_tiles / (tx * ty))
        tx, ty = max(1, int(tx * scale)), max(1, int(ty * scale))
    logger.info("large_format: tiling page into %dx%d at %ddpi", tx, ty, dpi)

    img = _render(page, dpi)
    parts: list[str] = []
    for r, c_, tile in _tiles(img, tx, ty, overlap):
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        try:
            txt = describe_image(buf.getvalue(), tile_prompt, vcfg).strip()
        except Exception as e:
            logger.warning("large_format: tile (%d,%d) failed (%s)", r, c_, e)
            txt = ""
        if txt:
            parts.append(txt)
    return _merge(parts, config)

def _remap_bbox(bbox, crop_box, img_size):
    """Convert a bbox normalized to a tile/region crop into full-page normalized
    coords. Returns None (never a garbage value) if `bbox` isn't a clean 0.0-1.0
    normalized box to begin with.

    Real bug found live, 3-Aug: the model is asked for 0.0-1.0 normalized
    coordinates but doesn't always comply -- it sometimes returns raw PIXEL
    coordinates for a tile instead. Un-validated, those get remapped as if they
    WERE normalized (multiplied by the tile's width/height fraction and divided
    by the full page size), producing nonsense values like 279.12 or 373.30
    instead of a 0-1 fraction -- silently corrupting every downstream consumer of
    that bbox (citations, a future zoom-into-region tool, anything spatial)."""
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        return None
    l, t, rgt, bot = crop_box
    W, H = img_size
    tw, th = (rgt - l), (bot - t)
    return [
        (l + x1 * tw) / W,
        (t + y1 * th) / H,
        (l + x2 * tw) / W,
        (t + y2 * th) / H,
    ]


def transcribe_large_page_blocks(page, config: dict, prompt: str) -> list[dict]:
    """Tile a large-format page, transcribe each tile as JSON region blocks (per
    `prompt`'s schema), remap bboxes to full-page coordinates, and return the combined
    block list — no text/Markdown merge, so table_data survives intact."""
    c = _cfg(config)
    dpi = int(c.get("dpi", 200))
    vlm_max_px = int(c.get("vlm_max_px", 2600))
    overlap = float(c.get("overlap", 0.12))
    max_tiles = int(c.get("max_tiles", 24))
    vcfg = {"vision": config.get("vision_ocr") or {}}

    tx, ty = tile_grid(page.rect.width, page.rect.height, dpi, vlm_max_px)
    img = _render(page, dpi)
    W, H = img.size

    if tx * ty <= 1:
        raw = describe_image(page.get_pixmap(dpi=dpi).tobytes("png"), prompt, vcfg).strip()
        return [{"_raw": raw, "_crop": (0, 0, W, H), "_img_size": (W, H)}]

    if tx * ty > max_tiles:
        scale = math.sqrt(max_tiles / (tx * ty))
        tx, ty = max(1, int(tx * scale)), max(1, int(ty * scale))
    logger.info("large_format: tiling page into %dx%d at %ddpi (structured)", tx, ty, dpi)

    all_blocks: list[dict] = []
    for r, c_, tile in _tiles(img, tx, ty, overlap):
        # recompute this tile's crop box in full-image pixel coords (same math as _tiles)
        w_step, h_step = W / tx, H / ty
        ox, oy = w_step * overlap, h_step * overlap
        l = max(0, int(c_ * w_step - ox))
        t = max(0, int(r * h_step - oy))
        rgt = min(W, int((c_ + 1) * w_step + ox))
        bot = min(H, int((r + 1) * h_step + oy))

        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        try:
            raw = describe_image(buf.getvalue(), prompt, vcfg).strip()
        except Exception as e:
            logger.warning("large_format: tile (%d,%d) failed (%s)", r, c_, e)
            continue

        from backend.vision.block_builder import _extract_json
        data = _extract_json(raw)
        items = data if isinstance(data, list) else (
            data.get("blocks") if isinstance(data, dict) and isinstance(data.get("blocks"), list)
            else None)
        if not items:
            continue
        for vb in items:
            if not isinstance(vb, dict):
                continue
            vb["bbox"] = _remap_bbox(vb.get("bbox"), (l, t, rgt, bot), (W, H))
            all_blocks.append(vb)

    return _dedup_blocks(all_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# AGENTIC alternative to blind grid tiling above: locate distinct regions first
# (one cheap coarse-resolution call), then zoom into each with a targeted
# high-res crop -- instead of mechanically slicing the whole sheet into a fixed
# grid regardless of content density. Real 2025-2026 research pattern
# ("coarse-to-fine" / "localized zoom": Zoom-Refine, ZoomEye) applied here after
# a live test found the blind-tiling path both slow (~7min on one real E-size
# sheet) AND producing corrupted bboxes on a meaningful fraction of tiles.
# ─────────────────────────────────────────────────────────────────────────────

def _region_crop_box(bbox_norm: list[float], img_size: tuple[int, int],
                     pad_frac: float = 0.03) -> tuple[int, int, int, int]:
    """A located region's normalized bbox -> a pixel crop box in the full-res
    render, with a small margin so content right at the located edge isn't clipped."""
    W, H = img_size
    x1, y1, x2, y2 = bbox_norm
    bw, bh = (x2 - x1) * W, (y2 - y1) * H
    padx, pady = bw * pad_frac, bh * pad_frac
    l = max(0, int(x1 * W - padx))
    t = max(0, int(y1 * H - pady))
    rgt = min(W, int(x2 * W + padx))
    bot = min(H, int(y2 * H + pady))
    return l, t, rgt, bot


def locate_regions(page, config: dict) -> list[dict]:
    """Coarse pass: locate distinct regions on a large-format sheet (bbox + type +
    label + one-line description) WITHOUT transcribing their contents in detail --
    one call, at a resolution low enough that even a huge sheet fits in one image,
    since layout doesn't need fine-print legibility. Returns [] on any failure
    (call error, unparseable reply, or a genuinely region-less sheet) -- the
    caller falls back to blind tiling in that case, never worse than before."""
    c = _cfg(config)
    locate_dpi = int(c.get("locate_dpi", 100))
    locate_max_px = int(c.get("locate_max_px", 2000))
    # Live-tested 4-Aug on a real E-size sheet: the CAD-shadowed free-tier vision
    # model (NVIDIA nemotron-nano-12b-v2-vl) returned bbox arrays that MIX
    # normalized (0.72) and raw-pixel (992, 215) values in the SAME array for this
    # coarse locate task -- every region got correctly rejected by the bbox
    # validation below, so locate_regions() always came back empty and every
    # oversized page silently fell back to blind tiling (no worse than before, but
    # none of the agentic benefit). gpt-4o-mini, tested on the identical image, got
    # 11/11 valid normalized boxes. `locate_vision` lets this ONE coarse call per
    # oversized page use a more reliable model while the many per-region zoom/detail
    # calls (transcribe_regions_blocks) stay on the cheaper/free CAD vision config --
    # falls back to the general vision_ocr config if not set, so this is opt-in.
    locate_vision = c.get("locate_vision")
    vcfg = {"vision": locate_vision or config.get("vision_ocr") or {}}

    w_px = page.rect.width * locate_dpi / 72.0
    h_px = page.rect.height * locate_dpi / 72.0
    scale = min(1.0, locate_max_px / max(w_px, h_px, 1.0))
    dpi = max(36, int(locate_dpi * scale))

    try:
        raw = describe_image(page.get_pixmap(dpi=dpi).tobytes("png"),
                             prompts.LOCATE_REGIONS, vcfg).strip()
    except Exception as e:
        logger.warning("large_format: region locate call failed (%s)", e)
        return []

    from backend.vision.block_builder import _extract_json
    data = _extract_json(raw)
    items = data if isinstance(data, list) else (
        data.get("regions") if isinstance(data, dict) and isinstance(data.get("regions"), list)
        else None)
    if not items:
        return []

    regions: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bbox = it.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
            continue
        regions.append({
            "type": it.get("type") or "view",
            "label": it.get("label") or f"region_{len(regions) + 1}",
            "description": (it.get("description") or "").strip(),
            "bbox": [x1, y1, x2, y2],
        })
    return regions


def transcribe_regions_blocks(page, config: dict, regions: list[dict],
                              detail_prompt: str) -> list[dict]:
    """Zoom pass: for each located region, crop it from a FULL-resolution render
    and run the detailed extraction prompt on just that crop. Returns the same raw
    vb-dict shape transcribe_large_page_blocks does (bbox already remapped to
    full-page normalized coords) -- a drop-in alternative at the call site. Each
    block's metadata also gets region_label/region_description so a block can
    always be traced back to which located region produced it."""
    c = _cfg(config)
    dpi = int(c.get("dpi", 200))
    vcfg = {"vision": config.get("vision_ocr") or {}}

    img = _render(page, dpi)
    W, H = img.size

    all_blocks: list[dict] = []
    for region in regions:
        l, t, rgt, bot = _region_crop_box(region["bbox"], (W, H))
        if rgt <= l or bot <= t:
            continue
        buf = io.BytesIO()
        img.crop((l, t, rgt, bot)).save(buf, format="PNG")
        try:
            raw = describe_image(buf.getvalue(), detail_prompt, vcfg).strip()
        except Exception as e:
            logger.warning("large_format: region '%s' failed (%s)", region.get("label"), e)
            continue

        from backend.vision.block_builder import _extract_json
        data = _extract_json(raw)
        items = data if isinstance(data, list) else (
            data.get("blocks") if isinstance(data, dict) and isinstance(data.get("blocks"), list)
            else None)
        if not items:
            continue
        region_blocks: list[dict] = []
        for vb in items:
            if not isinstance(vb, dict):
                continue
            vb["bbox"] = _remap_bbox(vb.get("bbox"), (l, t, rgt, bot), (W, H))
            md = vb.get("metadata") if isinstance(vb.get("metadata"), dict) else {}
            md["region_label"] = region.get("label")
            md["region_description"] = region.get("description")
            vb["metadata"] = md
            region_blocks.append(vb)
        region_blocks = _cross_check_table_cells(region_blocks, buf.getvalue(), detail_prompt, config)
        all_blocks.extend(region_blocks)

    return _dedup_blocks(all_blocks)


def _is_checkable_value(val: str) -> bool:
    """Skip trivial values where a substring match against independent free-text
    output would be meaningless noise rather than real corroboration -- very
    short strings and plain numbers/dimensions match too easily by chance."""
    v = val.strip()
    if len(v) < 3:
        return False
    if v.replace(".", "").replace(",", "").replace("-", "").isdigit():
        return False
    return True


def _build_verify_prompt(targets: list[tuple]) -> str:
    lines = [f'{i + 1}. field "{header}" = "{val}"' for i, (_, _, header, val) in enumerate(targets)]
    return (
        "Look CAREFULLY at this image. Below is a numbered list of specific values "
        "another reading claims appear in this image. For EACH numbered item, check "
        "whether the image actually, clearly shows that exact value.\n\n"
        + "\n".join(lines) +
        '\n\nReturn ONLY a JSON array, one object per numbered item, in order: '
        '{"n": <item number>, "confirmed": <true|false>}. confirmed=true ONLY if '
        "you can clearly see that exact value at that field in the image; otherwise "
        "confirmed=false (including if the field is illegible, absent, cropped out, "
        "or shows something different). No other text."
    )


def _cross_check_table_cells(blocks: list[dict], crop_bytes: bytes, prompt: str,
                             config: dict) -> list[dict]:
    """For each table_data cell, ask an INDEPENDENT vision model a TARGETED
    confirm/deny question against the SAME crop; a cell it can't confirm gets
    flagged rather than presented as confident fact. Real live finding (4-Aug):
    the primary CAD vision model produced specific, plausible-looking but
    FABRICATED title-block values (invented names like "M.Takeshita") sitting
    right alongside genuinely correct part numbers on the same crop -- this
    isn't a missing-data problem (the sparse-column cross-check in
    docling_extract.py fills gaps), it's a confident-wrong-data problem.

    Deliberately NOT a loose "re-transcribe independently, substring-match the
    free text" check (tried first, live-tested 4-Aug): the verify model is
    naturally conservative and often doesn't attempt full table transcription
    on its own, so almost everything failed to match even probably-correct
    values -- a near-useless signal. Asking it to confirm/deny SPECIFIC values
    one at a time is a much narrower, answerable question and was validated to
    give a real, usable signal instead. This only ADDS an "[unverified: ...]"
    marker, never silently drops or overwrites a value -- same consensus
    principle either way (correct reads converge, errors diverge -- arxiv
    2504.11101).

    Opt-in via extraction.cad.verify_vision (a full vision config dict, same
    shape as extraction.cad.vision) -- unset means this never fires, so a
    customer who hasn't configured it pays zero extra cost/latency."""
    verify_cfg = (config.get("extraction") or {}).get("cad", {}).get("verify_vision")
    if not verify_cfg:
        return blocks

    targets: list[tuple] = []  # (row, col, header, value)
    for b in blocks:
        td = b.get("table_data")
        if not isinstance(td, dict) or not td.get("rows"):
            continue
        headers = td.get("headers") or []
        for row in td["rows"]:
            for i, cell in enumerate(row):
                val = str(cell).strip()
                if not _is_checkable_value(val) or val.startswith("[unverified:"):
                    continue
                header = headers[i] if i < len(headers) else f"column {i}"
                targets.append((row, i, header, val))
    if not targets:
        return blocks

    try:
        raw = describe_image(crop_bytes, _build_verify_prompt(targets), {"vision": verify_cfg})
    except Exception as e:
        logger.warning("large_format: verify cross-check call failed (%s)", e)
        return blocks

    from backend.vision.block_builder import _extract_json
    data = _extract_json(raw)
    results = data if isinstance(data, list) else []
    confirmed_ns: set[int] = set()
    for r in results:
        if isinstance(r, dict) and r.get("confirmed") is True:
            try:
                confirmed_ns.add(int(r["n"]))
            except (KeyError, TypeError, ValueError):
                continue

    flagged = 0
    for i, (row, col, _header, val) in enumerate(targets):
        if (i + 1) not in confirmed_ns:
            row[col] = f"[unverified: {val}]"
            flagged += 1
    if flagged:
        logger.info("large_format: cross-check flagged %d/%d unverified table cell(s)",
                    flagged, len(targets))
    return blocks


def _dedup_table_rows(blocks: list[dict]) -> list[dict]:
    """Drop exact-duplicate ROWS within a single block's table_data. Distinct
    from _dedup_blocks below (which only catches whole repeated blocks): real
    live finding (4-Aug), one table came back with the same 5-value row
    repeated 4 times WITHIN one block's table_data -- the block itself wasn't a
    duplicate of another block, so block-level dedup alone missed it."""
    for b in blocks:
        td = b.get("table_data")
        if not isinstance(td, dict) or not td.get("rows"):
            continue
        seen: set[tuple] = set()
        deduped = []
        for row in td["rows"]:
            key = tuple(str(c) for c in row) if isinstance(row, list) else (str(row),)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        td["rows"] = deduped
    return blocks


def _dedup_blocks(blocks: list[dict]) -> list[dict]:
    """Drop exact-duplicate rows within each table (see _dedup_table_rows) and
    exact-duplicate blocks. Real live finding (4-Aug): a single VLM reply for
    ONE region sometimes repeats the same block twice (e.g. describing a
    title-block metadata table under two different framings) -- this only ever
    removes byte-identical (type, text, table_data) repeats, never distinct
    content, so it can't lose real data."""
    blocks = _dedup_table_rows(blocks)
    import json as _json
    seen: set[str] = set()
    out: list[dict] = []
    for b in blocks:
        key = _json.dumps(
            [b.get("type"), b.get("text"), b.get("table_data")],
            sort_keys=True, default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def transcribe_large_page_regions(page, config: dict, detail_prompt: str) -> list[dict]:
    """Agentic alternative to transcribe_large_page_blocks (blind grid tiling):
    locate distinct regions first (locate_regions), then zoom into each with the
    detailed extraction prompt (transcribe_regions_blocks) -- instead of
    mechanically slicing the whole sheet into a fixed grid regardless of content
    density. Falls back to blind tiling if the locate pass finds nothing."""
    regions = locate_regions(page, config)
    if not regions:
        logger.info("large_format: region locate found nothing on this page, "
                    "falling back to blind tiling")
        return transcribe_large_page_blocks(page, config, detail_prompt)

    logger.info("large_format: located %d region(s), zooming into each", len(regions))
    blocks = transcribe_regions_blocks(page, config, regions, detail_prompt)

    # Region index: one searchable block summarizing EVERY located region with its
    # own bbox + description, stored alongside the zoomed-in content blocks -- so a
    # later query (or a future "zoom into region X" agent tool) can look up what's
    # on this page and where, without re-running the locate pass.
    index_lines = [f"- [{r['label']}] ({r['type']}) {r['description']}" for r in regions]
    blocks.append({
        "type": "text",
        "text": "Regions on this sheet:\n" + "\n".join(index_lines),
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "confidence": 0.9,
        "metadata": {"kind": "region_index", "regions": regions},
    })
    return blocks
