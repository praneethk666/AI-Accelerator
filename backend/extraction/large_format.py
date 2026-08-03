"""Resolution-driven tiling for large-format / diagram pages (CAD, circuit schematics,
E-size engineering sheets).

Problem: a VLM downsamples a whole large sheet to ~1.5-2k px, so tiny reference
designators (R12, C4, U3), net labels and title-block text turn to mush. Sending the
page whole therefore loses exactly the detail these documents are about.

Fix: render the page at high DPI and split it into OVERLAPPING tiles small enough to
stay legible, transcribe each tile at full detail, then merge + deduplicate. The tile
COUNT falls out of the actual page size vs the VLM's legible resolution — so a normal
A4 is 1 tile (no change) and an E-size sheet becomes e.g. a 2x3 grid automatically.
No per-document constants, no "CAD = aspect > X" hardcoding.

Config (extraction.large_format):
  enabled    : master switch (default True)
  dpi        : render DPI for tiling (default 200)
  vlm_max_px : target max tile edge in px the VLM reads cleanly (default 2600 — keeps
               A4/Letter at 1 tile, tiles only genuinely oversized sheets A3/D/E)
  overlap    : fractional tile overlap so components on a seam aren't cut (default 0.12)
  merge      : "llm" (dedup-merge via the text LLM) or "concat" (default "llm")
  max_tiles  : safety cap on tiles per page (default 24)
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
