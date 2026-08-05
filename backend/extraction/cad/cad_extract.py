"""CADExtractionTool — vision extraction for CAD drawings and circuit/schematic PDFs.

CAD and circuit sheets are technical images (usually no useful text layer). This tool
renders each page and asks the VLM to extract structured regions (title block, parts /
revision / wire-list tables, drawing views, notes). It serves both routes:
    cad_route     -> document_type == "cad_drawing"      (mechanical prompt)
    circuit_route -> document_type == "circuit_diagram"  (electrical prompt)

Two things make it robust to REAL industrial drawings (validated need: Toyoda/JTEKT
ANSI-D sheets, 1000+ page schematic sets):

  1. LARGE-FORMAT TILING. An A2/A3/E-size sheet sent whole is downsampled until the tiny
     dimensions, part numbers and reference designators are illegible. For oversized
     pages we render high-DPI and either locate real regions then zoom into each
     (agentic, default) or blind-grid-tile (backend.extraction.large_format;
     extraction.large_format.strategy) — so detail survives. Normal-size pages use the
     single-shot region-JSON prompt (precise per-region boxes + table_data).

  2. FAULT TOLERANCE + COST CAP. One unreadable/garbled page must NOT kill a 1000-page
     document, so every page is wrapped: on any failure we log and skip (or fall back to
     a plain text block) instead of raising. extraction.cad.max_pages bounds VLM calls.

Critical: pending_vision is always False — this tool IS the vision step;
VisionEnrichmentTool must not re-caption these.
"""
from __future__ import annotations

import json
import logging
import os
import uuid

import fitz

from backend.core.paths import display_filename
from backend.core.tool import PipelineState
from backend.core.vision_client import describe_image
from backend.extraction.cad.drawing_prompt import PROMPTS

logger = logging.getLogger(__name__)


_MIN_BBOX_FRACTION = 0.001  # narrower than this (as a fraction of the page) is
                            # almost certainly corrupted geometry, not a real
                            # thin region
_MAX_SLIVER_ASPECT = 40     # combined with the floor above, catches boxes that
                            # ARE in-range and correctly ordered (x1<x2, y1<y2)
                            # but have collapsed to a near-zero-width sliver —
                            # e.g. width 0.00001 against height 0.08. Seen on
                            # tiled large-format pages, likely an offset/scale
                            # bug in the per-tile -> full-page bbox mapping in
                            # large_format.py; this is a defensive filter on
                            # the symptom, not a fix for that root cause.


def _is_degenerate_sliver(x1, y1, x2, y2) -> bool:
    w, h = x2 - x1, y2 - y1
    short, long_ = min(w, h), max(w, h)
    return short < _MIN_BBOX_FRACTION and (long_ / short) > _MAX_SLIVER_ASPECT


def _normalize_bbox(bbox, page_width=None, page_height=None) -> list[float] | None:
    """Validate a bbox and recover it if possible; otherwise drop it (return None)
    rather than storing unusable geometry.

    The prompt instructs the VLM to emit bboxes normalized 0.0-1.0, but on real
    pages (especially tiled large-format sheets) it sometimes leaks raw point
    coordinates instead (e.g. [96.3, 137.2, ...] against a ~600x800pt page).
    If we know the page's point dimensions we try dividing by them to recover
    a normalized box; if that still doesn't produce valid [0,1] geometry with
    x1<x2 and y1<y2, we give up and return None. A missing/None bbox is a
    normal, handled case downstream (chunking, spatial checks) — a corrupt one
    is not, so "drop it" is strictly safer than "store it."
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    def _in_unit_range(*vals):
        return all(0.0 <= v <= 1.0 for v in vals)

    if not _in_unit_range(x1, y1, x2, y2):
        if page_width and page_height:
            x1, y1, x2, y2 = x1 / page_width, y1 / page_height, x2 / page_width, y2 / page_height
            if not _in_unit_range(x1, y1, x2, y2):
                return None
        else:
            return None

    if x1 >= x2 or y1 >= y2:
        return None

    if _is_degenerate_sliver(x1, y1, x2, y2):
        return None

    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def _block(document_id, page, filename, btype, text, table_data=None, bbox=None,
           confidence=0.8, metadata=None, page_width=None, page_height=None) -> dict:
    md = {"source": "cad_extract", "pending_vision": False}
    if metadata:
        md.update(metadata)

    clean_bbox = _normalize_bbox(bbox, page_width, page_height)
    if bbox is not None and clean_bbox is None:
        # Had a bbox, but it didn't survive validation/recovery — keep the
        # block (text is still useful) but flag it and cap confidence so
        # downstream ranking doesn't trust geometry that isn't there.
        md["bbox_dropped"] = True
        confidence = min(confidence, 0.5)
        logger.warning("cad_extract: dropped invalid bbox %r for block on page %s (%s)",
                       bbox, page, md.get("label", btype))

    return {
        "block_id": str(uuid.uuid4()),
        "document_id": document_id,
        "type": btype,
        "text": text,
        "table_data": table_data,
        "source_ref": {"filename": filename, "page": page,
                       "sheet": None, "slide": None, "bbox": clean_bbox},
        "confidence": confidence,
        "language": "en",
        "metadata": md,
    }


def _vb_to_block(vb: dict, document_id, page, filename,
                 page_width=None, page_height=None) -> dict | None:
    if not isinstance(vb, dict):
        return None
    text = (vb.get("text") or "").strip()
    if not text:
        return None
    return _block(
        document_id, page, filename,
        vb.get("type") or "text", text,
        table_data=vb.get("table_data"),
        bbox=(vb.get("source_ref") or {}).get("bbox") or vb.get("bbox"),
        confidence=float(vb.get("confidence", 0.7) or 0.7),
        metadata=vb.get("metadata") if isinstance(vb.get("metadata"), dict) else None,
        page_width=page_width, page_height=page_height,
    )


def _region_blocks(raw: str, document_id, page, filename,
                   page_width=None, page_height=None) -> list[dict]:
    """Parse the VLM's region-JSON reply into blocks. Robust to the dense-page failure
    mode where the JSON array is truncated/malformed (hit the token limit): we first try
    a clean parse, then SALVAGE every complete {...} block object individually (so a cut
    array still yields its complete regions), and only as a last resort emit the cleaned
    text. A single bad page never aborts the document and never leaks JSON into a chunk."""
    import json as _json
    from backend.vision.block_builder import _extract_json, _balanced_objects, _strip_fences
    blocks: list[dict] = []

    data = _extract_json(raw)
    items = data if isinstance(data, list) else (
        data.get("blocks") if isinstance(data, dict) and isinstance(data.get("blocks"), list)
        else None)
    if items:
        blocks = [b for b in (_vb_to_block(vb, document_id, page, filename,
                                           page_width, page_height) for vb in items) if b]

    if not blocks:
        # Salvage: parse each balanced {...} object (recovers a truncated/partial array).
        for obj in _balanced_objects(raw or ""):
            try:
                vb = _json.loads(obj)
            except Exception:
                continue
            b = _vb_to_block(vb, document_id, page, filename, page_width, page_height)
            if b:
                blocks.append(b)

    if not blocks:
        # Nothing structured parsed — keep the content as plain text (fences stripped),
        # but only if it isn't itself JSON scaffolding, so no '{"type":...}' leaks in.
        txt = _strip_fences(raw).strip()
        if txt and not txt.lstrip().startswith(("[", "{")):
            blocks.append(_block(document_id, page, filename, "text", txt, confidence=0.5))
    return blocks


def _save_page_image(page, document_id, page_no, filename, dpi=150) -> dict | None:
    """Render the whole sheet to a JPEG under uploads/images/<doc>/ and return an
    image_caption block referencing it (visual grounding for the drawing). Best-effort."""
    try:
        img_dir = os.path.join("uploads", "images", document_id)
        os.makedirs(img_dir, exist_ok=True)
        bid = str(uuid.uuid4())
        pix = page.get_pixmap(dpi=dpi)
        with open(os.path.join(img_dir, f"{bid}.jpg"), "wb") as f:
            f.write(pix.tobytes("jpeg", jpg_quality=80))
        b = _block(document_id, page_no, filename, "image_caption", "[drawing]")
        b["block_id"] = bid
        b["metadata"]["image_path"] = f"/images/{document_id}/{bid}.jpg"
        return b
    except Exception as e:
        logger.debug("cad_extract: page-image save failed p%s (%s)", page_no, e)
        return None


class CADExtractionTool:
    """Vision-based extraction for CAD drawings and circuit diagrams (adaptive +
    fault-tolerant)."""

    name: str = "cad_extract"

    def run(self, state: PipelineState, config: dict) -> dict:
        file_path = state["file_path"]
        document_type = state.get("document_type") or "cad_drawing"
        document_id = state["document_id"]
        filename = display_filename(file_path)

        prompt = PROMPTS.get(document_type)
        if prompt is None:
            raise ValueError(
                f"CADExtractionTool: unsupported document_type {document_type!r}. "
                "Expected 'cad_drawing' or 'circuit_diagram'."
            )
        cadcfg = (config.get("extraction") or {}).get("cad") or {}
        
        # Shadow/override standard vision configs if custom CAD vision overrides are present
        run_config = config
        if "vision" in cadcfg:
            run_config = config.copy()
            run_config["vision"] = cadcfg["vision"]
            run_config["vision_ocr"] = cadcfg["vision"]

        cap = int(cadcfg.get("max_pages", 0) or 0)          # 0 = unlimited VLM pages
        dpi = int((run_config.get("vision") or {}).get("dpi", 200))

        from backend.extraction.page_router import profile_page, classify_page, should_tile

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"CADExtractionTool: cannot open {file_path!r}: {e}")
        if doc.page_count == 0:
            raise ValueError(f"CADExtractionTool: PDF has no pages: {file_path!r}")

        blocks: list[dict] = []
        vlm_pages = 0
        try:
            for i in range(doc.page_count):
                pg = i + 1
                page = doc[i]
                if cap and vlm_pages >= cap:
                    logger.warning("cad_extract: hit max_pages=%d cap at page %d; "
                                   "remaining pages skipped", cap, pg)
                    break
                try:
                    page_w, page_h = page.rect.width, page.rect.height
                    page_class = classify_page(profile_page(page), document_type)
                    if should_tile(page, page_class, run_config):
                        # Oversized sheet: locate real regions first, then zoom into
                        # each (agentic) -- or blind grid tiling if configured back to
                        # it. Live-tested 3-Aug: blind tiling took ~7min on one real
                        # E-size sheet and produced corrupted bboxes on a meaningful
                        # fraction of tiles (see large_format.py::_remap_bbox).
                        strategy = ((run_config.get("extraction") or {}).get("large_format") or {}).get("strategy", "agentic")
                        if strategy == "agentic":
                            from backend.extraction.large_format import transcribe_large_page_regions
                            tile_vbs = transcribe_large_page_regions(page, run_config, prompt)
                        else:
                            from backend.extraction.large_format import transcribe_large_page_blocks
                            tile_vbs = transcribe_large_page_blocks(page, run_config, prompt)
                        page_blocks = [b for b in (_vb_to_block(vb, document_id, pg, filename,
                                                                page_w, page_h) for vb in tile_vbs) if b]
                    else:
                        # Normal sheet: single-shot region-JSON extraction (precise boxes).
                        png_bytes = page.get_pixmap(dpi=dpi).tobytes("png")
                        raw = describe_image(png_bytes, prompt, run_config)
                        page_blocks = _region_blocks(raw, document_id, pg, filename, page_w, page_h)
                        from backend.extraction.large_format import _cross_check_table_cells, _dedup_blocks
                        page_blocks = _dedup_blocks(page_blocks)
                        page_blocks = _cross_check_table_cells(page_blocks, png_bytes, prompt, run_config)
                    if page_blocks:
                        vlm_pages += 1
                        blocks.extend(page_blocks)
                        img = _save_page_image(page, document_id, pg, filename, dpi=dpi)
                        if img:
                            blocks.append(img)
                    else:
                        logger.warning("cad_extract: page %d produced no blocks", pg)
                    logger.info("cad_extract: page %d/%d -> %d block(s) [%s, %s]",
                                pg, doc.page_count, len(page_blocks), document_type, page_class)
                except Exception as e:
                    # One bad page must not abort a 1000-page document.
                    logger.warning("cad_extract: page %d failed (%s); skipping", pg, e)
                    continue
        finally:
            doc.close()

        if not blocks:
            raise ValueError(
                f"CADExtractionTool: no blocks extracted from {file_path!r} "
                "(all pages failed). Check the VLM provider/key.")
        state["blocks"] = blocks
        logger.info("CADExtractionTool: wrote %d blocks for %r (%d VLM pages)",
                    len(blocks), filename, vlm_pages)
        return state