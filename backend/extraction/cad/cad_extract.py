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
     pages we render high-DPI and TILE (backend.extraction.large_format), transcribe
     each tile, and merge — so detail survives. Normal-size pages use the single-shot
     region-JSON prompt (precise per-region boxes + table_data).

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

from backend.core import prompts
from backend.core.paths import display_filename
from backend.core.tool import PipelineState
from backend.core.vision_client import describe_image
from backend.extraction.cad.drawing_prompt import PROMPTS

logger = logging.getLogger(__name__)


def _block(document_id, page, filename, btype, text, table_data=None, bbox=None,
           confidence=0.8, metadata=None) -> dict:
    md = {"source": "cad_extract", "pending_vision": False}
    if metadata:
        md.update(metadata)
    return {
        "block_id": str(uuid.uuid4()),
        "document_id": document_id,
        "type": btype,
        "text": text,
        "table_data": table_data,
        "source_ref": {"filename": filename, "page": page,
                       "sheet": None, "slide": None, "bbox": bbox},
        "confidence": confidence,
        "language": "en",
        "metadata": md,
    }


def _vb_to_block(vb: dict, document_id, page, filename) -> dict | None:
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
    )


def _region_blocks(raw: str, document_id, page, filename) -> list[dict]:
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
        blocks = [b for b in (_vb_to_block(vb, document_id, page, filename) for vb in items) if b]

    if not blocks:
        # Salvage: parse each balanced {...} object (recovers a truncated/partial array).
        for obj in _balanced_objects(raw or ""):
            try:
                vb = _json.loads(obj)
            except Exception:
                continue
            b = _vb_to_block(vb, document_id, page, filename)
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
        cap = int(cadcfg.get("max_pages", 0) or 0)          # 0 = unlimited VLM pages
        dpi = int((config.get("vision") or {}).get("dpi", 150))

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
                    page_class = classify_page(profile_page(page), document_type)
                    if should_tile(page, page_class, config):
                        # Oversized sheet: tile so small text/designators stay legible.
                        from backend.extraction.large_format import transcribe_large_page
                        md = transcribe_large_page(page, config, prompts.SCHEMATIC_TILE)
                        from backend.extraction.vision_ocr import markdown_to_blocks
                        page_blocks = markdown_to_blocks(md, document_id, pg, filename)
                    else:
                        # Normal sheet: single-shot region-JSON extraction (precise boxes).
                        raw = describe_image(page.get_pixmap(dpi=dpi).tobytes("png"),
                                             prompt, config)
                        page_blocks = _region_blocks(raw, document_id, pg, filename)
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
