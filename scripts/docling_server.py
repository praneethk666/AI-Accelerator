"""Docling extraction server — port 8083.

Accepts a PDF upload, runs Docling on GPU (DocLayout-YOLO + TableFormer), and returns
NormalizedBlock dicts (text / heading / table / image_caption) in reading order — the
same schema both AI-Accelerator and NEI-invoices expect from extract_docling().

Figure blocks are returned as image_caption PLACEHOLDERS with their bounding boxes and
text="[figure]". The client crops and captions them locally (it already has the PDF, and
the VLM gate is caller-side). This keeps the server stateless and avoids transferring
PNG bytes over the wire for every figure.

Tables: TableFormer by default. With table_source="auto", complex/ruled tables fall back
to pymupdf (pure geometry, free, no model) — same logic as AI-Accelerator's extract_docling.
table_source="vlm" is intentionally NOT supported here: VLM escalation is caller-side so
the server doesn't need VLM credentials or the OCR servers to be reachable from it.
Callers that need VLM table transcription receive the placeholder block with
metadata.table_complex=True and can escalate themselves.

Setup on the GPU box (own venv recommended to keep deps clean):
    python3 -m venv ~/docling_venv
    ~/docling_venv/bin/pip install \
        docling fastapi "uvicorn[standard]" python-multipart \
        pymupdf torch torchvision

Run via systemd (see docling-server.service). Port 8083.
"""
from __future__ import annotations

import gc
import io
import logging
import os
import re
import tempfile
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docling_server")

API_KEY = os.environ.get("DOCLING_API_KEY")
if not API_KEY:
    logger.warning("DOCLING_API_KEY not set — /extract is UNAUTHENTICATED.")

app = FastAPI()
_state: dict = {}
_lock = threading.Lock()

# ── block schema helpers ────────────────────────────────────────────────────

_HEADING_LABELS = {"section_header", "title", "subtitle_level_1"}
_DROP_LABELS    = {"page_header", "page_footer"}
_RUNNING_HEADER_GAP_RE = re.compile(r" {15,}")


def _with_list_marker(item, text: str) -> str:
    """Prepend an enumerated list item's own marker ("(1)", "(2)", ...) to its
    text. Docling parses this correctly (item.marker/item.enumerated) but this
    server's TextItem handling only ever read item.text, silently dropping the
    numbering every downstream numbered-step consumer (step_parser.py, client
    side) depends on. Mirrors the same fix in docling_extract.py's local-mode
    path -- found+fixed together, 11-Aug."""
    from docling_core.types.doc import ListItem
    if isinstance(item, ListItem) and getattr(item, "enumerated", False):
        marker = (getattr(item, "marker", "") or "").strip()
        if marker and not text.startswith(marker):
            return f"{marker} {text}"
    return text


def _block(document_id: str, page, filename: str, btype: str,
           text: str, table_data=None, bbox=None, metadata: dict | None = None) -> dict:
    b = {
        "block_id":   str(uuid.uuid4()),
        "document_id": document_id,
        "type":        btype,
        "text":        text,
        "table_data":  table_data,
        "source_ref":  {"filename": filename, "page": page,
                        "sheet": None, "slide": None, "bbox": bbox},
        "confidence":  0.95,
        "language":    "en",
        "metadata":    {"source": "docling_server"},
    }
    if metadata:
        b["metadata"].update(metadata)
    return b


def _is_running_header_leak(text: str, bbox) -> bool:
    if not bbox or (len(bbox) >= 2 and bbox[1] > 60):
        return False
    if not _RUNNING_HEADER_GAP_RE.search(text):
        return False
    return len(" ".join(text.split())) < 80


def _prov(item):
    prov = getattr(item, "prov", None)
    if prov:
        p = prov[0]
        return getattr(p, "page_no", None), getattr(p, "bbox", None)
    return None, None


def _bbox_topleft_pts(bbox, page_height):
    if bbox is None:
        return None
    try:
        tl = bbox.to_top_left_origin(page_height)
        l, t, r, b = float(tl.l), float(tl.t), float(tl.r), float(tl.b)
    except Exception:
        try:
            l, t, r, b = float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)
        except Exception:
            return None
    return [min(l, r), min(t, b), max(l, r), max(t, b)]


def _page_height(doc, page_no):
    try:
        if page_no in doc.pages:
            return float(doc.pages[page_no].size.height)
        first = next(iter(doc.pages.values()))
        return float(first.size.height)
    except Exception:
        return None


def _page_area(doc, page_no) -> float:
    try:
        if page_no in doc.pages:
            sz = doc.pages[page_no].size
            return float(sz.width) * float(sz.height)
        first = next(iter(doc.pages.values()))
        sz = first.size
        return float(sz.width) * float(sz.height)
    except Exception:
        return 0.0


# ── table helpers ───────────────────────────────────────────────────────────

def _table_data_from_item(table, doc):
    try:
        try:
            df = table.export_to_dataframe(doc)
        except TypeError:
            df = table.export_to_dataframe()
        if df is None or df.empty:
            return None
        headers = [str(c) for c in df.columns]
        rows = [[("" if v is None else str(v)) for v in r] for r in df.values.tolist()]
        return {"headers": headers, "rows": rows}
    except Exception:
        return None


def _table_markdown_from_item(table, doc) -> str:
    try:
        return table.export_to_markdown(doc)
    except TypeError:
        try:
            return table.export_to_markdown()
        except Exception:
            return ""
    except Exception:
        return ""


def _table_as_text(table, doc) -> str:
    td = _table_data_from_item(table, doc)
    if not td:
        return ""
    return "\n".join(
        " ".join(str(c).strip() for c in r if str(c).strip())
        for r in td.get("rows") or []
    )


def _table_is_complex(table) -> bool:
    cells = getattr(getattr(table, "data", None), "table_cells", None) or []
    if not cells:
        return False
    if any((getattr(c, "col_span", 1) or 1) > 1 or (getattr(c, "row_span", 1) or 1) > 1
           for c in cells):
        return True
    list_heavy = 0
    for c in cells:
        if getattr(c, "column_header", False):
            continue
        txt = getattr(c, "text", "") or ""
        if len(re.findall(r"(?:^|\s)\d{1,2}\.\s", txt)) >= 2 or txt.count("\n") >= 2 or len(txt) > 160:
            list_heavy += 1
            if list_heavy >= 2:
                return True
    heights, hdr_heights = [], []
    for c in cells:
        bb = getattr(c, "bbox", None)
        if bb is None:
            continue
        try:
            h = abs(float(bb.b) - float(bb.t))
        except Exception:
            continue
        heights.append(h)
        if getattr(c, "column_header", False):
            hdr_heights.append(h)
    if heights and hdr_heights:
        med = sorted(heights)[len(heights) // 2]
        if med > 0 and any(h >= 2 * med for h in hdr_heights):
            return True
    return False


def _table_has_span(table) -> bool:
    cells = getattr(getattr(table, "data", None), "table_cells", None) or []
    return any((getattr(c, "col_span", 1) or 1) > 1 or (getattr(c, "row_span", 1) or 1) > 1
               for c in cells)


def _bbox_iou(a, b) -> float:
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, bo = min(a[2], b[2]), min(a[3], b[3])
    if r <= l or bo <= t:
        return 0.0
    inter = (r - l) * (bo - t)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pymupdf_table_data(fdoc, page_no: int, bb, cache: dict, min_iou: float = 0.3):
    if fdoc is None or bb is None or not isinstance(page_no, int):
        return None
    if page_no not in cache:
        try:
            page = fdoc[page_no - 1]
            found = page.find_tables()
            cache[page_no] = [(list(t.bbox), t.extract()) for t in found.tables]
        except Exception:
            cache[page_no] = []
    best, best_iou = None, 0.0
    for tbbox, rows in cache[page_no]:
        iou = _bbox_iou(bb, tbbox)
        if iou > best_iou:
            best, best_iou = rows, iou
    if best is None or best_iou < min_iou:
        return None
    if not best or len(best) < 2:
        return None
    clean = [[("" if c is None else str(c).strip()) for c in r] for r in best]
    headers, body = clean[0], clean[1:]
    if not any(any(c for c in r) for r in body):
        return None
    return {"headers": headers, "rows": body}


def _has_blank_continuation_rows(rows: list) -> bool:
    """Ragged pymupdf result detector — same as chunk_tool.py's version."""
    if not rows:
        return False
    blank_count = sum(1 for r in rows if r and not any(str(c).strip() for c in r[1:]))
    return blank_count >= 2 and blank_count / len(rows) >= 0.25


def _render_table_markdown(td: dict | None) -> str:
    if not isinstance(td, dict):
        return ""
    headers = [str(h) for h in (td.get("headers") or [])]
    rows = td.get("rows") or []
    if not headers and not rows:
        return ""
    lines = []
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# ── model loading ───────────────────────────────────────────────────────────

def _ensure_converter(do_ocr: bool, do_table_structure: bool) -> None:
    key = (do_ocr, do_table_structure)
    if _state.get("converter_key") == key and "converter" in _state:
        return
    with _lock:
        if _state.get("converter_key") == key and "converter" in _state:
            return
        t0 = time.time()
        logger.info("Loading Docling converter (do_ocr=%s do_table=%s)...", do_ocr, do_table_structure)
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice)
        opts = PdfPipelineOptions()
        opts.do_ocr = do_ocr
        opts.do_table_structure = do_table_structure
        opts.generate_picture_images = False
        opts.images_scale = 2.0
        if do_table_structure:
            opts.table_structure_options.do_cell_matching = True
        try:
            opts.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.CUDA)
        except Exception as e:
            logger.warning("CUDA accelerator unavailable (%s), using AUTO", e)
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        _state["converter"] = conv
        _state["converter_key"] = key
        logger.info("Docling converter loaded in %.1fs", time.time() - t0)


# ── extraction logic ────────────────────────────────────────────────────────

def _extract(pdf_path: str, document_id: str, filename: str,
             table_source: str, do_ocr: bool, do_table_structure: bool,
             min_picture_pts: float) -> tuple[list[dict], int]:
    from docling_core.types.doc import TextItem, TableItem, PictureItem

    _ensure_converter(do_ocr, do_table_structure)
    conv = _state["converter"]

    try:
        import fitz
        fdoc = fitz.open(pdf_path)
        total_pages = len(fdoc)
        if table_source not in ("auto", "pymupdf"):
            fdoc.close()
            fdoc = None
    except Exception:
        fdoc = None
        total_pages = 0

    blocks: list[dict] = []
    pymupdf_cache: dict = {}

    def _process_page(pg_num: int, doc):
        ph = _page_height(doc, pg_num)
        pa = _page_area(doc, pg_num)
        for item, _level in doc.iterate_items():
            if isinstance(item, TextItem):
                label = (str(getattr(item, "label", "")) or "").lower().split(".")[-1]
                text  = (item.text or "").strip()
                if not text or label in _DROP_LABELS:
                    continue
                text = _with_list_marker(item, text)
                _, bbox = _prov(item)
                bb = _bbox_topleft_pts(bbox, ph)
                if label not in _HEADING_LABELS and _is_running_header_leak(text, bb):
                    continue
                btype = "heading" if label in _HEADING_LABELS else "text"
                blocks.append(_block(document_id, pg_num, filename, btype, text, bbox=bb))

            elif isinstance(item, TableItem):
                _, bbox = _prov(item)
                bb = _bbox_topleft_pts(bbox, ph)
                ncols = getattr(getattr(item, "data", None), "num_cols", 0) or 0
                if ncols == 1:
                    txt = _table_as_text(item, doc)
                    if txt:
                        blocks.append(_block(document_id, pg_num, filename, "text", txt, bbox=bb))
                    continue

                complex_ = _table_is_complex(item)
                has_span = _table_has_span(item)

                # pymupdf fallback for complex non-spanning tables in auto mode
                pmd_td = None
                if fdoc is not None and bb and (table_source == "pymupdf" or
                                                 (table_source == "auto" and complex_ and not has_span)):
                    pmd_td = _pymupdf_table_data(fdoc, pg_num, bb, pymupdf_cache)
                    if table_source == "auto" and pmd_td and _has_blank_continuation_rows(pmd_td.get("rows") or []):
                        pmd_td = None

                if pmd_td is not None:
                    td  = pmd_td
                    md  = _render_table_markdown(td)
                    src = "pymupdf"
                else:
                    td  = _table_data_from_item(item, doc)
                    md  = _table_markdown_from_item(item, doc)
                    src = "tableformer"

                # Mark complex tables so the client can choose to escalate to VLM/OCR
                meta = {"table_source": src, "table_complex": complex_}
                if complex_ and pmd_td is None:
                    meta["escalation_hint"] = "vlm_or_local"

                blocks.append(_block(document_id, pg_num, filename, "table",
                                     md or _render_table_markdown(td) or "[table]",
                                     table_data=td, bbox=bb, metadata=meta))

            elif isinstance(item, PictureItem):
                _, bbox = _prov(item)
                bb = _bbox_topleft_pts(bbox, ph)
                if bb:
                    w, h = bb[2] - bb[0], bb[3] - bb[1]
                    if w >= min_picture_pts and h >= min_picture_pts:
                        blocks.append(_block(document_id, pg_num, filename,
                                             "image_caption", "[figure]", bbox=bb))

    if total_pages > 0:
        for pg_num in range(1, total_pages + 1):
            try:
                res = conv.convert(pdf_path, page_range=(pg_num, pg_num))
                _process_page(pg_num, res.document)
            except Exception as e:
                logger.warning("Page %d failed: %s", pg_num, e)
    else:
        res = conv.convert(pdf_path)
        for pg_num in range(1, len(res.document.pages) + 1):
            _process_page(pg_num, res.document)
        total_pages = len(res.document.pages)

    if fdoc is not None:
        fdoc.close()

    return blocks, total_pages


# ── endpoints ───────────────────────────────────────────────────────────────

@app.post("/extract")
async def extract(
    pdf:               UploadFile = File(...),
    x_api_key:         str | None = Header(None),
    document_id:       str        = Form(""),
    filename:          str        = Form(""),
    table_source:      str        = Form("auto"),     # docling | auto | pymupdf
    do_ocr:            bool       = Form(False),
    do_table_structure: bool      = Form(True),
    min_picture_pts:   float      = Form(24.0),
):
    """Extract NormalizedBlock dicts from a PDF.

    table_source:
      "docling"  — TableFormer only (fast, good for simple ruled tables)
      "auto"     — TableFormer + pymupdf fallback for complex tables (recommended)
      "pymupdf"  — pymupdf geometry only (fastest, digital-only, no ML)

    Returns:
      {blocks: [...], n_pages: int, elapsed_s: float}

    image_caption blocks have text="[figure]" and bbox set — client crops+captions.
    Table blocks with metadata.escalation_hint="vlm_or_local" are complex; client
    may re-crop and call a VLM or OCR server for a better transcription.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing/invalid X-API-Key")
    if table_source not in ("docling", "auto", "pymupdf"):
        raise HTTPException(400, f"unknown table_source {table_source!r}")

    data = await pdf.read()
    if not data:
        raise HTTPException(400, "empty PDF upload")

    fname = filename or pdf.filename or "document.pdf"
    doc_id = document_id or str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        tmp_path = f.name

    t0 = time.time()
    try:
        blocks, n_pages = _extract(
            tmp_path, doc_id, fname, table_source,
            do_ocr, do_table_structure, min_picture_pts)
    finally:
        os.unlink(tmp_path)
        gc.collect()

    elapsed = time.time() - t0
    n_tables  = sum(1 for b in blocks if b["type"] == "table")
    n_figs    = sum(1 for b in blocks if b["type"] == "image_caption")
    logger.info("extract: %d blocks (%d tables, %d figures) across %d pages in %.1fs",
                len(blocks), n_tables, n_figs, n_pages, elapsed)
    return {"blocks": blocks, "n_pages": n_pages, "elapsed_s": elapsed}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "docling_loaded": "converter" in _state,
        "converter_key":  _state.get("converter_key"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
