"""Docling-based PDF extraction.

Docling (IBM, MIT) is a maintained document-AI platform: a layout model finds
text/table/figure regions and reading order, TableFormer recovers real table
structure (rows/cols/merged cells), and OCR handles scanned pages — all the things
the hand-rolled pymupdf+YOLO+VLM path approximates. It runs on CPU (M2) and
auto-accelerates on a CUDA GPU, so the same code scales from dev to a GPU box.

This module converts a PDF to Docling's `DoclingDocument`, then emits the project's
NormalizedBlock dicts (text / heading / table / image_caption) in reading order so
the rest of the pipeline (chunk -> enrich -> embed -> index) is unchanged. Figures
are located by Docling, cropped with the shared PDFCropper, then run through the
semantic gate (vision_ocr.classify_caption_crop): it classifies each crop and drops
page furniture (logos/banners/text) while captioning real figures.

Wiring: this is a normal extractor (`docling_pdf`). To use it for PDFs, point
config `pdf_extractors` (digital/scanned/mixed) at `docling_pdf`. The native tools
stay registered as a fallback — flipping the map is the whole A/B switch.
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import defaultdict

from backend.core.paths import display_filename

logger = logging.getLogger(__name__)

# Suppress noisy internal library loggers from docling, docling_core, etc.
for _noisy in [
    "docling",
    "docling.document_converter",
    "docling.pipeline",
    "docling.pipeline.standard_pdf_pipeline",
    "docling.pipeline.base_pipeline",
    "docling_core",
    "deepsearch_glm",
]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# One converter per process — building it loads the layout + TableFormer models.
_CONVERTER = None
_CONVERTER_KEY = None


def _build_converter(dcfg: dict):
    """Construct a Docling DocumentConverter from config. Cached by the settings
    that affect the model graph so a config change rebuilds it."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = bool(dcfg.get("do_ocr", True))            # OCR scanned pages
    opts.do_table_structure = bool(dcfg.get("do_table_structure", True))
    opts.generate_picture_images = False                    # we crop via PDFCropper
    opts.images_scale = float(dcfg.get("images_scale", 2.0))
    if opts.do_table_structure:
        # cell matching ties recovered structure back to PDF text cells (accurate).
        opts.table_structure_options.do_cell_matching = True

    # Accelerator: CPU on Mac dev, "cuda" on a GPU box, "mps" to try Apple GPU.
    device = (dcfg.get("device") or "auto").lower()
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        dev_map = {"auto": AcceleratorDevice.AUTO, "cpu": AcceleratorDevice.CPU,
                   "cuda": AcceleratorDevice.CUDA, "mps": AcceleratorDevice.MPS}
        opts.accelerator_options = AcceleratorOptions(
            num_threads=int(dcfg.get("num_threads", 4)),
            device=dev_map.get(device, AcceleratorDevice.AUTO))
    except Exception as e:  # older docling: accelerator config not exposed
        logger.debug("docling: accelerator options unavailable (%s)", e)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _converter(dcfg: dict):
    global _CONVERTER, _CONVERTER_KEY
    key = (dcfg.get("do_ocr", True), dcfg.get("do_table_structure", True),
           dcfg.get("images_scale", 2.0), dcfg.get("device", "auto"),
           dcfg.get("num_threads", 4))
    if _CONVERTER is None or key != _CONVERTER_KEY:
        _CONVERTER = _build_converter(dcfg)
        _CONVERTER_KEY = key
    return _CONVERTER


def _block(document_id, page, filename, btype, text, table_data=None, bbox=None) -> dict:
    return {
        "block_id": str(uuid.uuid4()),
        "document_id": document_id,
        "type": btype,
        "text": text,
        "table_data": table_data,
        "source_ref": {"filename": filename, "page": page,
                       "sheet": None, "slide": None, "bbox": bbox},
        "confidence": 0.95,
        "language": "en",
        "metadata": {"source": "docling"},
    }


# Docling text labels -> our block type. Headings get merged into the next text
# block by chunk_tool; structural furniture (page header/footer) is dropped.
_HEADING_LABELS = {"section_header", "title", "subtitle_level_1"}
_DROP_LABELS = {"page_header", "page_footer"}

_RUNNING_HEADER_GAP_RE = re.compile(r" {15,}")


def _is_running_header_leak(text: str, bbox: list[float] | None) -> bool:
    """A running page-header Docling didn't label page_header/page_footer (so
    _DROP_LABELS never catches it) — real finding, 27-Jul, on the servo manual:
    a two-column running header (chapter number on the left, chapter title on the
    right, joined by a run of literal spaces spanning most of the page width)
    surfaced as ordinary body TEXT on 23 of 105 pages, producing a near-empty
    noise chunk whenever it happened to sit right before a table/figure that
    forced a flush. Generic signature, not tied to this document's specific
    chapter titles: near the top margin AND an abnormally large single run of
    literal spaces relative to how little real content there is — a genuine
    sentence doesn't have 15+ consecutive spaces in the middle of it."""
    if not bbox or bbox[1] > 60:
        return False
    if not _RUNNING_HEADER_GAP_RE.search(text):
        return False
    return len(" ".join(text.split())) < 80


def _block_page(b: dict):
    ref = b.get("source_ref") or {}
    return ref.get("page") if isinstance(ref, dict) else None


def _update_page_progress(document_id: str, tool_name: str, current_page: int, total_pages: int) -> None:
    """Helper to update PostgreSQL with page-by-page progress. Safe from circular
    imports and DB issues, fallback to PostgresStore if pool isn't loaded."""
    try:
        from backend.api.main import _pool_get_conn, _pool_return_conn
        conn = _pool_get_conn()
        try:
            conn.execute(
                "UPDATE documents SET current_step = %s, updated_at = NOW() WHERE document_id::text = %s",
                (f"{tool_name} (page {current_page} of {total_pages})", document_id)
            )
        finally:
            _pool_return_conn(conn)
    except Exception:
        try:
            from backend.storage.postgres_store import PostgresStore
            pg = PostgresStore()
            try:
                pg.conn.execute(
                    "UPDATE documents SET current_step = %s, updated_at = NOW() WHERE document_id::text = %s",
                    (f"{tool_name} (page {current_page} of {total_pages})", document_id)
                )
            finally:
                pg.close()
        except Exception:
            pass


def _checkpoint_page_blocks(document_id: str, page_no: int, page_blocks: list[dict]) -> None:
    """Persist one page's worth of TEXT/TABLE blocks immediately (see call site for why).
    Best-effort: a checkpoint failure must never abort real extraction progress, so
    every error is swallowed here exactly like _update_page_progress above."""
    if not page_blocks:
        return
    try:
        from backend.storage.postgres_store import PostgresStore
        pg = PostgresStore()
        try:
            pg.write_page_blocks(document_id, page_no, page_blocks)
        finally:
            pg.close()
    except Exception as e:
        logger.debug("docling: checkpoint failed (page %s): %s", page_no, e)


def _pp(report: dict | None, page) -> dict | None:
    """Get/create the per-page action record in the report (None if no report or page).
    Records what we did to each page so the decision trail is queryable per page."""
    if report is None or not isinstance(page, int):
        return None
    per = report.setdefault("per_page", {})
    rec = per.get(str(page))
    if rec is None:
        rec = {"page": page, "route": "digital", "reason": None,
               "tables": {"tableformer": 0, "vlm": 0},
               "figures": {"proposed": 0, "kept": 0, "dropped": 0, "kinds": []}}
        per[str(page)] = rec
    return rec


def _table_data(table, doc):
    """TableItem -> {headers, rows} via the exported dataframe; None if empty."""
    try:
        try:
            df = table.export_to_dataframe(doc)        # newer API wants doc
        except TypeError:
            df = table.export_to_dataframe()
        if df is None or df.empty:
            return None
        headers = [str(c) for c in df.columns]
        rows = [[("" if v is None else str(v)) for v in r] for r in df.values.tolist()]
        return {"headers": headers, "rows": rows}
    except Exception as e:
        logger.debug("docling: table->dataframe failed (%s)", e)
        return None


def _table_as_text(table, doc) -> str:
    """Flatten a (1-column) 'table' to plain text lines — a single-column TableFormer
    result is really a list / table-of-contents, not a table, so we emit it as text
    instead of a junk 1-col table block with placeholder '0' headers."""
    td = _table_data(table, doc)
    if not td:
        return ""
    lines = []
    for r in td.get("rows") or []:
        line = " ".join(str(c).strip() for c in r if str(c).strip())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _vlm_table(pdf_path, page_no, bbox, config) -> str:
    """Crop a table region and transcribe it via the VLM (GLM). Used when a pdf-kind's
    table_source is 'vlm', or 'auto' escalates a complex table whose ruled structure
    TableFormer would mis-segment (merged headers, dense multi-line list cells)."""
    from backend.vision.pdf_cropper import PDFCropper
    from backend.core.vision_client import describe_image
    from backend.core import prompts
    logger.info("👁️ [Vision API] Page %s: Complex table detected -> Transcribing via Vision model...", page_no)
    png = PDFCropper().crop_region(pdf_path, page_no, bbox)
    vcfg = {"vision": config.get("vision_ocr")}
    res = describe_image(png, prompts.TABLE_TRANSCRIBE, vcfg).strip()
    logger.info("✅ [Vision API] Page %s: Vision table transcription completed", page_no)
    return res


def _local_table_engine(config) -> bool:
    """True when hard digital tables should escalate to the self-hosted Unlimited-OCR
    server (extraction.docling.table_engine: local) instead of the hosted VLM. Real
    finding, 27-Jul: cropping just the hard table region (not the whole page) got a
    dense nested table 100% correct where whole-page parsing dropped every value —
    same server as vision_ocr's rescue path (vision_ocr.local_endpoint), different
    escalation trigger (a specific table Docling/pymupdf can't handle, not a whole
    scanned/garbled page)."""
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    return (dcfg.get("table_engine") or "vlm") == "local"


def _local_table_ocr_engine(config) -> str:
    """Which self-hosted engine handles per-table-crop escalation, once
    _local_table_engine() has already picked "local" over "vlm". Real finding,
    27-Jul: PaddleOCR-VL-1.6 transcribed all 3 of Unlimited-OCR's hardest known
    real failures on this project's servo manual (2 decoder-length-bound OOMs
    + 1 silently-dropped-row misalignment) perfectly — see
    backend/extraction/paddleocr_vl.py. Kept as a separate config knob, not a
    hardcoded swap, so Unlimited-OCR (with its own validated OOM-retry-ladder
    and split-crop fallback) stays available with no code change if needed."""
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    return dcfg.get("local_table_ocr_engine") or "unlimited_ocr"


def _local_table(pdf_path, page_no, bbox, config) -> dict | None:
    """Crop a table region and transcribe it via a self-hosted local engine
    (picked by _local_table_ocr_engine), returning {headers, rows} DIRECTLY (no
    markdown round trip) so the rowspan denormalization survives intact.

    engine="paddleocr_vl" (see backend/extraction/paddleocr_vl.py): single
    request, no retry ladder — no OOM observed on this engine in any real
    testing so far.

    engine="unlimited_ocr" (default, see backend/extraction/unlimited_ocr.py):
    if the whole-table crop exhausts transcribe_table_local's own OOM retry
    ladder (real finding, 27-Jul: some tables OOM regardless of base_size
    because the real bottleneck is decoder/generation-length, not input
    resolution — a long, dense table needs a long output sequence, and THAT's
    what runs out of memory, not the vision encoder), split the region into
    top/bottom halves (with overlap so a row straddling the seam is still
    whole in at least one half) and transcribe each half separately — half the
    rows means half the output sequence length, which directly targets THIS
    bottleneck in a way no input-side parameter can. Only engaged as a last
    resort after the cheaper single-shot retries are exhausted."""
    from backend.vision.pdf_cropper import PDFCropper
    import httpx
    png = PDFCropper().crop_region(pdf_path, page_no, bbox)

    if _local_table_ocr_engine(config) == "paddleocr_vl":
        from backend.extraction.paddleocr_vl import transcribe_table_paddleocr_vl
        table_data = transcribe_table_paddleocr_vl(png, config)
        if table_data is not None:
            table_data = _cross_check_sparse_columns(table_data, png, config)
        return table_data

    from backend.extraction.unlimited_ocr import transcribe_table_local
    try:
        return transcribe_table_local(png, config)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 503:
            raise
        logger.warning("docling: table crop still OOM after every base_size retry "
                       "(page %s), splitting into row-halves", page_no)
        top_bbox, bottom_bbox = _split_bbox_vertically(bbox)
        top_png = PDFCropper().crop_region(pdf_path, page_no, top_bbox)
        bottom_png = PDFCropper().crop_region(pdf_path, page_no, bottom_bbox)
        top_td = transcribe_table_local(top_png, config)
        bottom_td = transcribe_table_local(bottom_png, config)
        return _merge_split_table_data(top_td, bottom_td)


def _sparse_column_indices(table_data: dict, min_empty_frac: float = 0.5,
                            max_sibling_empty_frac: float = 0.2) -> list[int]:
    """Columns that are empty in MOST rows while OTHER columns in the same table
    are consistently populated — a real, confirmed failure signature (28-Jul:
    PaddleOCR-VL dropped a 7-segment-display "Indication" icon column on 9/11
    rows of a real alarm table while every other column, including short ones
    like "Code" ("82H" etc), stayed fully populated). A naive "short values"
    heuristic would also flag that correct Code column, which is why this
    checks EMPTINESS, not value length.

    Different from unlimited_ocr.py's _has_fully_empty_column, which only
    catches a column blank in EVERY row (avoiding false positives on a table
    that's genuinely sparse everywhere) — this catches the softer, more common
    "mostly blank" pattern, gated on at least one sibling column being
    well-populated so a table that's legitimately sparse throughout (e.g. an
    optional-notes column) doesn't trigger an unnecessary cross-check call."""
    headers = table_data.get("headers") or []
    rows = table_data.get("rows") or []
    if len(headers) < 2 or len(rows) < 2:
        return []
    empty_fracs = []
    for col in range(len(headers)):
        vals = [str(r[col]).strip() if col < len(r) else "" for r in rows]
        empty_fracs.append(sum(1 for v in vals if not v) / len(vals))
    if not any(f <= max_sibling_empty_frac for f in empty_fracs):
        return []  # whole table reads sparse -- not a signal, don't flag anything
    return [i for i, f in enumerate(empty_fracs) if f >= min_empty_frac]


def _vlm_table_from_png(png_bytes: bytes, config: dict) -> str:
    """Same as _vlm_table but takes already-cropped PNG bytes directly, so a
    cross-check reads the EXACT SAME visual crop a local engine already read
    (not a fresh re-crop that could differ slightly)."""
    from backend.core.vision_client import describe_image
    from backend.core import prompts
    vcfg = {"vision": config.get("vision_ocr")}
    return describe_image(png_bytes, prompts.TABLE_TRANSCRIBE, vcfg).strip()


def _cross_check_sparse_columns(table_data: dict, png_bytes: bytes, config: dict) -> dict:
    """For columns that look suspiciously under-read (see _sparse_column_indices),
    cross-check against an independent VLM reading of the SAME crop and fill in
    values it found that the local engine missed. Never overwrites a value the
    local engine already has -- this only fills gaps, since the local engine
    is otherwise the validated-better reader (see paddleocr_vl.py).

    Matches the ensemble/consensus pattern from OCR research (correct reads
    converge, errors/misses diverge — arxiv 2504.11101): cross-checking with an
    independent model catches exactly this kind of column-specific miss. Kept
    cheap by only firing on tables that actually show the sparse-column
    signature, not on every table.

    Rows are matched by their first-column identity value (this codebase's own
    row_id convention — e.g. alarm name/code), NOT by position, since the two
    engines can segment/merge a rowspan-wrapped row differently."""
    sparse = _sparse_column_indices(table_data)
    if not sparse:
        return table_data
    try:
        vlm_md = _vlm_table_from_png(png_bytes, config)
        vlm_td = _markdown_table_data(vlm_md)
    except Exception as e:
        logger.warning("docling: sparse-column cross-check VLM call failed (%s)", e)
        return table_data
    if not vlm_td or not vlm_td.get("rows"):
        return table_data

    def norm(s):
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    local_headers = [norm(h) for h in table_data.get("headers") or []]
    vlm_headers = [norm(h) for h in vlm_td.get("headers") or []]
    local_rows = table_data.get("rows") or []
    vlm_rows = vlm_td.get("rows") or []

    vlm_by_id: dict[str, list] = {}
    for r in vlm_rows:
        if r:
            vlm_by_id.setdefault(norm(r[0]), []).append(r)

    filled = 0
    for col in sparse:
        if col >= len(local_headers) or local_headers[col] not in vlm_headers:
            continue
        vcol = vlm_headers.index(local_headers[col])
        for row in local_rows:
            if not row or col >= len(row) or row[col].strip():
                continue
            candidates = vlm_by_id.get(norm(row[0])) if row[0] else None
            if not candidates:
                continue
            val = next((str(c[vcol]).strip() for c in candidates
                        if vcol < len(c) and str(c[vcol]).strip()), None)
            if val:
                row[col] = val
                filled += 1
    if filled:
        logger.info("docling: sparse-column cross-check filled %d cell(s) from VLM reading", filled)
    return table_data


def _split_bbox_vertically(bbox: list[float], overlap_frac: float = 0.15) -> tuple[list[float], list[float]]:
    """Split a [x0, y0, x1, y1] top-left-origin bbox into top/bottom halves with a
    modest vertical overlap, so a row whose text sits right at the geometric
    midpoint still lands fully inside at least one half instead of being cut
    across both (which would corrupt or drop that row's content in either crop)."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    mid = y0 + height / 2
    overlap = height * overlap_frac
    top = [x0, y0, x1, min(y1, mid + overlap)]
    bottom = [x0, max(y0, mid - overlap), x1, y1]
    return top, bottom


def _merge_split_table_data(top: dict | None, bottom: dict | None) -> dict | None:
    """Combine two table_data dicts from a split-crop retry into one. Headers come
    from whichever half has them (the top crop's header row, normally). Rows are
    concatenated top-then-bottom; if the two halves' overlap band caused the SAME
    row to be transcribed twice (once at the bottom of the top half, once at the
    top of the bottom half), the exact-duplicate is dropped rather than shipping a
    row twice."""
    if top is None:
        return bottom
    if bottom is None:
        return top
    headers = top.get("headers") or bottom.get("headers") or []
    top_rows = top.get("rows") or []
    bottom_rows = bottom.get("rows") or []
    if top_rows and bottom_rows and top_rows[-1] == bottom_rows[0]:
        bottom_rows = bottom_rows[1:]
    return {"headers": headers, "rows": top_rows + bottom_rows}


def _table_has_span(table) -> bool:
    """Merged/spanning cells (row_span/col_span > 1) specifically — a DIFFERENT kind
    of complexity than list-heavy/tall-header cells, because it's structurally
    incompatible with pymupdf's ruled-line reading, not just risky for TableFormer.
    Real finding, 27-Jul: pymupdf reads each visually-separate ruled cell literally
    — it has no notion of "this cell visually spans 3 rows" — so a rowspan table
    that pymupdf CAN find ruled lines for still comes out ragged (the spanning
    cell's value only appears once, blank in the rows below it). Since 'auto' tries
    pymupdf first whenever it succeeds at all, this let real rowspan tables (e.g.
    the servo manual's alarm-code table, F7H-FFH) skip vlm/local escalation
    entirely even with table_engine: local configured — pymupdf "succeeded" at
    finding ruled lines, just not at representing the actual structure. See
    _table_is_complex's caller in extract_docling for how this changes routing."""
    cells = getattr(getattr(table, "data", None), "table_cells", None) or []
    return any((getattr(c, "col_span", 1) or 1) > 1 or (getattr(c, "row_span", 1) or 1) > 1
               for c in cells)


def _table_is_complex(table) -> bool:
    """Decide if a table is risky for TableFormer and should go to the VLM instead.
    Signals validated on the Argo/Mendoza corpus:
      - merged/spanning cells (row_span/col_span > 1) — see _table_has_span, or
      - a multi-line HEADER cell: bbox height >= ~2x the table's median cell height
        (these wrapped cells are where TableFormer clipped text), or
      - LIST-HEAVY DATA cells: several cells each packing multiple enumerated items
        ("1.… 2.… 3.…") or long wrapped text — e.g. troubleshooting/procedure charts,
        where TableFormer bleeds a wrapped line into the neighbouring column (validated
        on the Argo 'Trouble Shooting' table, p17). GLM transcribes these cleanly.
    Simple ruled tables trip none and stay on free TableFormer."""
    cells = getattr(getattr(table, "data", None), "table_cells", None) or []
    if not cells:
        return False
    if _table_has_span(table):
        return True
    # List-heavy DATA cells: >=2 data cells that each hold multiple enumerated items or
    # a long wrapped paragraph. One such cell can be normal; several means the table is
    # a multi-line list grid TableFormer mis-segments across columns.
    list_heavy = 0
    for c in cells:
        if getattr(c, "column_header", False):
            continue
        txt = getattr(c, "text", "") or ""
        enums = len(re.findall(r"(?:^|\s)\d{1,2}\.\s", txt))   # "1. " "2. " items (not 1.5 decimals)
        if enums >= 2 or txt.count("\n") >= 2 or len(txt) > 160:
            list_heavy += 1
            if list_heavy >= 2:
                return True
    # Multi-line HEADER cell only: data cells wrap normally (TableFormer handles them);
    # it's tall *header* cells where TableFormer clips text (validated). Compare each
    # header cell's height to the median cell height (the single-line baseline).
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


def _bbox_iou(a, b) -> float:
    """Intersection-over-union of two [l,t,r,b] top-left-origin boxes."""
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, bo = min(a[2], b[2]), min(a[3], b[3])
    if r <= l or bo <= t:
        return 0.0
    inter = (r - l) * (bo - t)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pymupdf_page_tables(fdoc, page_no: int, _cache: dict) -> list:
    """find_tables() for one page, cached per document-extraction call (a page with
    N docling tables would otherwise re-run page geometry analysis N times).
    page_no is Docling's (1-indexed); fitz pages are 0-indexed."""
    if page_no not in _cache:
        try:
            page = fdoc[page_no - 1]
            found = page.find_tables()
            _cache[page_no] = [(list(t.bbox), t.extract()) for t in found.tables]
        except Exception as e:
            logger.debug("pymupdf: find_tables failed on page %s (%s)", page_no, e)
            _cache[page_no] = []
    return _cache[page_no]


def _rows_to_table_data(rows) -> dict | None:
    """pymupdf's find_tables().extract() rows -> our {headers, rows} shape. First row
    is the header; None cells (merged-cell continuations) become ''."""
    if not rows or len(rows) < 2:
        return None
    clean = [[("" if c is None else str(c).strip()) for c in r] for r in rows]
    headers, body = clean[0], clean[1:]
    if not any(any(c for c in r) for r in body):
        return None
    return {"headers": headers, "rows": body}


def _pymupdf_table_data(fdoc, page_no, bb, cache: dict, min_iou: float = 0.3) -> dict | None:
    """Match a Docling-located table region to a pymupdf ruled-line table on the same
    page (by bbox overlap) and return its structure, or None if no good match.

    Why: pymupdf's find_tables() reads the PDF's own vector ruling lines / text
    alignment — pure geometry, no ML — so it's immune to the failure mode where
    TableFormer (Docling's table model) mis-segments merged-cell / dense multi-line
    spec tables (validated on a Toyota machine-spec manual: pymupdf reproduced a
    37-row merged-cell table cleanly where both TableFormer and VLM transcription
    garbled row/column alignment). Only works for ruled/digital tables — scanned
    pages and borderless tables have no vector lines for it to find, hence the
    caller falls back to TableFormer/VLM when this returns None."""
    if fdoc is None or bb is None or not isinstance(page_no, int):
        return None
    best, best_iou = None, 0.0
    for tbbox, rows in _pymupdf_page_tables(fdoc, page_no, cache):
        iou = _bbox_iou(bb, tbbox)
        if iou > best_iou:
            best, best_iou = rows, iou
    if best is None or best_iou < min_iou:
        return None
    return _rows_to_table_data(best)


def _table_markdown(table, doc) -> str:
    try:
        return table.export_to_markdown(doc)          # newer signature
    except TypeError:
        try:
            return table.export_to_markdown()
        except Exception:
            return ""
    except Exception:
        return ""


def _markdown_table_data(md: str) -> dict | None:
    """Best-effort table reconstruction from markdown.

    Docling can sometimes recover a readable markdown table but not a structured
    dataframe for scanned pages. When that happens we still want downstream tools
    to receive table_data instead of null, so we parse simple pipe tables here.
    """
    lines = [line.strip() for line in (md or "").splitlines() if line.strip()]
    rows: list[list[str]] = []
    for line in lines:
        if "|" not in line:
            continue
        # Pipe-table row: strip leading/trailing pipes and split cells.
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # Skip separator rows like "| --- | --- |".
        if cells and all(re.fullmatch(r"[:-]{2,}", c.replace(" ", "")) for c in cells):
            continue
        if any(c for c in cells):
            rows.append(cells)

    if len(rows) < 2:
        return None

    headers = [str(c) for c in rows[0]]
    body = rows[1:]
    if not any(any(cell for cell in row) for row in body):
        return None

    # Normalize ragged rows so the output is stable JSON.
    width = len(headers)
    norm_rows = [row + [""] * max(0, width - len(row)) for row in body]
    norm_rows = [row[:width] for row in norm_rows]
    return {"headers": headers, "rows": norm_rows}

def _render_table_markdown(td: dict | None) -> str:
    """{headers, rows} -> a markdown pipe table. Fallback text representation used
    when export_to_markdown()/the VLM gave us nothing but export_to_dataframe() (or
    the VLM's own markdown parse) DID produce structured cells — so the block's
    citable `text` never regresses to a placeholder while `table_data` is populated."""
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

def _prov(item):
    """(page_no, BoundingBox) of an item's first provenance, or (None, None)."""
    prov = getattr(item, "prov", None)
    if prov:
        p = prov[0]
        return getattr(p, "page_no", None), getattr(p, "bbox", None)
    return None, None


def _bbox_topleft_pts(bbox, page_height):
    """Docling bbox (often BOTTOMLEFT origin, PDF points) -> [l,t,r,b] top-left
    points for PDFCropper (which uses fitz top-left coordinates)."""
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
    # normalize ordering (x0<x1, y0<y1) so PDFCropper's Rect is well-formed
    return [min(l, r), min(t, b), max(l, r), max(t, b)]


def _picture_caption_info(item, doc, page_no) -> dict | None:
    """Resolve a Docling PictureItem's LINKED caption (item.captions -> TextItem refs)
    to its exact text + bbox. Docling already parses "Figure 12: ..." as its own text
    item and links it to the picture -- reading that beats guessing crop padding and
    re-deriving the caption from pixels. Returns None if the picture has no caption
    link (common for YOLO-only figures and most scanned pages, which have no text
    layer to link from)."""
    try:
        refs = list(getattr(item, "captions", None) or [])
        if not refs:
            return None
        text = (item.caption_text(doc) or "").strip()
        if not text:
            return None
        bboxes = []
        for ref in refs:
            try:
                cap_item = ref.resolve(doc)
            except Exception:
                continue
            cpage, cbbox = _prov(cap_item)
            if cpage != page_no:
                continue
            cbb = _bbox_topleft_pts(cbbox, _page_height(doc, page_no))
            if cbb:
                bboxes.append(cbb)
        if not bboxes:
            return {"text": text, "bbox": None}
        return {
            "text": text,
            "bbox": [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                     max(b[2] for b in bboxes), max(b[3] for b in bboxes)],
        }
    except Exception as e:
        logger.debug("docling: caption-link resolve failed (page %s): %s", page_no, e)
        return None


def _page_height(doc, page_no):
    try:
        if page_no in doc.pages:
            return float(doc.pages[page_no].size.height)
        first_page = next(iter(doc.pages.values()))
        return float(first_page.size.height)
    except Exception:
        return None


def _page_area(doc, page_no) -> float:
    try:
        if page_no in doc.pages:
            sz = doc.pages[page_no].size
            return float(sz.width) * float(sz.height)
        first_page = next(iter(doc.pages.values()))
        sz = first_page.size
        return float(sz.width) * float(sz.height)
    except Exception:
        return 0.0


def _fig_too_small(bb, page_area: float, min_pic: float, min_area_frac: float) -> bool:
    """Universal speck floor ONLY: skip a region whose width or height is below min_pic
    points (a rule line, bullet, or stray mark — too small to be a figure and not worth
    a VLM call). Everything else goes to the semantic gate, which decides logo vs
    banner vs real content. We deliberately DO NOT use aspect-ratio or page-area
    heuristics here: a wide circuit schematic or a long CAD strip is a real figure, and
    a large logo is not — only the model can tell those apart, not geometry.
    (page_area / min_area_frac kept in the signature for callers; unused.)"""
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    return w < min_pic or h < min_pic


def picture_boxes(pdf_path: str, config: dict) -> dict[int, list]:
    """Figure-localization ONLY: run Docling's layout model and return
    {page_no: [bbox, ...]} in top-left PDF points for cropping.

    This is the role the VLM-only extraction path uses Docling for — the VLM still
    transcribes all text + tables; Docling just says WHERE the figures are (it splits
    collages that DocLayout-YOLO merges). OCR and TableFormer are forced OFF here
    (not needed for picture detection), so it's much faster than a full convert."""
    from docling_core.types.doc import PictureItem
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    min_pic = float(dcfg.get("min_picture_pts", 24))
    conv = _converter({**dcfg, "do_ocr": False, "do_table_structure": False})
    doc = conv.convert(pdf_path).document
    out: dict[int, list] = defaultdict(list)
    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem):
            page_no, bbox = _prov(item)
            bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
            if bb and (bb[2] - bb[0]) >= min_pic and (bb[3] - bb[1]) >= min_pic:
                out[int(page_no)].append(bb)
    logger.info("docling: located figures on %d page(s)", len(out))
    return dict(out)


def _yolo_extra_boxes(pdf_path, dfigs: dict, dtables: dict, min_pic: float,
                      min_area_frac: float):
    """Yield (page, bbox) for DocLayout-YOLO FIGURES that Docling missed. Only the
    'figure'/'chart' classes are considered (tables stay with TableFormer), tiny
    icons are dropped, and any YOLO box overlapping a Docling figure OR table is
    skipped — so we add real figures Docling missed without duplicating tables."""
    try:
        import fitz
        from backend.extraction.vision_ocr import _yolo_boxes_pts, _iou
    except Exception as e:
        logger.warning("docling: YOLO union unavailable (%s)", e)
        return
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return
    try:
        for pg in range(1, len(doc) + 1):
            page_area = doc[pg - 1].rect.width * doc[pg - 1].rect.height
            existing = list(dfigs.get(pg, [])) + list(dtables.get(pg, []))  # figs + tables
            for yb in _yolo_boxes_pts(doc[pg - 1], classes={"figure", "chart"}):
                if _fig_too_small(yb, page_area, min_pic, min_area_frac):
                    continue
                area = max(1.0, (yb[2] - yb[0]) * (yb[3] - yb[1]))
                dup = False
                for db in existing:
                    ix = max(0.0, min(yb[2], db[2]) - max(yb[0], db[0]))
                    iy = max(0.0, min(yb[3], db[3]) - max(yb[1], db[1]))
                    if _iou(yb, db) > 0.3 or (ix * iy) / area > 0.6:
                        dup = True
                        break
                if not dup:
                    existing.append(yb)          # dedup among YOLO boxes too
                    yield pg, yb
    finally:
        doc.close()


def _dedup_pic_jobs(pic_jobs: list[tuple]) -> list[tuple]:
    """Drop figure crops that duplicate another on the SAME page — a box that overlaps
    a larger kept box (IoU > 0.6) or is mostly CONTAINED in it (>= 0.8). Cover collages
    make Docling emit a big composite box AND its sub-photos; without this the shared
    region gets cropped + captioned several times (validated: p1 cover dashboard was
    captioned 3x). Keep the LARGER box so no content is lost; preserve reading order."""
    try:
        from backend.extraction.vision_ocr import _iou
    except Exception:
        return pic_jobs
    by_page: dict = defaultdict(list)
    for job in pic_jobs:
        by_page[job[1]].append(job)
    keep_idx: set = set()
    for _pg, jobs in by_page.items():
        kept: list = []
        for job in sorted(jobs, key=lambda j: -((j[2][2] - j[2][0]) * (j[2][3] - j[2][1]))):
            bb = job[2]
            area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            dup = False
            for kjob in kept:
                kb = kjob[2]
                ix = max(0.0, min(bb[2], kb[2]) - max(bb[0], kb[0]))
                iy = max(0.0, min(bb[3], kb[3]) - max(bb[1], kb[1]))
                if _iou(bb, kb) > 0.6 or (ix * iy) / area > 0.8:
                    dup = True
                    break
            if not dup:
                kept.append(job)
        keep_idx.update(j[0] for j in kept)        # j[0] = unique block index
    return [j for j in pic_jobs if j[0] in keep_idx]


def extract_docling(pdf_path: str, document_id: str, config: dict,
                    table_source: str = "docling", report: dict | None = None,
                    on_page=None) -> list[dict]:
    """Convert a PDF with Docling and emit NormalizedBlock dicts in reading order.

    table_source: who transcribes tables. 'docling' (default) = TableFormer
    (free, structured); 'pymupdf' = the PDF's own vector ruling lines (free,
    geometry-based — best for ruled/digital tables TableFormer mis-segments);
    'vlm' = crop each table region and transcribe via the vision model (for kinds
    whose tables come out wrong, e.g. some scans); 'auto' = TableFormer for simple
    tables, pymupdf for complex ones (merged cells/multi-line headers), VLM only if
    pymupdf finds no ruled table there (scanned/borderless).
    report: optional dict mutated with decision counts (tables, figures) for tracking.
    on_page: optional callback(page_no, total_pages, elapsed_s) fired after each
    page finishes converting — real per-page progress for a long document, since
    Docling converts one page at a time here specifically to bound memory."""
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    if dcfg.get("mode") == "remote":
        from backend.extraction.docling_remote import extract_docling_remote
        return extract_docling_remote(
            pdf_path, document_id, config,
            table_source=table_source, report=report, on_page=on_page)

    from docling_core.types.doc import TextItem, TableItem, PictureItem
    from backend.chunking.chunk_tool import _has_blank_continuation_rows
    min_pic = float(dcfg.get("min_picture_pts", 24))   # drop tiny marks/logos
    min_area_frac = float(dcfg.get("min_picture_area_frac", 0.004))  # drop <0.4%-of-page icons
    table_source = (table_source or "docling").lower()
    filename = display_filename(pdf_path)
    blocks: list[dict] = []
    
    # Initialize variables for both remote and local modes
    page_text: dict = {}
    pic_jobs: list[tuple] = []
    dfigs: dict = defaultdict(list)
    dtables: dict = defaultdict(list)

    from backend.core.tool import check_cancelled
    mode = (dcfg.get("mode") or "local").lower()

    if mode == "remote":
        import requests
        server_url = dcfg.get("server_url")
        if not server_url or "${" in str(server_url):
            server_url = os.environ.get("DOCLING_SERVER_URL", "http://localhost:8083")
        api_key = dcfg.get("server_key")
        if not api_key or "${" in str(api_key):
            api_key = os.environ.get("DOCLING_API_KEY") or os.environ.get("DOCLING_SERVER_KEY", "")
        
        server_url = server_url.rstrip("/")
        do_ocr = bool(dcfg.get("do_ocr", False))
        do_table_structure = bool(dcfg.get("do_table_structure", True))
        
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
            
        data = {
            "document_id": document_id,
            "filename": filename,
            "table_source": table_source,
            "do_ocr": str(do_ocr).lower(),
            "do_table_structure": str(do_table_structure).lower(),
            "min_picture_pts": str(min_pic),
        }
        
        logger.info("docling: Calling remote server at %s/extract for doc %s", server_url, document_id)
        try:
            with open(pdf_path, "rb") as f:
                files = {"pdf": (filename, f, "application/pdf")}
                resp = requests.post(
                    f"{server_url}/extract",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=600,
                )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            err_msg = str(e)
            logger.error("docling: Remote server call failed: %s. Falling back to local mode.", err_msg)
            if report is not None:
                report["remote_error"] = err_msg
            mode = "local"
        else:
            logger.info("docling: Remote server call successful for doc %s", document_id)
            remote_blocks = payload.get("blocks", [])
            n_pages = payload.get("n_pages", 0)
            if report is not None:
                report["pages"]["total"] = n_pages
                report["mode"] = "remote"
                
            for b in remote_blocks:
                page_no = b.get("source_ref", {}).get("page")
                btype = b.get("type")
                if btype in ("text", "heading"):
                    text = b.get("text", "")
                    if text:
                        page_text.setdefault(page_no, []).append(text)
                        
            for b in remote_blocks:
                btype = b.get("type")
                page_no = b.get("source_ref", {}).get("page")
                bbox = b.get("source_ref", {}).get("bbox")
                
                if not b.get("block_id"):
                    b["block_id"] = str(uuid.uuid4())
                
                if btype == "image_caption":
                    if bbox:
                        idx = len(blocks)
                        blocks.append(None)
                        # Remote server only returns rendered blocks, not a DoclingDocument
                        # object -- no item.captions link available client-side here.
                        pic_jobs.append((idx, page_no, bbox, None))
                        dfigs[int(page_no)].append(bbox)
                elif btype == "table":
                    if report is not None:
                        report["tables"]["total"] += 1
                        ts = b.get("metadata", {}).get("table_source") or "tableformer"
                        report["tables"][ts] = report["tables"].get(ts, 0) + 1
                    if bbox and isinstance(page_no, int):
                        dtables[page_no].append(bbox)
                    blocks.append(b)
                else:
                    blocks.append(b)

    import time
    start_time = time.perf_counter()

    if mode == "local":
        if report is not None:
            report["mode"] = "local"
        fdoc = None
        _pymupdf_cache: dict = {}
        if table_source in ("pymupdf", "auto"):
            try:
                import fitz
                fdoc = fitz.open(pdf_path)
            except Exception as e:
                logger.warning("pymupdf: could not open %s for table extraction (%s)", pdf_path, e)

        # Count pages via fitz to run page-by-page conversion (which avoids memory issues on large files)
        total_pages = 0
        try:
            import fitz
            fdoc_temp = fitz.open(pdf_path)
            total_pages = len(fdoc_temp)
            if fdoc is None and table_source in ("pymupdf", "auto"):
                fdoc = fdoc_temp
            else:
                if fdoc is not fdoc_temp:
                    fdoc_temp.close()
        except Exception as e:
            logger.warning("Failed to open PDF to count pages: %s", e)

        logger.info("📄 [Docling PDF] Starting extraction: '%s' (%d pages) [Mode: %s | Table Source: %s]",
                    filename, total_pages, mode, table_source)

        # accumulate page text so each figure is captioned WITH its page context
        page_text = {}
        pic_jobs = []   # defer figure crop/caption until page text is gathered
        dfigs = defaultdict(list)    # docling figure boxes per page (for the YOLO union)
        dtables = defaultdict(list)  # docling table boxes per page (excluded from figures)

        conv = _converter(dcfg)

        if total_pages > 0:
            if report is not None:
                report["pages"]["total"] = total_pages

            for pg_num in range(1, total_pages + 1):
                from backend.core.tool import check_cancelled
                check_cancelled(document_id)
                _update_page_progress(document_id, "docling_pdf", pg_num, total_pages)
                _page_start_idx = len(blocks)   # checkpoint boundary -- see end of loop body
                pg_start_blocks = len(blocks)   # structured per-page logging -- see end of loop body
                pg_start_pics = len(pic_jobs)
                try:
                    res = conv.convert(pdf_path, page_range=(pg_num, pg_num))
                    doc = res.document
                except Exception as e:
                    logger.warning("docling: failed to convert page %d of %s: %s", pg_num, filename, e)
                    continue

                for item, _level in doc.iterate_items():
                    if isinstance(item, TextItem):
                        label = (str(getattr(item, "label", "")) or "").lower().split(".")[-1]
                        text = (item.text or "").strip()
                        if not text or label in _DROP_LABELS:
                            continue
                        _, bbox = _prov(item)
                        page_no = pg_num
                        bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                        if label not in _HEADING_LABELS and _is_running_header_leak(text, bb):
                            continue
                        btype = "heading" if label in _HEADING_LABELS else "text"
                        blocks.append(_block(document_id, page_no, filename, btype, text, bbox=bb))
                        page_text.setdefault(page_no, []).append(text)

                    elif isinstance(item, TableItem):
                        _, bbox = _prov(item)
                        page_no = pg_num
                        bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                        ncols = getattr(getattr(item, "data", None), "num_cols", 0) or 0
                        if ncols == 1:
                            txt = _table_as_text(item, doc)
                            if txt:
                                blocks.append(_block(document_id, page_no, filename, "text", txt, bbox=bb))
                                page_text.setdefault(page_no, []).append(txt)
                            continue

                        complex_ = _table_is_complex(item)
                        has_span = _table_has_span(item)
                        pmd_td = None
                        # In "auto" mode, skip pymupdf for SPANNING tables specifically —
                        # it reads each visually-separate ruled cell literally, with no
                        # notion of a cell spanning multiple rows, so it "succeeds" at
                        # finding ruled lines but still produces the ragged/blank-
                        # continuation pattern. vlm/local can actually see the merge.
                        # Explicit table_source: pymupdf is a deliberate override, not
                        # subject to this — only "auto"'s smart-selection is affected.
                        if fdoc is not None and bb and (table_source == "pymupdf" or
                                                         (table_source == "auto" and complex_
                                                          and not has_span)):
                            pmd_td = _pymupdf_table_data(fdoc, page_no, bb, _pymupdf_cache)
                            # Real gap found 27-Jul: has_span relies on DOCLING's own
                            # span metadata, which this exact class of dense multi-line
                            # table can leave empty even though the source table genuinely
                            # has merged cells (TableFormer's structure model garbles the
                            # content instead of flagging a span). So has_span alone can
                            # miss the case it exists to catch. Closing the loop on the
                            # ACTUAL symptom instead: if pymupdf's own result comes out
                            # ragged (chunk_tool.py's own repair-trigger detector), don't
                            # accept it — escalate to vlm/local, which reads the true
                            # rowspan structure directly (see unlimited_ocr.py's
                            # _RowspanTableParser) instead of leaning on downstream LLM
                            # repair to patch up a result we already know is ragged.
                            if (table_source == "auto" and pmd_td is not None
                                    and _has_blank_continuation_rows(pmd_td.get("rows") or [])):
                                pmd_td = None
                        use_pymupdf = pmd_td is not None
                        use_vlm = (not use_pymupdf) and bb and (
                            table_source == "vlm" or (table_source == "auto" and complex_))
                        source = "pymupdf" if use_pymupdf else ("vlm" if use_vlm else "tableformer")
                        if use_pymupdf:
                            td = pmd_td
                            md = _render_table_markdown(td)
                            logger.info("📊 [Docling Table] Page %d: Ruled table extracted via PyMuPDF vector lines", page_no)
                        elif use_vlm:
                            td = None
                            if _local_table_engine(config):
                                try:
                                    td = _local_table(pdf_path, page_no, bb, config)
                                except Exception as e:
                                    logger.warning("docling: local table engine failed (page %s): %s; "
                                                   "falling back to VLM", page_no, e)
                                if td is not None:
                                    md = _render_table_markdown(td)
                                    source = "local"
                            if td is None:
                                try:
                                    md = _vlm_table(pdf_path, page_no, bb, config) or _table_markdown(item, doc)
                                except Exception as e:
                                    logger.warning("docling: VLM table failed (page %s): %s; using TableFormer", page_no, e)
                                    source = "tableformer"
                                    md, td = _table_markdown(item, doc), _table_data(item, doc)
                        else:
                            md, td = _table_markdown(item, doc), _table_data(item, doc)
                            logger.info("📊 [Docling Table] Page %d: Structured table extracted via TableFormer (%d cols)", page_no, ncols)
                        if td is None:
                            td = _markdown_table_data(md)
                        if report is not None:
                            report["tables"]["total"] += 1
                            report["tables"][source] = report["tables"].get(source, 0) + 1
                            rec = _pp(report, page_no)
                            if rec:
                                rec["tables"][source] = rec["tables"].get(source, 0) + 1
                        if bb and isinstance(page_no, int):
                            dtables[page_no].append(bb)
                        blocks.append(_block(document_id, page_no, filename, "table",
                                             md or _render_table_markdown(td) or "[table]", table_data=td, bbox=bb))

                    elif isinstance(item, PictureItem):
                        _, bbox = _prov(item)
                        page_no = pg_num
                        bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                        if bb and not _fig_too_small(bb, _page_area(doc, page_no), min_pic, min_area_frac):
                            idx = len(blocks)
                            blocks.append(None)
                            cap_info = _picture_caption_info(item, doc, page_no)
                            pic_jobs.append((idx, page_no, bb, cap_info))
                            dfigs[int(page_no)].append(bb)

                # Checkpoint this page's TEXT/TABLE blocks NOW, not at the end of the whole
                # document. Real gap found 3-Aug: extract_docling had ZERO persistence of
                # its own (only a status-string progress label) -- everything lived in an
                # in-memory list until write_blocks() ran once after the ENTIRE step
                # finished, so a crash/kill at page 900 of 1147 lost all 900 pages, not
                # just the tail. Figure blocks are still None placeholders here (captioned
                # in a separate batched pass below) so they're excluded -- this is a safety
                # net against total loss, not a full per-page resume/skip (that would also
                # need to reconcile pic_jobs on restart, a bigger change deferred for now).
                _checkpoint_page_blocks(
                    document_id, pg_num,
                    [b for b in blocks[_page_start_idx:] if b is not None])

                pg_text_cnt = sum(1 for b in blocks[pg_start_blocks:] if b and b.get("type") in ("text", "heading"))
                pg_tbl_cnt = sum(1 for b in blocks[pg_start_blocks:] if b and b.get("type") == "table")
                pg_pic_cnt = len(pic_jobs) - pg_start_pics
                logger.info("📑 [Docling PDF] Page %d/%d: Extracted %d text block(s), %d table(s), %d figure(s)",
                            pg_num, total_pages, pg_text_cnt, pg_tbl_cnt, pg_pic_cnt)
        else:
            # Fallback to full conversion if page count could not be retrieved
            res = conv.convert(pdf_path)
            doc = res.document
            for item, _level in doc.iterate_items():
                if isinstance(item, TextItem):
                    label = (str(getattr(item, "label", "")) or "").lower().split(".")[-1]
                    text = (item.text or "").strip()
                    if not text or label in _DROP_LABELS:
                        continue
                    page_no, bbox = _prov(item)
                    bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                    if label not in _HEADING_LABELS and _is_running_header_leak(text, bb):
                        continue
                    btype = "heading" if label in _HEADING_LABELS else "text"
                    blocks.append(_block(document_id, page_no, filename, btype, text, bbox=bb))
                    page_text.setdefault(page_no, []).append(text)

                elif isinstance(item, TableItem):
                    page_no, bbox = _prov(item)
                    bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                    ncols = getattr(getattr(item, "data", None), "num_cols", 0) or 0
                    if ncols == 1:
                        txt = _table_as_text(item, doc)
                        if txt:
                            blocks.append(_block(document_id, page_no, filename, "text", txt, bbox=bb))
                            page_text.setdefault(page_no, []).append(txt)
                        continue

                    complex_ = _table_is_complex(item)
                    has_span = _table_has_span(item)
                    pmd_td = None
                    # In "auto" mode, skip pymupdf for SPANNING tables specifically —
                    # it reads each visually-separate ruled cell literally, with no
                    # notion of a cell spanning multiple rows, so it "succeeds" at
                    # finding ruled lines but still produces the ragged/blank-
                    # continuation pattern. vlm/local can actually see the merge.
                    # Explicit table_source: pymupdf is a deliberate override, not
                    # subject to this — only "auto"'s smart-selection is affected.
                    if fdoc is not None and bb and (table_source == "pymupdf" or
                                                     (table_source == "auto" and complex_
                                                      and not has_span)):
                        pmd_td = _pymupdf_table_data(fdoc, page_no, bb, _pymupdf_cache)
                        # Real gap found 27-Jul: has_span relies on DOCLING's own
                        # span metadata, which this exact class of dense multi-line
                        # table can leave empty even though the source table genuinely
                        # has merged cells (TableFormer's structure model garbles the
                        # content instead of flagging a span). So has_span alone can
                        # miss the case it exists to catch. Closing the loop on the
                        # ACTUAL symptom instead: if pymupdf's own result comes out
                        # ragged (chunk_tool.py's own repair-trigger detector), don't
                        # accept it — escalate to vlm/local, which reads the true
                        # rowspan structure directly (see unlimited_ocr.py's
                        # _RowspanTableParser) instead of leaning on downstream LLM
                        # repair to patch up a result we already know is ragged.
                        if (table_source == "auto" and pmd_td is not None
                                and _has_blank_continuation_rows(pmd_td.get("rows") or [])):
                            pmd_td = None
                    use_pymupdf = pmd_td is not None
                    use_vlm = (not use_pymupdf) and bb and (
                        table_source == "vlm" or (table_source == "auto" and complex_))
                    source = "pymupdf" if use_pymupdf else ("vlm" if use_vlm else "tableformer")
                    if use_pymupdf:
                        td = pmd_td
                        md = _render_table_markdown(td)
                    elif use_vlm:
                        td = None
                        if _local_table_engine(config):
                            try:
                                td = _local_table(pdf_path, page_no, bb, config)
                            except Exception as e:
                                logger.warning("docling: local table engine failed (page %s): %s; "
                                               "falling back to VLM", page_no, e)
                            if td is not None:
                                md = _render_table_markdown(td)
                                source = "local"
                        if td is None:
                            try:
                                md = _vlm_table(pdf_path, page_no, bb, config) or _table_markdown(item, doc)
                            except Exception as e:
                                logger.warning("docling: VLM table failed (page %s): %s; using TableFormer", page_no, e)
                                source = "tableformer"
                                md, td = _table_markdown(item, doc), _table_data(item, doc)
                    else:
                        md, td = _table_markdown(item, doc), _table_data(item, doc)
                    if td is None:
                        td = _markdown_table_data(md)
                    if report is not None:
                        report["tables"]["total"] += 1
                        report["tables"][source] = report["tables"].get(source, 0) + 1
                        rec = _pp(report, page_no)
                        if rec:
                            rec["tables"][source] = rec["tables"].get(source, 0) + 1
                    if bb and isinstance(page_no, int):
                        dtables[page_no].append(bb)
                    blocks.append(_block(document_id, page_no, filename, "table",
                                         md or _render_table_markdown(td) or "[table]", table_data=td, bbox=bb))

                elif isinstance(item, PictureItem):
                    page_no, bbox = _prov(item)
                    bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                    if bb and not _fig_too_small(bb, _page_area(doc, page_no), min_pic, min_area_frac):
                        idx = len(blocks)
                        blocks.append(None)
                        cap_info = _picture_caption_info(item, doc, page_no)
                        pic_jobs.append((idx, page_no, bb, cap_info))
                        dfigs[int(page_no)].append(bb)

        if fdoc is not None:
            fdoc.close()

    # Figure UNION: add DocLayout-YOLO boxes that Docling missed (tested: Docling splits
    # collages but misses some figures; YOLO catches those). Skip YOLO boxes that overlap
    # a Docling box. Both free/CPU. Disable with extraction.docling.figure_union: false.
    n_docling_figs = len(pic_jobs)
    docling_job_ids = {pic_jobs[i][0] for i in range(n_docling_figs)}
    if dcfg.get("figure_union", True):
        for pg, yb in _yolo_extra_boxes(pdf_path, dfigs, dtables, min_pic, min_area_frac):
            idx = len(blocks)
            blocks.append(None)
            # YOLO-only detections have no Docling item to link a caption from.
            pic_jobs.append((idx, pg, yb, None))
    # Drop nested/duplicate figure crops (collage composite + its sub-photos). Dropped
    # jobs leave a None placeholder in `blocks` that the final filter removes.
    pic_jobs = _dedup_pic_jobs(pic_jobs)
    if report is not None:
        d = sum(1 for j in pic_jobs if j[0] in docling_job_ids)
        report["figures"]["docling"] += d
        report["figures"]["yolo_added"] += len(pic_jobs) - d
        report["figures"]["proposed"] = report["figures"].get("proposed", 0) + len(pic_jobs)

    # Every kept figure box on the SAME page, keyed by its own job idx so a figure
    # never excludes itself -- fed into _figure_block as exclusion zones so one
    # figure's padding can't bleed into a neighboring figure a few points away
    # (real bug on a scanned doc with no text layer at all: the collision guard
    # below has nothing else to check against there, so this matters most on scans,
    # but it's a strict improvement on digital pages too).
    page_to_boxes: dict = defaultdict(list)
    for j_idx, pg, bb, _ci in pic_jobs:
        page_to_boxes[pg].append((j_idx, bb))

    # Pages where Docling extracted NO text at all (do_ocr:false + no text layer --
    # i.e. scanned, as far as Docling itself can tell) get their figure captioning
    # DEFERRED to caption_deferred_figures(), called after route_and_rescue() has
    # produced the page's real OCR/VLM text -- otherwise the semantic gate runs
    # totally blind (no page context) on exactly the pages that need it most. Only
    # safe to defer if a later rescue pass will actually run to resolve it.
    defer_ok = bool(dcfg.get("page_rescue", True))

    # Size-based lazy figure captioning: real cost driver is FIGURE COUNT, not page
    # count (plain text/table extraction is ~free regardless of length -- validated
    # live, 3-Aug: a 1147-page text-only circuit diagram is cheap; a 50-page
    # CAD-drawing-heavy manual can cost more). Above defer_figures_above_pages,
    # figures on pages that DO have text also get deferred -- but PERMANENTLY (no
    # route_and_rescue pass will ever resolve these; they stay deferred until an
    # agent looks at that specific page on demand via view_page_image), unlike the
    # scanned-page case above which auto-resolves once real OCR text exists.
    figure_mode = dcfg.get("figure_caption_mode", "eager")
    lazy_threshold = int(dcfg.get("defer_figures_above_pages", 250) or 250)
    size_lazy = figure_mode == "size_based" and total_pages > lazy_threshold

    # Caption figures now that page text is complete. Run CONCURRENTLY (the slow part is
    # the VLM call latency): vision.max_concurrency workers, each paced by describe_image's
    # rate limiter so we overlap latency without bursting past the provider RPM. Context is
    # copied into each worker so the per-run token-usage sink still records off-thread.
    workers = max(1, int((config.get("vision") or {}).get("max_concurrency", 1) or 1))

    def _cap(idx, page_no, bb, cap_info):
        from backend.core.tool import check_cancelled
        check_cancelled(document_id)
        others = [b for j_idx, b in page_to_boxes.get(page_no, []) if j_idx != idx]
        no_text = not page_text.get(page_no)
        defer_reason = "scanned_no_text" if no_text else ("large_document_lazy" if size_lazy else None)
        defer = defer_ok and defer_reason is not None
        return idx, _figure_block(pdf_path, document_id, page_no, filename, bb,
                                  page_text.get(page_no, []), config, cap_info=cap_info,
                                  other_figure_bboxes=others, defer=defer,
                                  defer_reason=defer_reason)

    if workers > 1 and len(pic_jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from backend.core import usage as _usage
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_usage.copy_ctx().run, _cap, idx, pg, bb, cap_info)
                    for idx, pg, bb, cap_info in pic_jobs]
            for f in futs:
                from backend.core.tool import check_cancelled
                check_cancelled(document_id)
                idx, blk = f.result()
                blocks[idx] = blk
    else:
        for idx, page_no, bb, cap_info in pic_jobs:
            from backend.core.tool import check_cancelled
            check_cancelled(document_id)
            blocks[idx] = _cap(idx, page_no, bb, cap_info)[1]
    blocks = [b for b in blocks if b is not None]
    blocks = _merge_small_repeated_icons(blocks)

    # Figure totals reflect what the semantic gate KEPT (proposed - furniture dropped).
    if report is not None:
        kept = sum(1 for b in blocks if b.get("type") == "image_caption")
        report["figures"]["total"] = kept
        report["figures"]["dropped_by_gate"] = report["figures"].get("proposed", 0) - kept
        # Per-page figure ledger: proposed (pre-gate) vs kept, with the kept kinds.
        for _idx, pg, _bb, _cap_info in pic_jobs:
            rec = _pp(report, pg)
            if rec:
                rec["figures"]["proposed"] += 1
        for b in blocks:
            if b.get("type") == "image_caption":
                rec = _pp(report, _block_page(b))
                if rec:
                    rec["figures"]["kept"] += 1
                    k = (b.get("metadata") or {}).get("figure_kind")
                    if k:
                        rec["figures"]["kinds"].append(k)
        for rec in report.get("per_page", {}).values():
            rec["figures"]["dropped"] = rec["figures"]["proposed"] - rec["figures"]["kept"]

    elapsed = time.perf_counter() - start_time
    n_text = sum(1 for b in blocks if b.get("type") in ("text", "heading"))
    n_tables = sum(1 for b in blocks if b.get("type") == "table")
    n_figs = sum(1 for b in blocks if b.get("type") == "image_caption")
    logger.info("✨ [Docling PDF] Completed '%s': %d total blocks (%d text, %d tables, %d figures) in %.2fs",
                filename, len(blocks), n_text, n_tables, n_figs, elapsed)
    return blocks


def _merge_small_repeated_icons(blocks: list[dict], icon_pt: float = 40.0, min_group: int = 3) -> list[dict]:
    """Small warning/legend icons are real, correctly-classified figures (the VLM gate
    already keeps them, not page furniture) — but a safety-warnings page can carry 6-9
    of them (validated live on a real Toyota manual: diamond/triangle hazard symbols,
    each ~25-32pt), and indexing each as its own near-duplicate image_caption chunk
    fragments retrieval without adding distinct value — a query about 'fire hazard
    warning' would get several nearly-identical hits instead of one useful one.

    When a page has >= min_group image_caption blocks that are ALL under icon_pt in
    both dimensions, merge them into ONE combined block for that page (keeping every
    caption's content, just as one chunk) instead of dropping any of them — this is
    about de-fragmenting redundant real content, not filtering false positives."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        if b.get("type") != "image_caption":
            continue
        ref = b.get("source_ref") or {}
        bbox = ref.get("bbox") or []
        if len(bbox) == 4 and (bbox[2] - bbox[0]) <= icon_pt and (bbox[3] - bbox[1]) <= icon_pt:
            by_page[ref.get("page")].append(i)

    drop: set[int] = set()
    for page, idxs in by_page.items():
        if len(idxs) < min_group:
            continue
        captions = [(blocks[i].get("text") or "").strip() for i in idxs]
        merged_text = (
            f"This page has {len(idxs)} small warning/legend icons: "
            + " | ".join(c for c in captions if c)
        )
        merged = dict(blocks[idxs[0]])
        merged["text"] = merged_text
        merged["metadata"] = dict(merged.get("metadata") or {})
        merged["metadata"]["merged_icon_count"] = len(idxs)
        blocks[idxs[0]] = merged
        drop.update(idxs[1:])
        logger.info("docling: merged %d small icons on page %s into one chunk", len(idxs), page)
    return [b for i, b in enumerate(blocks) if i not in drop]


def _figure_block(pdf_path, document_id, page_no, filename, bbox, page_lines, config,
                  cap_info: dict | None = None, other_figure_bboxes: list | None = None,
                  defer: bool = False, defer_reason: str | None = None):
    """Crop a Docling-located figure, run the SEMANTIC GATE (classify + caption), and
    return the image_caption block — or None if the gate says it's page furniture
    (logo/banner/header/text/blank), so we never crop+index a non-figure. Keeps any
    real content incl. wide schematics/CAD that geometry filters would have dropped.

    cap_info (optional): {"text": ..., "bbox": [l,t,r,b] | None} from
    _picture_caption_info() -- Docling's OWN linked caption for this figure, when it
    has one. When present we union its bbox into the crop region (so the real caption
    is guaranteed inside the image instead of relying on guessed padding) and pass its
    exact text to the VLM gate as ground truth, then hard-append it to whatever caption
    the model returns so the real wording (part numbers, figure number) is never lost
    even if the model paraphrases or ignores it.

    other_figure_bboxes (optional): every OTHER kept figure's bbox on this same page --
    treated as collision zones (same as nearby text) so padding never grows into a
    neighboring figure. Matters most on scanned pages, which have no PDF text layer at
    all, so the text-collision check below has nothing else to guard against there.

    defer: True skips the VLM gate entirely -- crops and saves the image, marks the
    block metadata.caption_deferred=True, and returns a placeholder. Used for pages
    Docling extracted NO text from (do_ocr:false + no text layer = scanned), so the
    gate isn't run blind; caption_deferred_figures() resolves it later once
    route_and_rescue() has produced the page's real OCR/VLM text."""
    from backend.vision.pdf_cropper import PDFCropper
    import fitz

    # 0. Fold the linked caption's own bbox into the crop region FIRST so the real
    # caption text is guaranteed to be physically inside the crop -- padding below is
    # then just breathing room, not the only thing standing between us and a cut caption.
    crop_bbox = list(bbox)
    if cap_info and cap_info.get("bbox"):
        cb = cap_info["bbox"]
        crop_bbox = [min(crop_bbox[0], cb[0]), min(crop_bbox[1], cb[1]),
                     max(crop_bbox[2], cb[2]), max(crop_bbox[3], cb[3])]

    # 1. Compute collision-aware padded bounding box
    padded_bbox = list(crop_bbox)
    # PDFCropper.crop_region() ALSO adds its own (larger) unconditional padding on
    # top of padded_bbox (side_frac/top_frac/bottom_frac/caption_pad_pts, ~52pt on
    # the bottom by default) -- collision detection below is pointless unless THAT
    # padding is suppressed too on any edge where a collision was found, so this
    # gets filled in per-edge and passed through to the crop_region() call.
    crop_kwargs: dict = {}
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]
        page_rect = page.rect

        # Get all text block rects on the page, PLUS every other kept figure on this
        # page -- a scanned page has zero text rects (no text layer), so without the
        # figure boxes here padding has nothing at all to guard against and can bleed
        # straight into a neighboring figure a few points away.
        rects = []
        for block in page.get_text("blocks"):
            if block[6] == 0:  # Text block type
                rects.append(fitz.Rect(block[0], block[1], block[2], block[3]))
        for ob in (other_figure_bboxes or []):
            rects.append(fitz.Rect(ob[0], ob[1], ob[2], ob[3]))

        # Collision = the overlap covers >10% of whichever of the two rects is
        # SMALLER (pad strip vs the other rect). Using only the OTHER rect's area
        # (as this used to) works for small text words but is far too weak against
        # a full-size neighboring FIGURE: a thin 10pt pad strip fully swallowed by a
        # large figure can still be under 10% of that figure's total area, so a real
        # collision would go undetected. min() catches both cases.
        def _collides(pad_rect) -> bool:
            return any((pad_rect & r).get_area() > 0.1 * min(r.get_area(), pad_rect.get_area())
                       for r in rects)

        x0, y0, x1, y1 = crop_bbox
        pad = 10.0

        # Left edge check
        px0 = max(page_rect.x0, x0 - pad)
        left_collision = _collides(fitz.Rect(px0, y0, x0, y1))

        # Right edge check
        px1 = min(page_rect.x1, x1 + pad)
        right_collision = _collides(fitz.Rect(x1, y0, px1, y1))

        # Top edge check
        py0 = max(page_rect.y0, y0 - pad)
        top_collision = _collides(fitz.Rect(x0, py0, x1, y0))

        # Bottom edge check
        py1 = min(page_rect.y1, y1 + pad)  # was page_rect.x1 (width, not height) -- fixed
        bottom_collision = _collides(fitz.Rect(x0, y1, x1, py1))
        
        final_pad_left = 0.0 if left_collision else pad
        final_pad_right = 0.0 if right_collision else pad
        final_pad_top = 0.0 if top_collision else pad
        final_pad_bottom = 0.0 if bottom_collision else pad
        
        if left_collision or right_collision or top_collision or bottom_collision:
            logger.info(
                "Page %s figure bbox collision detected. Capped padding (Left: %s, Right: %s, Top: %s, Bottom: %s)",
                page_no, final_pad_left, final_pad_right, final_pad_top, final_pad_bottom
            )
            # crop_region()'s side padding is one symmetric value (no separate
            # left/right knob) -- a collision on EITHER side suppresses both,
            # trading a little missed side-callout margin for guaranteed no bleed.
            if left_collision or right_collision:
                crop_kwargs["side_frac"] = 0.0
                crop_kwargs["side_pad_pts"] = 0.0
            if top_collision:
                crop_kwargs["top_frac"] = 0.0
            if bottom_collision:
                crop_kwargs["bottom_frac"] = 0.0
                crop_kwargs["caption_pad_pts"] = 0.0

        padded_bbox = [
            max(page_rect.x0, x0 - final_pad_left),
            max(page_rect.y0, y0 - final_pad_top),
            min(page_rect.x1, x1 + final_pad_right),
            min(page_rect.y1, y1 + final_pad_bottom)
        ]
        doc.close()
    except Exception as e:
        logger.warning("docling: pad calculation failed (page %s): %s", page_no, e)
        padded_bbox = list(crop_bbox)

    b = _block(document_id, page_no, filename, "image_caption", "[figure]", bbox=bbox)
    try:
        png = PDFCropper().crop_region(pdf_path, page_no, padded_bbox, **crop_kwargs)
    except Exception as e:
        logger.warning("docling: crop failed (page %s): %s", page_no, e)
        b["metadata"]["pending_vision"] = True
        return b

    if defer:
        # No gate decision yet -- can't drop furniture we haven't classified, so save
        # the crop unconditionally (caption_deferred_figures cleans up any that turn
        # out to be furniture once it finally runs the gate with real page context).
        img_dir = os.path.join("uploads", "images", document_id)
        os.makedirs(img_dir, exist_ok=True)
        fname = f"{b['block_id']}.png"
        with open(os.path.join(img_dir, fname), "wb") as f:
            f.write(png)
        b["metadata"]["image_path"] = f"/images/{document_id}/{fname}"
        b["metadata"]["pending_vision"] = True
        b["metadata"]["caption_deferred"] = True
        b["metadata"]["defer_reason"] = defer_reason
        # "large_document_lazy" figures are NEVER auto-resolved during ingestion
        # (see caption_deferred_figures()) -- give them real, useful text now so
        # they're still findable via normal search instead of sitting as a bare
        # "[figure]" forever. view_page_image is how an agent actually looks at
        # one on demand.
        if defer_reason == "large_document_lazy":
            b["text"] = ("[Figure present on this page -- not yet captioned "
                         "(large document, captions load on demand). Use "
                         "view_page_image on this page to see it.]")
        return b

    known_caption = (cap_info or {}).get("text") or ""
    # Gate FIRST (before writing the crop to disk) so dropped furniture leaves no file.
    logger.info("👁️ [Vision API] Page %s: Captioning figure crop (bbox: %s) via Vision model...",
                page_no, [round(float(x), 1) for x in bbox] if bbox else [])
    try:
        from backend.extraction.vision_ocr import classify_caption_crop
        res = classify_caption_crop(png, " ".join(page_lines)[:1000], config,
                                    known_caption=known_caption)
    except Exception as e:
        logger.warning("docling: caption failed (page %s): %s", page_no, e)
        res = {"keep": True, "kind": "unknown", "caption": known_caption or "[figure]"}
        b["metadata"]["pending_vision"] = True
    if not res.get("keep"):
        logger.info("🗑️ [Semantic Gate] Page %s: Filtered non-informative element (kind: %s)",
                    page_no, res.get("kind", "unknown"))
        return None  # logo/banner/text/decoration/blank — not a real figure

    cap_snippet = (res.get("caption") or "[figure]").strip().replace("\n", " ")
    if len(cap_snippet) > 80:
        cap_snippet = cap_snippet[:77] + "..."
    logger.info("✅ [Vision API] Page %s: Caption generated [%s]: \"%s\"",
                page_no, res.get("kind", "unknown"), cap_snippet)

    img_dir = os.path.join("uploads", "images", document_id)
    os.makedirs(img_dir, exist_ok=True)
    fname = f"{b['block_id']}.png"
    with open(os.path.join(img_dir, fname), "wb") as f:
        f.write(png)
    b["metadata"]["image_path"] = f"/images/{document_id}/{fname}"
    b["metadata"]["pending_vision"] = b["metadata"].get("pending_vision", False)
    b["metadata"]["figure_kind"] = res.get("kind")
    caption = res.get("caption") or "[figure]"
    # Hard-guarantee the real caption text (exact figure number, part numbers) is in the
    # searchable text even if the model paraphrased or dropped it -- this is the one piece
    # we KNOW is correct (Docling's text layer), not a VLM guess.
    if known_caption:
        b["metadata"]["docling_caption"] = known_caption
        if known_caption.lower() not in caption.lower():
            caption = f"{known_caption} — {caption}"
    b["text"] = caption
    return b


def caption_deferred_figures(blocks: list[dict], config: dict) -> list[dict]:
    """Resolve figures _figure_block() deferred (metadata.caption_deferred=True --
    Docling saw no text on that page at crop-time, so the gate would've run blind).
    Call this AFTER route_and_rescue() has produced the page's real OCR/VLM text, so
    the semantic gate finally gets real context instead of guessing from pixels alone.
    No-op (returns blocks unchanged) if nothing was deferred. Runs concurrently with
    the same worker budget as the original captioning pass; a per-figure failure
    leaves it pending (self-healing on a future re-run) rather than failing the batch.

    Deliberately SKIPS defer_reason="large_document_lazy" figures -- those were
    deferred as a permanent cost-control policy (size-based lazy captioning, see
    extract_docling), not because they're waiting on OCR text that's now available.
    They stay deferred until an agent looks at that page on demand (view_page_image)."""
    pending = [b for b in blocks if b and b.get("type") == "image_caption"
               and (b.get("metadata") or {}).get("caption_deferred")
               and (b.get("metadata") or {}).get("defer_reason") != "large_document_lazy"]
    if not pending:
        return blocks

    page_text: dict = defaultdict(list)
    for b in blocks:
        if b and b.get("type") in ("text", "heading"):
            t = (b.get("text") or "").strip()
            if t:
                page_text[_block_page(b)].append(t)

    def _resolve(b: dict) -> None:
        page_no = _block_page(b)
        img_path = (b.get("metadata") or {}).get("image_path") or ""
        parts = img_path.split("/")
        if len(parts) < 4:   # expects "/images/<document_id>/<fname>.png"
            return
        fs_path = os.path.join("uploads", "images", *parts[2:])
        try:
            with open(fs_path, "rb") as f:
                png = f.read()
        except Exception as e:
            logger.warning("docling: deferred-caption image read failed (%s): %s", fs_path, e)
            return
        try:
            from backend.extraction.vision_ocr import classify_caption_crop
            res = classify_caption_crop(png, " ".join(page_text.get(page_no, []))[:1000], config)
        except Exception as e:
            logger.warning("docling: deferred caption failed (page %s): %s", page_no, e)
            return
        if not res.get("keep"):
            b["_drop"] = True
            try:
                os.remove(fs_path)
            except Exception:
                pass
            return
        b["metadata"]["figure_kind"] = res.get("kind")
        b["metadata"]["pending_vision"] = False
        b["metadata"]["caption_deferred"] = False
        b["text"] = res.get("caption") or "[figure]"

    workers = max(1, int((config.get("vision") or {}).get("max_concurrency", 1) or 1))
    if workers > 1 and len(pending) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from backend.core import usage as _usage
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_usage.copy_ctx().run, _resolve, b) for b in pending]
            for f in futs:
                f.result()
    else:
        for b in pending:
            _resolve(b)

    return [b for b in blocks if not b.get("_drop")]
