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
import uuid
from collections import defaultdict

from backend.core.paths import display_filename

logger = logging.getLogger(__name__)

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


def _block_page(b: dict):
    ref = b.get("source_ref") or {}
    return ref.get("page") if isinstance(ref, dict) else None


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
    png = PDFCropper().crop_region(pdf_path, page_no, bbox)
    vcfg = {"vision": config.get("vision_ocr")}
    return describe_image(png, prompts.TABLE_TRANSCRIBE, vcfg).strip()


def _table_is_complex(table) -> bool:
    """Decide if a table is risky for TableFormer and should go to the VLM instead.
    Signals validated on the Argo/Mendoza corpus:
      - merged/spanning cells (row_span/col_span > 1), or
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
    for c in cells:
        if (getattr(c, "col_span", 1) or 1) > 1 or (getattr(c, "row_span", 1) or 1) > 1:
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
                    table_source: str = "docling", report: dict | None = None) -> list[dict]:
    """Convert a PDF with Docling and emit NormalizedBlock dicts in reading order.

    table_source: who transcribes tables. 'docling' (default) = TableFormer
    (free, structured); 'pymupdf' = the PDF's own vector ruling lines (free,
    geometry-based — best for ruled/digital tables TableFormer mis-segments);
    'vlm' = crop each table region and transcribe via the vision model (for kinds
    whose tables come out wrong, e.g. some scans); 'auto' = TableFormer for simple
    tables, pymupdf for complex ones (merged cells/multi-line headers), VLM only if
    pymupdf finds no ruled table there (scanned/borderless).
    report: optional dict mutated with decision counts (tables, figures) for tracking."""
    from docling_core.types.doc import TextItem, TableItem, PictureItem

    dcfg = (config.get("extraction") or {}).get("docling") or {}
    min_pic = float(dcfg.get("min_picture_pts", 24))   # drop tiny marks/logos
    min_area_frac = float(dcfg.get("min_picture_area_frac", 0.004))  # drop <0.4%-of-page icons
    table_source = (table_source or "docling").lower()
    filename = display_filename(pdf_path)
    blocks: list[dict] = []

    from backend.core.tool import check_cancelled
    check_cancelled(document_id)

    # Opened lazily only when a table_source that can use it is active; closed at the
    # end. _pymupdf_cache memoizes find_tables() per page across a page's tables.
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

    # accumulate page text so each figure is captioned WITH its page context
    page_text: dict = {}
    pic_jobs: list[tuple] = []   # defer figure crop/caption until page text is gathered
    dfigs: dict = defaultdict(list)    # docling figure boxes per page (for the YOLO union)
    dtables: dict = defaultdict(list)  # docling table boxes per page (excluded from figures)

    conv = _converter(dcfg)

    if total_pages > 0:
        if report is not None:
            report["pages"]["total"] = total_pages

        for pg_num in range(1, total_pages + 1):
            from backend.core.tool import check_cancelled
            check_cancelled(document_id)
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
                    pmd_td = None
                    if fdoc is not None and bb and (table_source == "pymupdf" or
                                                     (table_source == "auto" and complex_)):
                        pmd_td = _pymupdf_table_data(fdoc, page_no, bb, _pymupdf_cache)
                    use_pymupdf = pmd_td is not None
                    use_vlm = (not use_pymupdf) and bb and (
                        table_source == "vlm" or (table_source == "auto" and complex_))
                    source = "pymupdf" if use_pymupdf else ("vlm" if use_vlm else "tableformer")
                    if use_pymupdf:
                        td = pmd_td
                        md = _render_table_markdown(td)
                    elif use_vlm:
                        try:
                            md = _vlm_table(pdf_path, page_no, bb, config) or _table_markdown(item, doc)
                            td = None
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
                    _, bbox = _prov(item)
                    page_no = pg_num
                    bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
                    if bb and not _fig_too_small(bb, _page_area(doc, page_no), min_pic, min_area_frac):
                        idx = len(blocks)
                        blocks.append(None)
                        pic_jobs.append((idx, page_no, bb))
                        dfigs[int(page_no)].append(bb)
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
                pmd_td = None
                if fdoc is not None and bb and (table_source == "pymupdf" or
                                                 (table_source == "auto" and complex_)):
                    pmd_td = _pymupdf_table_data(fdoc, page_no, bb, _pymupdf_cache)
                use_pymupdf = pmd_td is not None
                use_vlm = (not use_pymupdf) and bb and (
                    table_source == "vlm" or (table_source == "auto" and complex_))
                source = "pymupdf" if use_pymupdf else ("vlm" if use_vlm else "tableformer")
                if use_pymupdf:
                    td = pmd_td
                    md = _render_table_markdown(td)
                elif use_vlm:
                    try:
                        md = _vlm_table(pdf_path, page_no, bb, config) or _table_markdown(item, doc)
                        td = None
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
                    pic_jobs.append((idx, page_no, bb))
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
            pic_jobs.append((idx, pg, yb))
    # Drop nested/duplicate figure crops (collage composite + its sub-photos). Dropped
    # jobs leave a None placeholder in `blocks` that the final filter removes.
    pic_jobs = _dedup_pic_jobs(pic_jobs)
    if report is not None:
        d = sum(1 for j in pic_jobs if j[0] in docling_job_ids)
        report["figures"]["docling"] += d
        report["figures"]["yolo_added"] += len(pic_jobs) - d
        report["figures"]["proposed"] = report["figures"].get("proposed", 0) + len(pic_jobs)

    # Caption figures now that page text is complete. Run CONCURRENTLY (the slow part is
    # the VLM call latency): vision.max_concurrency workers, each paced by describe_image's
    # rate limiter so we overlap latency without bursting past the provider RPM. Context is
    # copied into each worker so the per-run token-usage sink still records off-thread.
    workers = max(1, int((config.get("vision") or {}).get("max_concurrency", 1) or 1))

    def _cap(idx, page_no, bb):
        from backend.core.tool import check_cancelled
        check_cancelled(document_id)
        return idx, _figure_block(pdf_path, document_id, page_no, filename, bb,
                                  page_text.get(page_no, []), config)

    if workers > 1 and len(pic_jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from backend.core import usage as _usage
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_usage.copy_ctx().run, _cap, idx, pg, bb)
                    for idx, pg, bb in pic_jobs]
            for f in futs:
                from backend.core.tool import check_cancelled
                check_cancelled(document_id)
                idx, blk = f.result()
                blocks[idx] = blk
    else:
        for idx, page_no, bb in pic_jobs:
            from backend.core.tool import check_cancelled
            check_cancelled(document_id)
            blocks[idx] = _cap(idx, page_no, bb)[1]
    blocks = [b for b in blocks if b is not None]
    blocks = _merge_small_repeated_icons(blocks)

    # Figure totals reflect what the semantic gate KEPT (proposed - furniture dropped).
    if report is not None:
        kept = sum(1 for b in blocks if b.get("type") == "image_caption")
        report["figures"]["total"] = kept
        report["figures"]["dropped_by_gate"] = report["figures"].get("proposed", 0) - kept
        # Per-page figure ledger: proposed (pre-gate) vs kept, with the kept kinds.
        for _idx, pg, _bb in pic_jobs:
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

    logger.info("docling: %d blocks from %d pages (%d tables, %d pictures)",
                len(blocks), len(doc.pages), len(doc.tables), len(doc.pictures))
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


def _figure_block(pdf_path, document_id, page_no, filename, bbox, page_lines, config):
    """Crop a Docling-located figure, run the SEMANTIC GATE (classify + caption), and
    return the image_caption block — or None if the gate says it's page furniture
    (logo/banner/header/text/blank), so we never crop+index a non-figure. Keeps any
    real content incl. wide schematics/CAD that geometry filters would have dropped."""
    from backend.vision.pdf_cropper import PDFCropper
    import fitz
    
    # 1. Compute collision-aware padded bounding box
    padded_bbox = list(bbox)
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]
        page_rect = page.rect
        
        # Get all text block rects on the page
        rects = []
        for block in page.get_text("blocks"):
            if block[6] == 0:  # Text block type
                rects.append(fitz.Rect(block[0], block[1], block[2], block[3]))
                
        x0, y0, x1, y1 = bbox
        pad = 10.0
        
        # Left edge check
        px0 = max(page_rect.x0, x0 - pad)
        left_pad_rect = fitz.Rect(px0, y0, x0, y1)
        left_collision = any((left_pad_rect & r).get_area() > 0.1 * r.get_area() for r in rects)
        
        # Right edge check
        px1 = min(page_rect.x1, x1 + pad)
        right_pad_rect = fitz.Rect(x1, y0, px1, y1)
        right_collision = any((right_pad_rect & r).get_area() > 0.1 * r.get_area() for r in rects)
        
        # Top edge check
        py0 = max(page_rect.y0, y0 - pad)
        top_pad_rect = fitz.Rect(x0, py0, x1, y0)
        top_collision = any((top_pad_rect & r).get_area() > 0.1 * r.get_area() for r in rects)
        
        # Bottom edge check
        py1 = min(page_rect.x1, y1 + pad)
        bottom_pad_rect = fitz.Rect(x0, y1, x1, py1)
        bottom_collision = any((bottom_pad_rect & r).get_area() > 0.1 * r.get_area() for r in rects)
        
        final_pad_left = 0.0 if left_collision else pad
        final_pad_right = 0.0 if right_collision else pad
        final_pad_top = 0.0 if top_collision else pad
        final_pad_bottom = 0.0 if bottom_collision else pad
        
        if left_collision or right_collision or top_collision or bottom_collision:
            logger.info(
                "Page %s figure bbox collision detected. Capped padding (Left: %s, Right: %s, Top: %s, Bottom: %s)",
                page_no, final_pad_left, final_pad_right, final_pad_top, final_pad_bottom
            )
            
        padded_bbox = [
            max(page_rect.x0, x0 - final_pad_left),
            max(page_rect.y0, y0 - final_pad_top),
            min(page_rect.x1, x1 + final_pad_right),
            min(page_rect.y1, y1 + final_pad_bottom)
        ]
        doc.close()
    except Exception as e:
        logger.warning("docling: pad calculation failed (page %s): %s", page_no, e)
        padded_bbox = list(bbox)

    b = _block(document_id, page_no, filename, "image_caption", "[figure]", bbox=bbox)
    try:
        png = PDFCropper().crop_region(pdf_path, page_no, padded_bbox)
    except Exception as e:
        logger.warning("docling: crop failed (page %s): %s", page_no, e)
        b["metadata"]["pending_vision"] = True
        return b
    # Gate FIRST (before writing the crop to disk) so dropped furniture leaves no file.
    try:
        from backend.extraction.vision_ocr import classify_caption_crop
        res = classify_caption_crop(png, " ".join(page_lines)[:1000], config)
    except Exception as e:
        logger.warning("docling: caption failed (page %s): %s", page_no, e)
        res = {"keep": True, "kind": "unknown", "caption": "[figure]"}
        b["metadata"]["pending_vision"] = True
    if not res.get("keep"):
        return None  # logo/banner/text/decoration/blank — not a real figure
    img_dir = os.path.join("uploads", "images", document_id)
    os.makedirs(img_dir, exist_ok=True)
    fname = f"{b['block_id']}.png"
    with open(os.path.join(img_dir, fname), "wb") as f:
        f.write(png)
    b["metadata"]["image_path"] = f"/images/{document_id}/{fname}"
    b["metadata"]["pending_vision"] = b["metadata"].get("pending_vision", False)
    b["metadata"]["figure_kind"] = res.get("kind")
    b["text"] = res.get("caption") or "[figure]"
    return b
