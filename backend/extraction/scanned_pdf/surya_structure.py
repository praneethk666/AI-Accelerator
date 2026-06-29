"""Surya layout + table-structure recognition for scanned PDFs (GPU-friendly).

Optional, config-gated upgrade over the contour/YOLO region detector. When enabled
(ocr.layout_engine: surya), each page goes through Surya's LayoutPredictor, which
labels regions (Table / Figure / Picture / Text / ...). We then:

  - Table  -> crop + TableRecPredictor (structure: rows/cols/cells) + OCR the crop,
              map OCR lines into cells -> STRUCTURED table_data {headers, rows}
              (a real table block, like the digital PDF path — not just a caption).
  - Figure / Picture -> a visual region for vision captioning (as before).
  - Text / headers   -> ignored here (the OCR paragraph pass already covers them),
              which is what removes the text-as-visual false positives at the source.

Everything is lazy-loaded and fully guarded: if Surya/models aren't available the
caller falls back to the existing detector, so CPU deployments are unaffected.

Surya runs best on GPU; these transformer models are slow on CPU, hence the gate.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Surya layout labels we treat as captionable visuals (everything else that isn't a
# Table is text-like and handled by the OCR paragraph pass).
_FIGURE_LABELS = {"Figure", "Picture", "Chart", "Diagram"}
_TABLE_LABELS = {"Table", "TableOfContents"}

_layout_predictor = None
_table_predictor = None
_surya_unavailable = False


def surya_layout_available() -> bool:
    """Whether Surya layout/table predictors can be loaded (cached after first try)."""
    return _get_layout() is not None


def _get_layout():
    global _layout_predictor, _surya_unavailable
    if _layout_predictor is None and not _surya_unavailable:
        try:
            from surya.layout import LayoutPredictor
            _layout_predictor = LayoutPredictor()   # manager auto-created (lazy)
            logger.info("Surya LayoutPredictor loaded")
        except Exception as e:
            logger.warning("Surya layout unavailable (%s); using fallback detector", e)
            _surya_unavailable = True
    return _layout_predictor


def _get_table_rec():
    global _table_predictor
    if _table_predictor is None and not _surya_unavailable:
        try:
            from surya.table_rec import TableRecPredictor
            _table_predictor = TableRecPredictor()
            logger.info("Surya TableRecPredictor loaded")
        except Exception as e:
            logger.warning("Surya table_rec unavailable (%s); tables stay unstructured", e)
    return _table_predictor


def _bbox_of(item):
    """Return an axis-aligned [x0,y0,x1,y1] from a Surya result item (it exposes
    .bbox; fall back to the polygon's extent if only that is present)."""
    bbox = getattr(item, "bbox", None)
    if bbox and len(bbox) == 4:
        return [float(v) for v in bbox]
    poly = getattr(item, "polygon", None)
    if poly:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def classify_regions(pil_image: Image.Image) -> Optional[Tuple[list, list]]:
    """Run Surya layout on one page image. Returns (figure_bboxes, table_bboxes) in
    pixel coords, or None if layout is unavailable (caller falls back).

    figure_bboxes -> [x1,y1,x2,y2] regions to caption with vision.
    table_bboxes  -> [x1,y1,x2,y2] regions to send to recognize_table().
    """
    predictor = _get_layout()
    if predictor is None:
        return None
    try:
        result = predictor([pil_image])[0]
    except Exception as e:
        logger.warning("Surya layout inference failed: %s", e)
        return None

    figures, tables = [], []
    for region in getattr(result, "bboxes", []) or []:
        label = getattr(region, "label", "") or ""
        box = _bbox_of(region)
        if box is None:
            continue
        if label in _TABLE_LABELS:
            tables.append(box)
        elif label in _FIGURE_LABELS:
            figures.append(box)
        # Text / SectionHeader / etc. -> skip (OCR paragraph pass covers them)
    return figures, tables


def recognize_table(table_crop: Image.Image, ocr_lines: List[Tuple[str, tuple]]) -> Optional[dict]:
    """Build structured table_data {headers, rows} for one cropped table region.

    table_crop : the cropped table image (so cell + OCR coords share one frame).
    ocr_lines  : [(text, (x1,y1,x2,y2)), ...] OCR lines IN CROP COORDINATES.

    TableRecPredictor gives geometry only (rows/cols/cells) — it does not OCR — so we
    place each OCR line into the cell whose box contains the line's centre, then read
    the grid out row by row. Returns None if table_rec is unavailable or finds no cells.
    """
    predictor = _get_table_rec()
    if predictor is None:
        return None
    try:
        result = predictor([table_crop])[0]
    except Exception as e:
        logger.warning("Surya table_rec failed: %s", e)
        return None

    cells = getattr(result, "cells", None) or []
    if not cells:
        return None

    # Place each OCR line into the cell containing its centre.
    grid: dict[tuple, list] = {}
    for text, (lx1, ly1, lx2, ly2) in ocr_lines:
        cx, cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
        for c in cells:
            cb = _bbox_of(c)
            if cb and cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]:
                grid.setdefault((int(c.row_id), int(c.col_id)), []).append((lx1, text))
                break

    n_rows = max((int(c.row_id) for c in cells), default=-1) + 1
    n_cols = max((int(c.col_id) for c in cells), default=-1) + 1
    if n_rows <= 0 or n_cols <= 0:
        return None

    matrix = []
    for r in range(n_rows):
        row = []
        for col in range(n_cols):
            parts = sorted(grid.get((r, col), []), key=lambda p: p[0])  # left-to-right
            row.append(" ".join(t for _, t in parts).strip())
        matrix.append(row)

    if not any(any(cell for cell in row) for row in matrix):
        return None  # all cells empty — nothing useful recognized

    headers = matrix[0]
    rows = matrix[1:]
    return {"headers": headers, "rows": rows}


def table_data_to_markdown(table_data: dict) -> str:
    """Markdown rendering of {headers, rows} — matches the digital PDF table format."""
    headers = table_data.get("headers") or []
    rows = table_data.get("rows") or []
    header_row = "| " + " | ".join(headers) + " |" if headers else ""
    separator = "| " + " | ".join(["---"] * len(headers)) + " |" if headers else ""
    data_rows = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(p for p in [header_row, separator] + data_rows if p)
