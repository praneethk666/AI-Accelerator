"""
PyMuPDF PDF Extractor

Extracts text blocks, structured tables, embedded images, and hyperlinks from PDFs
using PyMuPDF (fitz). Normalizes all outputs directly into NormalizedBlock dicts
fully compliant with backend/core/schemas.py.
"""

import uuid
import logging
from pathlib import Path
import fitz  # PyMuPDF

from backend.core.schemas import NormalizedBlock, SourceRef, as_dicts
from backend.extraction.table_reconcile import reconcile_tables

logger = logging.getLogger(__name__)


def extract_pymupdf(pdf_path: str, document_id: str, config: dict | None = None) -> list[dict]:
    """Extract PDF data using PyMuPDF and output plain dict NormalizedBlock objects."""
    logger.info("Running PyMuPDF extraction on: %s", pdf_path)
    filename = Path(pdf_path).name
    doc = fitz.open(pdf_path)

    blocks: list[dict] = []

    for page_idx, page in enumerate(doc, start=1):
        table_rects = []
        # 1. Extract tables
        try:
            tab_finder = page.find_tables()
            if tab_finder and getattr(tab_finder, "tables", None):
                for t_idx, table in enumerate(tab_finder.tables, start=1):
                    table_rects.append(fitz.Rect(table.bbox))
                    extracted_grid = table.extract()
                    if not extracted_grid:
                        continue
                    clean_grid = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in extracted_grid
                    ]
                    headers = clean_grid[0] if len(clean_grid) > 0 else []
                    rows = clean_grid[1:] if len(clean_grid) > 1 else []

                    table_md = _build_markdown_table(headers, rows)
                    bbox = [round(c, 2) for c in table.bbox]

                    source_ref = SourceRef(
                        filename=filename,
                        page=page_idx,
                        bbox=bbox
                    )

                    nb = NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type="table",
                        text=table_md,
                        table_data={"headers": headers, "rows": rows, "grid": clean_grid},
                        source_ref=source_ref
                    )
                    blocks.append(as_dicts([nb])[0])
        except Exception as err:
            logger.warning("Table extraction error on page %d: %s", page_idx, err)

        def _overlaps_table(b_rect):
            r = fitz.Rect(b_rect)
            return any(r.intersects(tr) for tr in table_rects)

        # 2. Extract text blocks - using dict mode for font metadata heading detection
        page_dict = page.get_text("dict")
        font_sizes = []
        for b in page_dict.get("blocks", []):
            if "lines" in b:
                for line in b["lines"]:
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(span.get("size", 10.0))

        body_font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

        raw_blocks = page.get_text("blocks")
        for b in raw_blocks:
            text_content = (b[4] or "").strip()
            if not text_content:
                continue

            bbox = [round(c, 2) for c in b[:4]]
            if _overlaps_table(b[:4]):
                # Only skip text blocks inside extracted tables if their text is already
                # captured in table_md — if the table finder missed text (e.g. unbordered cells),
                # keep the text block so no information is lost.
                table_texts = " ".join([b["text"] for b in blocks if b.get("type") == "table" and b.get("source_ref", {}).get("page") == page_idx])
                clean_t = text_content.replace("\n", " ").strip()
                if len(clean_t) > 10 and clean_t[:30] in table_texts:
                    continue

            is_heading = False
            if len(text_content) < 120 and "\n" not in text_content:
                for block_d in page_dict.get("blocks", []):
                    if is_heading:
                        break
                    if "lines" in block_d:
                        for l in block_d["lines"]:
                            for span in l.get("spans", []):
                                if text_content in span.get("text", ""):
                                    sz = span.get("size", 10.0)
                                    flags = span.get("flags", 0)
                                    is_bold = bool(flags & (1 << 4))
                                    if sz > body_font_size * 1.15 or (is_bold and sz >= body_font_size):
                                        is_heading = True
                                        break

            btype = "heading" if is_heading else "text"

            source_ref = SourceRef(
                filename=filename,
                page=page_idx,
                bbox=bbox
            )

            nb = NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type=btype,
                text=text_content,
                source_ref=source_ref
            )
            blocks.append(as_dicts([nb])[0])

    doc.close()

    # Document-level multi-page table reconciliation before chunking
    cfg = config or {}
    if (cfg.get("extraction") or {}).get("stitch_tables", True):
        blocks = reconcile_tables(blocks, pdf_path, config)

    return blocks


def _build_markdown_table(headers: list, rows: list[list]) -> str:
    if not headers and not rows:
        return "[table]"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
