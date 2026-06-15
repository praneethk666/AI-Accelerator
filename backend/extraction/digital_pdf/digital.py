"""Extract content from a digital PDF into NormalizedBlock list.

Merges spans into paragraphs to reduce fragmentation.
Includes bounding boxes for raster image placeholders (individual crops).
Pages with vector drawings get ONE full-page image block instead of
multiple small cropped vector regions — gives vision enrichment full context.
Vector drawings that overlap tables (by >50% area) are ignored in the profile,
and extraction uses the profile's has_vector_graphics flag (single source of truth).
"""
import os
import fitz
import uuid
from typing import List, Optional

from backend.core.schemas import NormalizedBlock, SourceRef, PageProfile

# Configuration (move to config later)
MIN_IMAGE_AREA = 1000   # ignore logos/icons smaller than ~32x32 px
LINE_GAP_FACTOR = 1.5   # lines closer than this * line_height are same paragraph


def _span_overlaps_table(span_bbox, table_bboxes, threshold=0.8):
    """Return True if span's bbox overlaps any table bbox by > threshold area."""
    sx1, sy1, sx2, sy2 = span_bbox
    for tb in table_bboxes:
        ix1 = max(sx1, tb[0])
        iy1 = max(sy1, tb[1])
        ix2 = min(sx2, tb[2])
        iy2 = min(sy2, tb[3])
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            span_area = (sx2 - sx1) * (sy2 - sy1)
            if span_area > 0 and inter_area / span_area > threshold:
                return True
    return False


def extract_digital(
    pdf_path: str,
    document_id: str,
    page_profiles: Optional[List[PageProfile]] = None,
) -> List[NormalizedBlock]:
    """Extract content from a digital PDF.

    Uses pre‑computed page profiles (if provided) to know which pages have vector
    graphics. This keeps extraction and profiles perfectly in sync.

    Args:
        pdf_path: Path to the PDF file on disk
        document_id: Document identifier (UUID string)
        page_profiles: Optional list of PageProfile objects (one per page).
                       If provided, the function uses profile.has_vector_graphics
                       to decide whether to add a full‑page vector block.

    Returns:
        List of NormalizedBlock objects (text / heading / table / image)
    """
    doc = fitz.open(pdf_path)
    blocks: List[NormalizedBlock] = []
    filename = os.path.basename(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            page_area = page.rect.width * page.rect.height

            # Retrieve profile for this page if available
            profile = None
            if page_profiles and page_num < len(page_profiles):
                profile = page_profiles[page_num]

            # ── 1. Extract tables FIRST so we have their bboxes ───────────────
            table_bboxes: list = []
            try:
                tables = page.find_tables()
                table_bboxes = [t.bbox for t in tables.tables] if tables.tables else []

                for table in tables.tables:
                    extracted = table.extract()
                    if not extracted:
                        continue
                    # Replace None with empty string in headers and rows
                    headers = [str(h) if h is not None else "" for h in (table.header.names if table.header else [])]
                    rows = [[str(cell) if cell is not None else "" for cell in row] for row in extracted]
                    # Skip rows that are completely empty
                    rows = [row for row in rows if any(cell.strip() for cell in row)]
                    if not rows:
                        continue
                    header_row = "| " + " | ".join(headers) + " |" if headers else ""
                    separator = "| " + " | ".join(["---"] * len(headers)) + " |" if headers else ""
                    data_rows = ["| " + " | ".join(r) + " |" for r in rows]
                    md_parts = [p for p in [header_row, separator] + data_rows if p]
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="table",
                            text="\n".join(md_parts),
                            table_data={"headers": headers, "rows": rows},
                            source_ref=SourceRef(filename=filename, page=page_number),
                            confidence=0.95,
                        )
                    )
            except Exception:
                # Errors are propagated via state in the tool; no print here
                pass

            # ── 2. Text as paragraphs (merged), skipping spans inside tables ──
            para_blocks = _extract_paragraphs(page, page_number, filename, document_id, table_bboxes)
            blocks.extend(para_blocks)

            # ── 3. Raster image blocks (individual crops) with deduplication ──
            seen_raster_bboxes = set()
            try:
                for img in page.get_images(full=True):
                    rects = page.get_image_rects(img)
                    if not rects:
                        continue
                    bbox = rects[0]
                    rounded_bbox = tuple(round(c, 2) for c in bbox)
                    if rounded_bbox in seen_raster_bboxes:
                        continue
                    seen_raster_bboxes.add(rounded_bbox)

                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width * height < MIN_IMAGE_AREA:
                        continue
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="image",
                            text="",
                            source_ref=SourceRef(
                                filename=filename,
                                page=page_number,
                                bbox=list(bbox),
                            ),
                            confidence=0.9,
                            metadata={
                                "pending_vision": True,
                                "is_vector": False,
                                "is_full_page": False,
                            },
                        )
                    )
            except Exception:
                # Errors are propagated via state; no print here
                pass

            # ── 4. Vector diagrams → ONE full-page image block ────────────────
            # Use the profile's has_vector_graphics flag (single source of truth)
            if profile and profile.has_vector_graphics:
                full_page_bbox = [
                    0.0,
                    0.0,
                    float(page.rect.width),
                    float(page.rect.height),
                ]
                blocks.append(
                    NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type="image",
                        text="",
                        source_ref=SourceRef(
                            filename=filename,
                            page=page_number,
                            bbox=full_page_bbox,
                        ),
                        confidence=0.9,
                        metadata={
                            "pending_vision": True,
                            "is_vector": True,
                            "is_full_page": True,
                        },
                    )
                )
                # No print – silent operation

    finally:
        doc.close()

    return blocks


def _extract_paragraphs(
    page,
    page_num: int,
    filename: str,
    document_id: str,
    table_bboxes: list = None,
) -> List[NormalizedBlock]:
    """Extract text from a page, merging spans into paragraphs.

    Skips text spans that lie inside any table bbox.

    Args:
        page: PyMuPDF page object
        page_num: Page number (1-indexed)
        filename: Original filename
        document_id: Document identifier
        table_bboxes: List of table bounding boxes (optional)

    Returns:
        List of NormalizedBlock objects (type "text" or "heading")
    """
    if table_bboxes is None:
        table_bboxes = []

    text_dict = page.get_text("dict")
    lines = []

    for block in text_dict.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                if _span_overlaps_table(span["bbox"], table_bboxes):
                    continue
                spans.append({
                    "text": text,
                    "size": span["size"],
                    "font": span.get("font", ""),
                    "bbox": span["bbox"],
                })
            if spans:
                line_bbox = list(spans[0]["bbox"])
                for s in spans[1:]:
                    line_bbox[0] = min(line_bbox[0], s["bbox"][0])
                    line_bbox[1] = min(line_bbox[1], s["bbox"][1])
                    line_bbox[2] = max(line_bbox[2], s["bbox"][2])
                    line_bbox[3] = max(line_bbox[3], s["bbox"][3])
                line_text = " ".join(s["text"] for s in spans)
                avg_size = sum(s["size"] for s in spans) / len(spans)
                is_bold = any(
                    "Bold" in s["font"] or "bold" in s["font"].lower()
                    for s in spans
                )
                lines.append({
                    "y0": line_bbox[1],
                    "y1": line_bbox[3],
                    "text": line_text,
                    "font_size": avg_size,
                    "is_bold": is_bold,
                })

    if not lines:
        return []

    lines.sort(key=lambda l: l["y0"])

    paragraphs = []
    current_para = []
    prev_line = None

    for line in lines:
        if prev_line is None:
            current_para.append(line)
        else:
            prev_height = prev_line["y1"] - prev_line["y0"]
            gap = line["y0"] - prev_line["y1"]
            if gap < LINE_GAP_FACTOR * prev_height:
                current_para.append(line)
            else:
                paragraphs.append(current_para)
                current_para = [line]
        prev_line = line

    if current_para:
        paragraphs.append(current_para)

    result = []
    for para in paragraphs:
        para_text = " ".join(l["text"] for l in para)
        first_line = para[0]
        is_heading = (
            first_line["font_size"] >= 14
            or (first_line["font_size"] >= 12 and first_line["is_bold"])
        )
        result.append(
            NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type="heading" if is_heading else "text",
                text=para_text,
                source_ref=SourceRef(filename=filename, page=page_num),
                confidence=1.0,
            )
        )
    return result