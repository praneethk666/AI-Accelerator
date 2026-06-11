"""Handle mixed PDFs: use digital extraction for digital pages
and OCR extraction for scanned pages."""

import fitz
import uuid
import numpy as np

from typing import List

from backend.extraction.pdf.detector import detect_pdf_type
from backend.extraction.ocr.scanned import get_ocr, page_to_pil
from backend.core.schemas import NormalizedBlock, SourceRef
from backend.utils.save_json import save_blocks

# Configuration (move to config later)
MIN_IMAGE_AREA = 1000               # ignore logos/icons smaller than ~32x32 px
MIN_DRAWING_AREA_RATIO = 0.005      # 0.5% of page area -> captures small vector drawings
MIN_VECTOR_COMPLEXITY = 5           # minimum number of drawing items to be considered meaningful
TABLE_OVERLAP_THRESHOLD = 0.5       # skip vector if >50% of its area overlaps a table


def extract_mixed(
    pdf_path: str,
    document_id: str
) -> List[NormalizedBlock]:
    """
    Process a mixed PDF.

    Digital pages:
        -> PyMuPDF extraction (text, tables, images, vectors)

    Scanned pages:
        -> OCR extraction (text only)

    Returns:
        List[NormalizedBlock]
    """

    _, per_page_types = detect_pdf_type(pdf_path)

    doc = fitz.open(pdf_path)

    all_blocks: List[NormalizedBlock] = []

    filename = pdf_path.split("/")[-1]

    try:
        for page_num, page_type in enumerate(per_page_types):

            page = doc[page_num]
            page_number = page_num + 1

            print(
                f"Processing page {page_number} "
                f"({page_type})"
            )

            if page_type == "digital":

                blocks = _extract_digital_page(
                    page,
                    page_number,
                    filename,
                    document_id,
                )

            else:

                pil_img = page_to_pil(
                    page,
                    dpi=200,
                )

                blocks = _extract_scanned_page(
                    pil_img,
                    page_number,
                    filename,
                    document_id,
                )

            all_blocks.extend(blocks)

    finally:
        doc.close()

    # Auto-save JSON
    try:
        save_blocks(
            all_blocks,
            pdf_path,
        )
    except Exception as e:
        print(
            f"Failed to save blocks JSON: {e}"
        )

    return all_blocks


def _extract_digital_page(
    page,
    page_number: int,
    filename: str,
    document_id: str,
) -> List[NormalizedBlock]:
    """
    Extract a single digital page.
    Includes text, headings, tables, images, and vector drawings (with bbox).
    Vector drawings that overlap tables (by >50% area) are ignored.
    Vector drawings with low complexity (few drawing items) are also ignored.
    """

    blocks: List[NormalizedBlock] = []
    page_area = page.rect.width * page.rect.height

    # ---- Detect tables for overlap filtering ----
    try:
        tables_found = page.find_tables()
        table_bboxes = [t.bbox for t in tables_found.tables] if tables_found.tables else []
    except Exception:
        table_bboxes = []

    # ---- 1. Text extraction ----
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):

        if block["type"] != 0:
            continue

        for line in block.get("lines", []):

            for span in line.get("spans", []):

                text = span["text"].strip()

                if not text:
                    continue

                font_size = span["size"]
                font_name = span.get("font", "")

                is_bold = (
                    "bold" in font_name.lower()
                )

                block_type = (
                    "heading"
                    if (
                        font_size >= 14
                        or (
                            font_size >= 12
                            and is_bold
                        )
                    )
                    else "text"
                )

                blocks.append(
                    NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type=block_type,
                        text=text,
                        source_ref=SourceRef(
                            filename=filename,
                            page=page_number,
                        ),
                        confidence=1.0,
                    )
                )

    # ---- 2. Table extraction ----
    try:
        tables = page.find_tables()

        for table in tables.tables:

            extracted = table.extract()

            if not extracted:
                continue

            headers = (
                [str(h) for h in table.header.names]
                if table.header
                else []
            )

            rows = [
                [str(cell) for cell in row]
                for row in extracted
            ]

            blocks.append(
                NormalizedBlock(
                    block_id=str(uuid.uuid4()),
                    document_id=document_id,
                    type="table",
                    table_data={
                        "headers": headers,
                        "rows": rows,
                    },
                    source_ref=SourceRef(
                        filename=filename,
                        page=page_number,
                    ),
                    confidence=0.95,
                )
            )

    except Exception as e:
        print(
            f"Table extraction failed "
            f"on page {page_number}: {e}"
        )

    # ---- 3. Image placeholders (significant only) WITH BBOX ----
    try:
        for img in page.get_images(full=True):
            rects = page.get_image_rects(img)
            if not rects:
                continue
            bbox = rects[0]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width * height < MIN_IMAGE_AREA:
                continue   # skip tiny logos/icons
            blocks.append(
                NormalizedBlock(
                    block_id=str(uuid.uuid4()),
                    document_id=document_id,
                    type="image_caption",
                    text="[Image - awaiting vision enrichment]",
                    source_ref=SourceRef(
                        filename=filename,
                        page=page_number,
                        bbox=list(bbox)
                    ),
                    confidence=0.5,
                    metadata={"pending_vision": True, "is_vector": False},
                )
            )
    except Exception as e:
        print(f"Image extraction failed on page {page_number}: {e}")

    # ---- 4. Vector drawing placeholders (filtered: area + complexity + table overlap) ----
    try:
        drawings = page.get_drawings()
        for dr in drawings:
            rect = dr.get("rect")
            if not rect:
                continue
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            # Area threshold
            if area / page_area < MIN_DRAWING_AREA_RATIO:
                continue

            # Complexity filter: skip simple drawings (e.g., single rectangle)
            items = dr.get("items", [])
            if len(items) < MIN_VECTOR_COMPLEXITY:
                continue

            # Overlap check with tables
            skip = False
            for t_bbox in table_bboxes:
                # intersection area
                ix1 = max(rect[0], t_bbox[0])
                iy1 = max(rect[1], t_bbox[1])
                ix2 = min(rect[2], t_bbox[2])
                iy2 = min(rect[3], t_bbox[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / area > TABLE_OVERLAP_THRESHOLD:
                        skip = True
                        break
            if skip:
                continue

            blocks.append(
                NormalizedBlock(
                    block_id=str(uuid.uuid4()),
                    document_id=document_id,
                    type="image_caption",
                    text="[Vector drawing - awaiting vision enrichment]",
                    source_ref=SourceRef(
                        filename=filename,
                        page=page_number,
                        bbox=list(rect)
                    ),
                    confidence=0.5,
                    metadata={"pending_vision": True, "is_vector": True},
                )
            )
    except Exception as e:
        print(f"Vector drawing extraction failed on page {page_number}: {e}")

    return blocks


def _extract_scanned_page(
    pil_img,
    page_number: int,
    filename: str,
    document_id: str,
) -> List[NormalizedBlock]:
    """
    OCR a single scanned page.
    Returns only text blocks (no images/vectors for scanned pages).
    """

    ocr = get_ocr()

    img_np = np.array(pil_img)

    result = ocr.ocr(img_np)

    blocks: List[NormalizedBlock] = []

    if not result or not result[0]:
        return blocks

    lines = []

    for line in result[0]:

        text = line[1][0]
        bbox = line[0]

        y_center = (
            bbox[0][1] +
            bbox[2][1]
        ) / 2

        lines.append(
            (y_center, text)
        )

    lines.sort(
        key=lambda x: x[0]
    )

    paragraph = []
    last_y = None

    for y, text in lines:

        if (
            last_y is None
            or (y - last_y) < 15
        ):
            paragraph.append(text)

        else:

            if paragraph:

                para_text = (
                    " ".join(paragraph)
                )

                block_type = (
                    "heading"
                    if (
                        para_text.isupper()
                        and len(para_text) < 100
                    )
                    else "text"
                )

                blocks.append(
                    NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type=block_type,
                        text=para_text,
                        source_ref=SourceRef(
                            filename=filename,
                            page=page_number,
                        ),
                        confidence=0.8,
                    )
                )

            paragraph = [text]

        last_y = y

    if paragraph:

        para_text = (
            " ".join(paragraph)
        )

        block_type = (
            "heading"
            if (
                para_text.isupper()
                and len(para_text) < 100
            )
            else "text"
        )

        blocks.append(
            NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type=block_type,
                text=para_text,
                source_ref=SourceRef(
                    filename=filename,
                    page=page_number,
                ),
                confidence=0.8,
            )
        )

    return blocks