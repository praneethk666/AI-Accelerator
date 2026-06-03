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


def extract_mixed(
    pdf_path: str,
    document_id: str
) -> List[NormalizedBlock]:
    """
    Process a mixed PDF.

    Digital pages:
        -> PyMuPDF extraction

    Scanned pages:
        -> OCR extraction

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
    """

    blocks: List[NormalizedBlock] = []

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

    # Table extraction
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

    return blocks


def _extract_scanned_page(
    pil_img,
    page_number: int,
    filename: str,
    document_id: str,
) -> List[NormalizedBlock]:
    """
    OCR a single scanned page.
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