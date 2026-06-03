"""Extract content from scanned PDF using OCR."""

import fitz
import io
import uuid
import numpy as np

from PIL import Image
from paddleocr import PaddleOCR
from typing import List

from backend.core.schemas import NormalizedBlock, SourceRef
from backend.utils.save_json import save_blocks


# Global OCR engine (load once)
_ocr = None


def get_ocr():
    global _ocr

    if _ocr is None:
        _ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            show_log=False,
        )

    return _ocr


def page_to_pil(page, dpi=200):
    """
    Convert a PDF page to PIL image.
    """

    pix = page.get_pixmap(dpi=dpi)

    return Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    ).convert("RGB")


def extract_scanned(
    pdf_path: str,
    document_id: str,
) -> List[NormalizedBlock]:
    """
    Extract text from scanned PDF using OCR.

    Returns:
        List[NormalizedBlock]
    """

    doc = fitz.open(pdf_path)

    ocr = get_ocr()

    blocks: List[NormalizedBlock] = []

    filename = pdf_path.split("/")[-1]

    try:
        for page_num in range(len(doc)):

            page = doc[page_num]

            print(
                f"Processing scanned page "
                f"{page_num + 1}"
            )

            pil_img = page_to_pil(
                page,
                dpi=200,
            )

            img_np = np.array(pil_img)

            # OCR
            result = ocr.ocr(img_np)

            if not result or not result[0]:
                continue

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

            # Sort top-to-bottom
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
                                    page=page_num + 1,
                                ),
                                confidence=0.8,
                            )
                        )

                    paragraph = [text]

                last_y = y

            # Flush last paragraph
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
                            page=page_num + 1,
                        ),
                        confidence=0.8,
                    )
                )

    finally:
        doc.close()

    # Auto-save JSON
    try:
        save_blocks(
            blocks,
            pdf_path,
        )
    except Exception as e:
        print(
            f"Failed to save blocks JSON: {e}"
        )

    return blocks