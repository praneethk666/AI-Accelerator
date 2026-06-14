"""Detect if a PDF is digital, scanned, or mixed (per-page)."""

import logging
import fitz
from typing import Tuple, List

logger = logging.getLogger(__name__)

# A page with almost no extractable text layer is treated as scanned (image-only).
_DIGITAL_TEXT_THRESHOLD = 5


def detect_pdf_type(file_path: str) -> Tuple[str, List[str]]:
    doc = fitz.open(file_path)
    per_page: List[str] = []
    try:
        for page_num in range(len(doc)):
            text_len = len(doc[page_num].get_text().strip())
            page_type = "digital" if text_len > _DIGITAL_TEXT_THRESHOLD else "scanned"
            per_page.append(page_type)
            logger.debug("Page %s: %s (%s chars)", page_num + 1, page_type, text_len)
    finally:
        doc.close()

    if all(t == "digital" for t in per_page):
        overall = "digital"
    elif all(t == "scanned" for t in per_page):
        overall = "scanned"
    else:
        overall = "mixed"

    logger.debug("Overall PDF type: %s", overall)
    return overall, per_page