"""Text extraction helpers for multiple file types (PDF, Excel, PPT, images).

Used for:
- industry keyword scoring (first 3 pages)
- TOC extraction (no API cost)

The categorization engine still relies on vision for document_type.
"""

from __future__ import annotations

from typing import List
import os

import fitz

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

try:
    from PIL import Image
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def extract_text(file_path: str, max_pages: int = 3) -> str:
    """Extract raw text from the first `max_pages` pages of any supported file type."""
    if not os.path.exists(file_path):
        return ""

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return _extract_pdf_text(file_path, max_pages)
        elif ext in [".xlsx", ".xls"]:
            return _extract_excel_text(file_path, max_pages)
        elif ext in [".pptx", ".ppt"]:
            return _extract_ppt_text(file_path, max_pages)
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            return _extract_image_text(file_path)
        else:
            # For unknown types, try PDF first (might be misnamed)
            try:
                return _extract_pdf_text(file_path, max_pages)
            except Exception:
                return ""
    except Exception:
        return ""


def _extract_pdf_text(file_path: str, max_pages: int = 3) -> str:
    """Extract raw text from PDF pages."""
    text_parts: List[str] = []
    try:
        doc = fitz.open(file_path)
        pages_to_read = min(max_pages, len(doc))
        for page_num in range(pages_to_read):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
    except Exception:
        return ""
    return "\n".join(text_parts)


def _extract_excel_text(file_path: str, max_pages: int = 3) -> str:
    """Extract text from Excel workbook."""
    if not HAS_OPENPYXL:
        return ""

    text_parts: List[str] = []
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet_count = 0
        for sheet in wb.sheetnames:
            if sheet_count >= max_pages:
                break
            ws = wb[sheet]
            text_parts.append(f"Sheet: {sheet}")
            for row in ws.iter_rows(values_only=True):
                text_parts.append(" | ".join(str(v) if v is not None else "" for v in row))
            sheet_count += 1
        wb.close()
    except Exception:
        pass
    return "\n".join(text_parts)


def _extract_ppt_text(file_path: str, max_pages: int = 3) -> str:
    """Extract text from PowerPoint presentation."""
    if not HAS_PPTX:
        return ""

    text_parts: List[str] = []
    try:
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides):
            if slide_num >= max_pages:
                break
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    if shape.text.strip():
                        text_parts.append(shape.text)
    except Exception:
        pass
    return "\n".join(text_parts)


def _extract_image_text(file_path: str) -> str:
    """Extract text from image using OCR (requires pytesseract)."""
    if not HAS_PYTESSERACT:
        return ""

    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""


def extract_toc_text(file_path: str, max_pages: int = 2) -> str:
    """Best-effort TOC extraction.

    If the PDF has a table of contents, fitz may expose it via doc.get_toc().
    We also do a minimal heuristic scan in the first few pages as a backup.
    """
    if not os.path.exists(file_path):
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    # Only PDFs have TOC structure
    if ext != ".pdf":
        return extract_text(file_path, max_pages)

    try:
        doc = fitz.open(file_path)
        toc = []
        try:
            toc = doc.get_toc() or []
        except Exception:
            toc = []

        if toc:
            # toc entries: list of (level, title, page_number)
            lines = []
            for entry in toc:
                if len(entry) >= 2:
                    title = entry[1]
                    lines.append(str(title))
            doc.close()
            return "\n".join(lines)

        # fallback: scan first N pages for TOC-like patterns
        pages_to_read = min(max_pages, len(doc))
        parts: List[str] = []
        for i in range(pages_to_read):
            page = doc[i]
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)

    except Exception:
        return ""

