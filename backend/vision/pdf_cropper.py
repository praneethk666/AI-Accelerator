# backend/vision/pdf_cropper.py
import fitz  # PyMuPDF

class PDFCropper:
    """Crops a rectangular region from a PDF page and returns PNG bytes."""

    def crop_region(self, pdf_path, page_number, bbox, dpi=200):
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            rect = fitz.Rect(*bbox)
            pix = page.get_pixmap(clip=rect, dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()