"""Rasterize a digital PDF into a SCANNED (image-only) PDF for OCR testing.

Renders every page to a JPEG at the given DPI and rebuilds a PDF whose pages are
just those images — no text layer. detector.detect_pdf_type() then classifies it
as 'scanned', exercising the OCR path. JPEG (not PNG) keeps the file realistic and
small, like an actual scan. Usage:

    python scripts/make_scanned.py <input.pdf> <output.pdf> [dpi] [jpg_quality]
"""
import os
import sys
import fitz


def make_scanned(src: str, dst: str, dpi: int = 150, quality: int = 75) -> None:
    doc = fitz.open(src)
    out = fitz.open()
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            jpg = pix.tobytes("jpeg", jpg_quality=quality)
            # New page same size as the rendered image, insert JPEG full-bleed.
            rect = fitz.Rect(0, 0, pix.width, pix.height)
            npage = out.new_page(width=pix.width, height=pix.height)
            npage.insert_image(rect, stream=jpg)
        out.save(dst, deflate=True, garbage=4)
        mb = os.path.getsize(dst) / 1e6
        print(f"wrote {dst} ({len(doc)} pages @ {dpi} dpi jpeg q{quality}, {mb:.1f} MB, image-only)")
    finally:
        doc.close()
        out.close()


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    quality = int(sys.argv[4]) if len(sys.argv) > 4 else 75
    make_scanned(src, dst, dpi, quality)
