# backend/vision/pdf_cropper.py
import fitz  # PyMuPDF


class PDFCropper:
    """Crops a rectangular region from a PDF page and returns PNG bytes."""

    # Named so docling_extract.py's collision-detector can check the SAME
    # widths this class actually pads with, instead of a disconnected guess
    # (real bug found live, 12-Aug: the old check only scanned a fixed 10pt
    # strip while the real default side padding is width*0.10+24pt, ~49pt for
    # a ~250pt-wide figure -- unrelated text 10-49pt away bled into crops
    # completely undetected). Keep any change here in sync with that check.
    DEFAULT_SIDE_FRAC = 0.10
    DEFAULT_SIDE_PAD_PTS = 24
    DEFAULT_TOP_FRAC = 0.02
    DEFAULT_BOTTOM_FRAC = 0.04
    DEFAULT_CAPTION_PAD_PTS = 52

    def crop_region(self, pdf_path, page_number, bbox, dpi=200,
                    side_frac=DEFAULT_SIDE_FRAC, side_pad_pts=DEFAULT_SIDE_PAD_PTS,
                    top_frac=DEFAULT_TOP_FRAC, bottom_frac=DEFAULT_BOTTOM_FRAC,
                    caption_pad_pts=DEFAULT_CAPTION_PAD_PTS):
        """Crop the region with ASYMMETRIC padding so the figure's caption/label and
        side callouts are included without bleeding into the figure above.

        - Sides: a fraction of width PLUS a fixed amount (`side_pad_pts`) so callout
          labels beside the figure (e.g. "Remove Battery", "Disconnect Filler Hose")
          and arrows aren't clipped — small figures need the fixed part, wide ones
          the fraction.
        - Bottom: a fixed amount (`caption_pad_pts` ≈ 3 lines) + a small fraction —
          captions are a fixed number of text lines regardless of figure size.
        - Top: minimal, to avoid catching the figure/caption above.
        All clamped to the page."""
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            x0, y0, x1, y1 = bbox
            w, h = (x1 - x0), (y1 - y0)
            padx = w * side_frac + side_pad_pts
            rect = fitz.Rect(
                x0 - padx,
                y0 - h * top_frac,
                x1 + padx,
                y1 + h * bottom_frac + caption_pad_pts,
            ) & page.rect          # clamp to page bounds
            pix = page.get_pixmap(clip=rect, dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()