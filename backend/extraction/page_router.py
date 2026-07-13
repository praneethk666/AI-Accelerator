"""Per-page routing decision for the Docling hybrid extractor: measure a page and
decide HOW to extract it — from runtime signals, never a hard-coded per-document
assumption. The same features work for a manual text page, an invoice, a dense spec
sheet, a scanned form, or a large CAD / circuit sheet.

(Distinct from the native pipeline's heavier `page_profile.PageProfile`, which also
runs OCR; this one is a cheap fitz-only x-ray used to pick the Docling-path route.)

page_class one of: text, scanned, diagram, large_format.
Routing keys off `page_class` + the document's `document_type` (from categorize), so a
schematic page inside a manual is still tiled, and a manual page inside a CAD set is
still read as text.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LETTER_AREA = 612.0 * 792.0           # US Letter in PDF points^2 (8.5x11in)
_A3_LONG_EDGE = 1190.0                  # ~A3 long edge in points (oversized => engineering)
_DIAGRAM_DOCTYPES = {"cad_drawing", "circuit_diagram", "schematic", "engineering_drawing"}

def _image_area_frac(page) -> float:
    """Fraction of the page area covered by embedded raster images. Catches the case
    a plain text-char count misses: a page that's a full scanned image with only a
    thin machine-typed caption on top (e.g. 'Academic Year: 2012-2013') has enough
    native text to look 'digital', but is visually a scan — this signal exposes it."""
    try:
        infos = page.get_image_info()  # [{'bbox': (l,t,r,b), ...}, ...]
    except Exception:
        return 0.0
    page_area = page.rect.width * page.rect.height
    if page_area <= 0 or not infos:
        return 0.0
    total = 0.0
    for info in infos:
        bbox = info.get("bbox")
        if bbox:
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            total += w * h
    return min(1.0, total / page_area)

def profile_page(page) -> dict:
    """Cheap measured features for one fitz page (no model calls, no OCR)."""
    try:
        native = (page.get_text() or "").strip()
    except Exception:
        native = ""
    try:
        vector_paths = len(page.get_drawings())
    except Exception:
        vector_paths = 0
    try:
        raster_images = len(page.get_images(full=False))
    except Exception:
        raster_images = 0
    image_area_frac = _image_area_frac(page)    
    w, h = float(page.rect.width), float(page.rect.height)
    return {
        "text_chars": len(native),
        "has_text_layer": len(native) > 5,
        "vector_paths": vector_paths,
        "raster_images": raster_images,
        "image_area_frac": image_area_frac,
        "width_pt": w,
        "height_pt": h,
        "is_large_format": (w * h) > 1.5 * _LETTER_AREA or max(w, h) > _A3_LONG_EDGE,
        "image_dominant": image_area_frac >= 0.3 and len(native) < 150,
    }


def classify_page(prof: dict, document_type: str = "") -> str:
    """Map page features (+ the document's type) to a handling class. A diagram page is
    vector-dense with sparse text, OR belongs to a known drawing-type document; a
    large-format page is any oversized sheet (tiled regardless of content)."""
    dt = (document_type or "").lower()
    vector_dense = prof.get("vector_paths", 0) >= 60 and prof.get("text_chars", 0) < 400
    # Diagram-ness wins over size: an E-size schematic is a "diagram" (schematic prompt),
    # not a generic "large_format" sheet.
    if dt in _DIAGRAM_DOCTYPES or vector_dense:
        return "diagram"
    if prof.get("is_large_format"):
        return "large_format"
    if not prof.get("has_text_layer"):
        return "scanned"
    return "text"


def should_tile(page, page_class: str, config: dict) -> bool:
    """Tile only when the page is a diagram/large-format sheet AND it's big enough that
    a whole-page render would lose detail at the VLM's legible resolution."""
    if page_class not in ("diagram", "large_format"):
        return False
    try:
        from backend.extraction.large_format import needs_tiling
        return needs_tiling(page, config)
    except Exception:
        return False
