"""Render full PDF pages to JPEGs for visual grounding ("pull up the source page").

Shared by the API's /upload handler AND scripts/run_ingest.py — real finding,
28-Jul: this used to live ONLY in backend/api/main.py's _run_ingestion, so the CLI
ingestion path never populated document_pages at all (confirmed live: 0 rows after
a real 105-page ingest via run_ingest.py). Extracted here so both callers get the
same behavior instead of the CLI path silently testing a narrower pipeline than
production.
"""
from __future__ import annotations

import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
PAGES_DIR = os.path.join(UPLOAD_DIR, "pages")


def save_page_images(document_id: str, pdf_path: str, pages: set, dpi: int = 150) -> list:
    """Render the given (1-based) PDF pages to JPEGs under uploads/pages/<doc_id>/
    and return [(page, web_path, width, height)]. PDF-only; best-effort. fitz
    rendering is pure C (no torch/paddle), so it's safe in the backend process."""
    import fitz
    page_dir = os.path.join(PAGES_DIR, document_id)
    os.makedirs(page_dir, exist_ok=True)
    out = []
    doc = fitz.open(pdf_path)
    try:
        for p in sorted(pages):
            if p < 1 or p > len(doc):
                continue
            pix = doc[p - 1].get_pixmap(dpi=dpi)
            fname = f"p{p}.jpg"
            with open(os.path.join(page_dir, fname), "wb") as f:
                f.write(pix.tobytes("jpeg", jpg_quality=80))
            out.append((p, f"/pages/{document_id}/{fname}", pix.width, pix.height))
    finally:
        doc.close()
    return out


def pages_with_chunks(chunks: list[dict]) -> set:
    """1-based page numbers that produced at least one chunk, from each chunk's
    source_ref.page. Shared helper so both callers compute the same page set."""
    return {
        int(c["source_ref"]["page"]) for c in chunks
        if isinstance(c.get("source_ref"), dict) and c["source_ref"].get("page") is not None
    }


def physical_path(image_path: str) -> str:
    """Map a stored '/pages/<doc_id>/p{N}.jpg' web path (document_pages.image_path)
    back to the file on disk. Single source of truth for the '/pages' <-> PAGES_DIR
    mapping — also what the API's StaticFiles mount (backend/api/main.py) serves
    from, so a caller reading the file directly (e.g. for image-grounded answers)
    stays consistent with what a browser would see at that URL."""
    rel = image_path.split("/pages/", 1)[-1] if "/pages/" in image_path else image_path.lstrip("/")
    return os.path.join(PAGES_DIR, rel)
