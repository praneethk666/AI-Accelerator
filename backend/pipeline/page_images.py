"""Path mapping for rendered page images (document_pages / uploads/pages/<doc_id>/).

Page/slide image rendering + document_pages persistence now happens INSIDE
ingest_document() itself (backend/pipeline/ingest.py), covering the API and
CLI paths identically and adding PPT slide support. What's left here is the
one thing a READER of an already-saved image needs: mapping the stored
'/pages/<doc_id>/p{N}.jpg' web path back to a file on disk (used by
image-grounded answering, backend/retrieval/answerer.py).
"""
from __future__ import annotations

import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
PAGES_DIR = os.path.join(UPLOAD_DIR, "pages")


def physical_path(image_path: str) -> str:
    """Map a stored '/pages/<doc_id>/p{N}.jpg' web path (document_pages.image_path)
    back to the file on disk. Single source of truth for the '/pages' <-> PAGES_DIR
    mapping — also what the API's StaticFiles mount (backend/api/main.py) serves
    from, so a caller reading the file directly (e.g. for image-grounded answers)
    stays consistent with what a browser would see at that URL."""
    rel = image_path.split("/pages/", 1)[-1] if "/pages/" in image_path else image_path.lstrip("/")
    return os.path.join(PAGES_DIR, rel)
