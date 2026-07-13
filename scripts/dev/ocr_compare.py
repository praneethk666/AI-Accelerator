"""Run ONE OCR engine over a scanned PDF and emit per-page + total metrics as JSON.

Run once per engine (separate processes — Surya and PaddleOCR don't share a process
cleanly: module-level engine singletons + the paddle<->torch allocator ordering).

    python scripts/ocr_compare.py surya  test-data/argo_scanned.pdf
    python scripts/ocr_compare.py paddle test-data/argo_scanned.pdf

Emits a single JSON object on the last stdout line (prefixed RESULT:) so the caller
can parse it regardless of model log noise.
"""
import json
import sys
import time

import fitz

from backend.extraction.scanned_pdf import scanned as S


def run(engine: str, pdf_path: str, max_pages: int = 0) -> dict:
    S.set_ocr_engine(engine)
    doc = fitz.open(pdf_path)
    pages = []
    t0 = time.time()
    try:
        n = len(doc) if max_pages <= 0 else min(max_pages, len(doc))
        for i in range(n):
            pil = S.page_to_pil(doc[i], dpi=200)
            ts = time.time()
            text, boxes = S.extract_ocr_text_and_boxes(pil)
            lines = len(text.splitlines())
            regions = S.detect_visual_regions(pil, boxes, lines, min_area=50000)
            pages.append({
                "page": i + 1,
                "chars": len(text),
                "text_boxes": len(boxes),
                "regions": len(regions),
                "secs": round(time.time() - ts, 2),
                "sample": " ".join(text.split())[:240],
            })
    finally:
        doc.close()
    total_secs = round(time.time() - t0, 2)
    return {
        "engine": engine,
        # which engine actually ran (surya falls back to paddle if it errors)
        "engine_active": getattr(S, "_OCR_ENGINE", engine),
        "pages": len(pages),
        "total_chars": sum(p["chars"] for p in pages),
        "total_text_boxes": sum(p["text_boxes"] for p in pages),
        "total_regions": sum(p["regions"] for p in pages),
        "total_secs": total_secs,
        "per_page": pages,
    }


if __name__ == "__main__":
    engine, pdf_path = sys.argv[1], sys.argv[2]
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    result = run(engine, pdf_path, max_pages)
    print("RESULT:" + json.dumps(result))
