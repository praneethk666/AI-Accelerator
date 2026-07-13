"""Compare scanned OCR against the digital twin's native text (ground truth).

    python scripts/ocr_diff.py <engine> <digital.pdf> <scanned.pdf> [n_sample_pages]

Same-page comparison (the two PDFs must be the same document, one digital one
scanned). Prints per-page char counts + a word-level similarity ratio vs the native
text, plus side-by-side samples. Emits RESULT:{json} on the last line.
"""
import difflib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from backend.extraction.scanned_pdf import scanned as S

_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", (t or "").lower()).strip()


def _word_f1(native: str, ocr: str) -> float:
    """Order-INDEPENDENT word-overlap F1 (ignores reading order, which difflib
    penalizes). Better gauge of 'did OCR capture the same words'."""
    n, o = Counter(_norm(native).split()), Counter(_norm(ocr).split())
    if not n or not o:
        return 0.0
    overlap = sum((n & o).values())
    p, r = overlap / sum(o.values()), overlap / sum(n.values())
    return round(2 * p * r / (p + r), 3) if (p + r) else 0.0


def run(engine: str, digital: str, scanned: str, n_sample: int = 6) -> dict:
    S.set_ocr_engine(engine)
    ddoc, sdoc = fitz.open(digital), fitz.open(scanned)
    n = min(len(ddoc), len(sdoc))
    # evenly spaced sample pages across the doc (1-based for display)
    if n_sample <= 0 or n_sample >= n:
        sample = list(range(n))
    else:
        step = n / n_sample
        sample = sorted({int(i * step) for i in range(n_sample)})

    rows = []
    for i in sample:
        native = ddoc[i].get_text()
        pil = S.page_to_pil(sdoc[i], dpi=200)
        ocr, _boxes = S.extract_ocr_text_and_boxes(pil)
        ratio = difflib.SequenceMatcher(None, _norm(native).split(),
                                        _norm(ocr).split()).ratio()
        rows.append({
            "page": i + 1,
            "native_chars": len(native),
            "ocr_chars": len(ocr),
            "word_similarity": round(ratio, 3),     # order-sensitive (difflib)
            "word_f1": _word_f1(native, ocr),       # order-independent overlap
            "native_sample": " ".join(native.split())[:200],
            "ocr_sample": " ".join(ocr.split())[:200],
        })
    ddoc.close(); sdoc.close()
    avg = round(sum(r["word_similarity"] for r in rows) / len(rows), 3) if rows else 0
    avg_f1 = round(sum(r["word_f1"] for r in rows) / len(rows), 3) if rows else 0
    return {
        "engine": engine,
        "engine_active": getattr(S, "_OCR_ENGINE", engine),
        "pages_total": n,
        "sampled": [r["page"] for r in rows],
        "avg_word_similarity": avg,
        "avg_word_f1": avg_f1,
        "per_page": rows,
    }


if __name__ == "__main__":
    engine, digital, scanned = sys.argv[1], sys.argv[2], sys.argv[3]
    n_sample = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    print("RESULT:" + json.dumps(run(engine, digital, scanned, n_sample)))
