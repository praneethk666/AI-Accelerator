"""Empirical check of PDF-kind detection (digital/scanned/mixed) — NO API keys, NO DB.

Runs backend.extraction.detector.detect_pdf_type on every PDF given (files or a
directory) and prints what it detected, plus the raw signal it decided on (per-page
extractable-text length) so borderline / garbled pages are visible. This is a
feature-verification harness: it answers "given a random PDF, does kind-detection
get it right, and how confident is the signal?"

    python scripts/check_detection.py <pdf-or-dir> [<pdf-or-dir> ...]
    python scripts/check_detection.py <dir> --json out.json   # machine-readable record

Garble caveat: the detector keys on text *length* only. A digital PDF with a broken
subset font emits junk glyphs (still >5 chars) -> classified "digital" even though
the text is unreadable. That is NOT caught here (docling's _text_unreliable handles
it downstream). This harness flags it as a heuristic risk via the control-char ratio.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF


def _gather(paths: list[str]) -> list[str]:
    pdfs: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(p, name))
        elif p.lower().endswith(".pdf"):
            pdfs.append(p)
    return pdfs


def _ctrl_ratio(text: str) -> float:
    """Fraction of non-printable / replacement chars — a proxy for garbled text."""
    if not text:
        return 0.0
    bad = sum(1 for c in text if (ord(c) < 32 and c not in "\n\r\t") or c == "�")
    return bad / len(text)


def analyze(pdf_path: str) -> dict:
    from backend.extraction.detector import detect_pdf_type

    overall, per_page = detect_pdf_type(pdf_path)

    doc = fitz.open(pdf_path)
    lengths, ctrl = [], []
    try:
        for i in range(len(doc)):
            t = doc[i].get_text().strip()
            lengths.append(len(t))
            ctrl.append(_ctrl_ratio(t))
    finally:
        doc.close()

    n = len(per_page)
    digital_pages = sum(1 for t in per_page if t == "digital")
    max_ctrl = max(ctrl) if ctrl else 0.0
    return {
        "file": os.path.basename(pdf_path),
        "pages": n,
        "overall": overall,
        "digital_pages": digital_pages,
        "scanned_pages": n - digital_pages,
        "min_text_len": min(lengths) if lengths else 0,
        "max_text_len": max(lengths) if lengths else 0,
        "max_ctrl_ratio": round(max_ctrl, 3),
        "garble_risk": max_ctrl > 0.05,  # same 5% trip docling uses for _text_unreliable
    }


def main(argv: list[str]) -> None:
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]

    pdfs = _gather(argv)
    if not pdfs:
        print("no PDFs found in:", argv)
        return

    rows = [analyze(p) for p in pdfs]

    hdr = f"{'file':<48} {'pp':>4} {'overall':>8} {'dig/scn':>9} {'maxctrl':>8} {'garble?':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['file'][:48]:<48} {r['pages']:>4} {r['overall']:>8} "
            f"{str(r['digital_pages'])+'/'+str(r['scanned_pages']):>9} "
            f"{r['max_ctrl_ratio']:>8} {('YES' if r['garble_risk'] else '-'):>8}"
        )

    # summary
    from collections import Counter
    kinds = Counter(r["overall"] for r in rows)
    garbled = sum(1 for r in rows if r["garble_risk"])
    print("-" * len(hdr))
    print(f"{len(rows)} PDFs | kinds: {dict(kinds)} | garble-risk flagged: {garbled}")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1:])
