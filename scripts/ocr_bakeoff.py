"""Scanned-page OCR bake-off: PaddleOCR vs Docling(RapidOCR) vs GLM vs Qwen3-VL,
scored against the DIGITAL TWIN page (ground truth).

Usage:
    python scripts/ocr_bakeoff.py <scanned.pdf> <digital_twin.pdf> <page> [page2 ...]

Prints, per page, each engine's text-similarity to the digital twin plus a snippet,
so we decide the scanned-text engine on evidence. VLM calls use:
  GLM  -> config['vision_ocr'] (Z.ai, free)
  Qwen -> OpenRouter qwen/qwen3-vl-30b-a3b-instruct (OPENROUTER_API_KEY)
"""
import os
import re
import sys
import difflib

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import fitz
from backend.core.config import load_config
from backend.core.vision_client import describe_image
from backend.extraction.vision_ocr import transcribe_page, _PROMPT


def _words(t):
    # content words only: lowercased alphanumeric tokens length>=3 (drops punctuation
    # and most layout noise); multiset so repeats count.
    return re.findall(r"[a-z0-9]{3,}", (t or "").lower())


def sim(cand, gt):
    """Word-level F1 of candidate vs ground truth (order/whitespace/boilerplate-robust).
    Better than char-sequence ratio, which the VLMs lose on by (correctly) dropping
    repeated header/footer boilerplate present in the digital twin."""
    from collections import Counter
    c, g = Counter(_words(cand)), Counter(_words(gt))
    if not c or not g:
        return 0.0
    overlap = sum((c & g).values())
    prec = overlap / sum(c.values())
    rec = overlap / sum(g.values())
    return round(2 * prec * rec / (prec + rec), 3) if (prec + rec) else 0.0


def paddle_text(scanned_pdf, page):
    from backend.extraction.scanned_pdf import scanned as S
    S.set_ocr_engine("paddle")
    pil = S.page_to_pil(fitz.open(scanned_pdf)[page - 1], dpi=200)
    text, _boxes = S.extract_ocr_text_and_boxes(pil)
    return text


def docling_text(scanned_pdf, page):
    from docling.document_converter import DocumentConverter
    src = fitz.open(scanned_pdf)
    tmp = fitz.open()
    tmp.insert_pdf(src, from_page=page - 1, to_page=page - 1)
    path = "/tmp/_ocr_bakeoff_1p.pdf"
    tmp.save(path)
    return DocumentConverter().convert(path).document.export_to_markdown()


def qwen_text(scanned_pdf, page, cfg):
    png = fitz.open(scanned_pdf)[page - 1].get_pixmap(dpi=200).tobytes("png")
    qcfg = {"vision": {
        "provider": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "max_retries": 3, "timeout_s": 120, "min_interval_s": 0,
    }}
    return describe_image(png, _PROMPT, qcfg)


def main():
    scanned, digital = sys.argv[1], sys.argv[2]
    pages = [int(p) for p in sys.argv[3:]] or [1]
    cfg = load_config("config/global.yaml")
    for page in pages:
        gt = fitz.open(digital)[page - 1].get_text()
        engines = {}
        for name, fn in (("Paddle", lambda: paddle_text(scanned, page)),
                         ("Docling", lambda: docling_text(scanned, page)),
                         ("GLM", lambda: transcribe_page(fitz.open(scanned)[page - 1], cfg)),
                         ("Qwen3VL", lambda: qwen_text(scanned, page, cfg))):
            try:
                engines[name] = fn()
            except Exception as e:
                engines[name] = f"(FAILED: {type(e).__name__}: {e})"
        print("\n" + "=" * 78)
        print(f"PAGE {page}  | ground-truth (digital twin) chars: {len(gt.strip())}")
        print("-" * 78)
        for name, txt in engines.items():
            print(f"{name:9s} sim={sim(txt, gt):<6} chars={len(txt.strip()):<6}")
        for name, txt in engines.items():
            print(f"\n----- {name} [first 400] -----\n{txt.strip()[:400]}")
        print(f"\n----- GROUND TRUTH [first 400] -----\n{gt.strip()[:400]}")


if __name__ == "__main__":
    main()
