import os
import sys
import json
from dataclasses import asdict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.extraction.mixed import extract_mixed


def test_mixed_pdf():
    # ===== CHANGE THIS PATH TO YOUR MIXED PDF =====
    pdf_path = "test-data/Mixed.pdf"   # <-- Put your file name here
    # =============================================

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Extract using mixed pipeline (digital + scanned)
    blocks = extract_mixed(pdf_path, document_id="mixed_test")

    print("=" * 50)
    print("MIXED PDF EXTRACTION TEST")
    print("=" * 50)
    print(f"Total blocks extracted: {len(blocks)}")

    # Group blocks by page
    pages = defaultdict(list)
    for b in blocks:
        if b.source_ref and b.source_ref.page:
            pages[b.source_ref.page].append(b)

    # Print per-page summary
    for page_num in sorted(pages.keys()):
        text_blocks = [b for b in pages[page_num] if b.type in ("text", "heading")]
        image_blocks = [b for b in pages[page_num] if b.type == "image_caption"]
        table_blocks = [b for b in pages[page_num] if b.type == "table"]
        metrics_blocks = [b for b in pages[page_num] if b.type == "page_metrics"]

        total_text_len = sum(len(b.text) for b in text_blocks)

        # Determine page kind from confidence (digital = 1.0, scanned OCR = 0.8)
        kind = "unknown"
        if text_blocks:
            if any(b.confidence > 0.85 for b in text_blocks):
                kind = "digital"
            else:
                kind = "scanned"

        print(f"\nPage {page_num} ({kind})")
        print(f"  Text blocks: {len(text_blocks)} (total chars: {total_text_len})")
        print(f"  Images: {len(image_blocks)}")
        print(f"  Tables: {len(table_blocks)}")
        if metrics_blocks:
            meta = metrics_blocks[0].metadata
            print(f"  Metadata text_length: {meta.get('text_length', 'N/A')}")
        # Preview first 200 characters
        if text_blocks:
            preview = " ".join(b.text for b in text_blocks[:2])[:200]
            print(f"  Preview: {preview}...")

    # Save blocks JSON
    os.makedirs("output/blocks", exist_ok=True)
    out_path = "output/blocks/mixed_blocks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in blocks], f, indent=2, default=str)
    print(f"\nFull output saved to: {out_path}")


if __name__ == "__main__":
    test_mixed_pdf()