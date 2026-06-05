import os
import sys
import json
from dataclasses import asdict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.extraction.ocr.scanned import extract_scanned
from backend.extraction.pdf.page_profile import page_profile


def test_scanned_pages():
    pdf_path = "test-data/Expansion Motor_1pages.pdf"

    # Step 1: Extract full content using OCR + visual detection
    blocks = extract_scanned(pdf_path, document_id="test_scanned", min_visual_area=50000)

    print("=" * 50)
    print("SCANNED PAGE EXTRACTION TEST (OCR)")
    print("=" * 50)
    print(f"Total blocks extracted: {len(blocks)}")

    # Group blocks by page for preview
    pages = defaultdict(list)
    for b in blocks:
        if b.source_ref and b.source_ref.page:
            pages[b.source_ref.page].append(b)

    for page_num in sorted(pages.keys()):
        print(f"\n--- Page {page_num} OCR output ---")
        # Combine all text blocks (skip image_caption and page_metrics for preview)
        text_blocks = [b.text for b in pages[page_num] if b.type in ("text", "heading")]
        text = "\n".join(text_blocks)
        if text.strip():
            print(text[:1000])   # show first 1000 characters
        else:
            print("No text extracted (page may be blank)")

    # Step 2: Generate page profile (metadata with correct text_len for scanned pages)
    profiles = page_profile(pdf_path)   # now internally uses OCR for scanned pages

    print("\n" + "=" * 50)
    print("PAGE PROFILES (with correct text length)")
    print("=" * 50)
    for p in profiles:
        print(
            f"Page {p.page_number} | "
            f"kind={p.kind} | "
            f"text_len={p.text_len} | "
            f"images={len(p.images)} | "
            f"vectors={p.has_vector_graphics} | "
            f"table={p.table_hint}"
        )

    # Save blocks JSON for inspection
    os.makedirs("output/blocks", exist_ok=True)
    output_path = "output/blocks/scanned_blocks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in blocks], f, indent=2, default=str)
    print(f"\nSaved blocks to: {output_path}")

    # Also save page profiles to a readable JSON (already done by page_profile, but we can also print path)
    print(f"Page profiles saved to: output/page_profiles/")


if __name__ == "__main__":
    test_scanned_pages()