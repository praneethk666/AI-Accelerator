import os
import sys

# Allow imports from project root
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from backend.extraction.pdf.detector import detect_pdf_type
from backend.extraction.pdf.page_profile import page_profile
from backend.extraction.pdf.digital import extract_digital

import json
from dataclasses import asdict


def test_digital():
    pdf = "test-data/Digital_40pages.pdf"

    overall, per_page = detect_pdf_type(pdf)

    print("=" * 50)
    print("PDF TYPE DETECTION")
    print("=" * 50)
    print(f"Overall: {overall}")
    print(f"Per page: {per_page}")

    blocks = extract_digital(
        pdf,
        document_id="test1"
    )

    print("\n" + "=" * 50)
    print("DIGITAL EXTRACTION")
    print("=" * 50)
    print(f"Total blocks: {len(blocks)}")

    for b in blocks[:5]:
        print(
            f"{b.type}: "
            f"{str(b.text)[:80]}"
        )


def test_page_profile():
    pdf = "test-data/Digital_40pages.pdf"

    profiles = page_profile(pdf)

    print("\n" + "=" * 50)
    print("PAGE PROFILE")
    print("=" * 50)

    print(
        json.dumps(
            [asdict(p) for p in profiles],
            indent=2
        )
    )


if __name__ == "__main__":
    test_digital()
    test_page_profile()