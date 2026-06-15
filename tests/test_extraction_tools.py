"""Test extraction tools for any PDF (digital/scanned/mixed) using automatic detection."""

import os
import json
import time
import pytest
from unittest.mock import patch

from backend.extraction.detector import detect_pdf_type
from backend.extraction.digital_pdf.tool import PDFDigitalTool
from backend.extraction.scanned_pdf.tool import ScannedPDFTool
from backend.extraction.mixed_pdf.tool import MixedPDFTool


OUTPUT_DIR = "output"


def get_tool_for_pdf(pdf_path: str):
    """Return the appropriate tool instance based on the PDF's overall type."""
    overall_type, _ = detect_pdf_type(pdf_path)
    if overall_type == "digital":
        return PDFDigitalTool()
    elif overall_type == "scanned":
        return ScannedPDFTool()
    else:
        return MixedPDFTool()


def _block_to_dict(block):
    """Convert a NormalizedBlock (pydantic model or plain object) to a dict."""
    if hasattr(block, "model_dump"):       # pydantic v2
        return block.model_dump()
    if hasattr(block, "dict"):             # pydantic v1
        return block.dict()
    if hasattr(block, "__dict__"):
        return dict(block.__dict__)
    return block


def save_blocks_to_json(blocks, pdf_path: str, document_id: str):
    """Save normalized blocks as a JSON file in the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{pdf_stem}_{document_id}_blocks.json")

    blocks_data = [_block_to_dict(b) for b in blocks]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blocks_data, f, indent=2, ensure_ascii=False, default=str)

    return out_path


# Patch the YOLO model loader for all tests to avoid DLL crash on Windows
@pytest.fixture(autouse=True)
def mock_yolo():
    with patch("backend.extraction.scanned_pdf.scanned.get_yolo_model", return_value=None):
        yield


# Test all three PDF types automatically
@pytest.mark.parametrize("pdf_path", [
    "test-data/Scanned_22pages.pdf",
])
def test_pdf_extraction(pdf_path):
    # ------------------------------------------------------------
    # 1. Detect total number of pages (for progress awareness)
    # ------------------------------------------------------------
    _, per_page_types = detect_pdf_type(pdf_path)
    total_pages = len(per_page_types)
    print(f"\n📄 Processing PDF: {pdf_path} ({total_pages} pages)")

    # ------------------------------------------------------------
    # 2. Run extraction with timing
    # ------------------------------------------------------------
    tool = get_tool_for_pdf(pdf_path)
    document_id = "test"
    state = {"file_path": pdf_path, "document_id": document_id}

    start_time = time.perf_counter()
    result = tool.run(state, {})
    extract_time = time.perf_counter() - start_time

    # ------------------------------------------------------------
    # 3. Extract results
    # ------------------------------------------------------------
    blocks = result["blocks"]
    profiles = result["page_profiles"]

    # ------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------
    print(f"\n✅ Extraction completed in {extract_time:.2f} seconds")
    print(f"   - Pages processed: {len(profiles)}")
    print(f"   - Normalized blocks created: {len(blocks)}")
    print("   (Page‑by‑page progress is printed by the extraction tool itself)")

    # ------------------------------------------------------------
    # 5. Save output
    # ------------------------------------------------------------
    out_path = save_blocks_to_json(blocks, pdf_path, document_id)
    print(f"   - Blocks saved to: {out_path}")

    # ------------------------------------------------------------
    # 6. Assertions
    # ------------------------------------------------------------
    assert "blocks" in result
    assert "page_profiles" in result
    assert isinstance(blocks, list)
    assert isinstance(profiles, list)
    assert len(blocks) > 0
    assert len(profiles) > 0
    assert os.path.exists(out_path)