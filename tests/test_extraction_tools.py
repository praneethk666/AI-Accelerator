"""Test extraction tools for any PDF (digital/scanned/mixed) using automatic detection."""

# Suppress third-party deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paddle")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*imghdr.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")

import pytest
from unittest.mock import patch

from backend.extraction.detector import detect_pdf_type
from backend.extraction.digital_pdf.tool import PDFDigitalTool
from backend.extraction.scanned_pdf.tool import ScannedPDFTool
from backend.extraction.mixed_pdf.tool import MixedPDFTool


def get_tool_for_pdf(pdf_path: str):
    """Return the appropriate tool instance based on the PDF's overall type."""
    overall_type, _ = detect_pdf_type(pdf_path)
    if overall_type == "digital":
        return PDFDigitalTool()
    elif overall_type == "scanned":
        return ScannedPDFTool()
    else:
        return MixedPDFTool()


# Patch the YOLO model loader for all tests to avoid DLL crash on Windows
@pytest.fixture(autouse=True)
def mock_yolo():
    with patch("backend.extraction.scanned_pdf.scanned.get_yolo_model", return_value=None):
        yield


# Test all three PDF types automatically
@pytest.mark.parametrize("pdf_path", [
    "test-data/Digital_40pages.pdf",
    "test-data/Scanned_22pages.pdf",
    "test-data/Mixed.pdf",
])
def test_pdf_extraction(pdf_path):
    tool = get_tool_for_pdf(pdf_path)
    state = {"file_path": pdf_path, "document_id": "test"}
    result = tool.run(state, {})

    assert "blocks" in result
    assert "page_profiles" in result
    assert isinstance(result["blocks"], list)
    assert isinstance(result["page_profiles"], list)
    assert len(result["blocks"]) > 0
    assert len(result["page_profiles"]) > 0