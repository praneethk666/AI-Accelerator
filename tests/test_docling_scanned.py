"""
Regression tests for scanned Docling + Vision fallback extraction.

Purpose
-------
Protect against regressions where:
1. Tables are flattened into text.
2. Pages disappear.
3. Blocks span multiple pages.
4. table_data is lost.

Runs the extraction ONCE for the whole test session.
"""

import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.extraction.docling_pdf.tool import DoclingPDFTool

PDF = "test-data/Digital_4pages_page-0001.pdf"

pytestmark = pytest.mark.skipif(
    not os.path.exists(PDF),
    reason="Regression PDF not present."
)


# ---------------------------------------------------------------------
# Run extraction ONCE
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def extracted_state():
    """
    Executes the real Docling pipeline only once.
    """

    tool = DoclingPDFTool()

    state = tool.run(
        {
            "file_path": PDF,
            "document_id": "pytest-docling-regression",
        },
        {},
    )

    assert "blocks" in state
    assert state["blocks"]

    return state


@pytest.fixture(scope="session")
def blocks(extracted_state):
    return extracted_state["blocks"]


# ---------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------

def test_blocks_nonempty(blocks):
    assert len(blocks) > 0


# ---------------------------------------------------------------------
# Ensure every page exists
# ---------------------------------------------------------------------

def test_all_pages_present(blocks):

    pages = sorted(
        {
            b["source_ref"]["page"]
            for b in blocks
            if b.get("source_ref")
        }
    )

    assert pages == [1, 2, 3, 4]


# ---------------------------------------------------------------------
# Every block has a valid page
# ---------------------------------------------------------------------

def test_valid_page_numbers(blocks):

    for block in blocks:

        assert block.get("source_ref")

        page = block["source_ref"]["page"]

        assert isinstance(page, int)

        assert 1 <= page <= 4


# ---------------------------------------------------------------------
# At least one table exists
# ---------------------------------------------------------------------

def test_table_blocks_exist(blocks):

    tables = [
        b
        for b in blocks
        if b["type"] == "table"
    ]

    assert tables, "No table blocks found."


# ---------------------------------------------------------------------
# Every table has structured table_data
# ---------------------------------------------------------------------

def test_table_data_present(blocks):

    tables = [
        b
        for b in blocks
        if b["type"] == "table"
    ]

    assert tables

    for table in tables:

        td = table["table_data"]

        assert td is not None

        assert "headers" in td

        assert "rows" in td

        assert isinstance(td["rows"], list)


# ---------------------------------------------------------------------
# Ensure no page disappears
# ---------------------------------------------------------------------

def test_every_page_has_blocks(blocks):

    counts = Counter(
        b["source_ref"]["page"]
        for b in blocks
    )

    for page in range(1, 5):

        assert counts[page] > 0


# ---------------------------------------------------------------------
# No block spans pages
# ---------------------------------------------------------------------

def test_blocks_do_not_span_pages(blocks):

    for block in blocks:

        page = block["source_ref"]["page"]

        assert isinstance(page, int)


# ---------------------------------------------------------------------
# Print extraction summary
# ---------------------------------------------------------------------

def test_summary(blocks):

    counts = Counter(
        b["source_ref"]["page"]
        for b in blocks
    )

    print("\n")
    print("=" * 60)
    print("DOCILING REGRESSION SUMMARY")
    print("=" * 60)

    for page in sorted(counts):

        page_blocks = [
            b for b in blocks
            if b["source_ref"]["page"] == page
        ]

        tables = [
            b for b in page_blocks
            if b["type"] == "table"
        ]

        texts = [
            b for b in page_blocks
            if b["type"] == "text"
        ]

        print(
            f"Page {page}"
            f" | Blocks={len(page_blocks)}"
            f" | Tables={len(tables)}"
            f" | Text={len(texts)}"
        )

    print("=" * 60)


if __name__ == "__main__":
    pytest.main(["-v", __file__])