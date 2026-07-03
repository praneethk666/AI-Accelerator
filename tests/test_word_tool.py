import os
import pytest

from backend.extraction.word.tool import WordExtractorTool

WORD_FILE = "test-data/test.docx"

pytestmark = pytest.mark.skipif(
    not os.path.exists(WORD_FILE),
    reason="test-data fixture not present (git-ignored)",
)


def test_word_blocks_nonempty():
    state = WordExtractorTool().run({"file_path": WORD_FILE, "document_id": "test-doc-003"}, {})
    assert state["blocks"], "No blocks extracted from Word file"
    assert all(isinstance(b, dict) for b in state["blocks"]), "blocks must be plain dicts"


def test_word_errors_key_exists():
    state = WordExtractorTool().run({"file_path": WORD_FILE, "document_id": "test-doc-003"}, {})
    assert "errors" in state


def test_word_text_and_heading_blocks():
    state = WordExtractorTool().run({"file_path": WORD_FILE, "document_id": "test-doc-003"}, {})
    texts = [b for b in state["blocks"] if b["type"] == "text"]
    headings = [b for b in state["blocks"] if b["type"] == "heading"]
    assert texts, "No text blocks found"
    assert headings, "No heading blocks found"


def test_word_table_blocks():
    state = WordExtractorTool().run({"file_path": WORD_FILE, "document_id": "test-doc-003"}, {})
    tables = [b for b in state["blocks"] if b["type"] == "table"]
    assert tables, "No table blocks found"
    for t in tables:
        assert t["text"]
        assert "headers" in t["table_data"]
        assert "rows" in t["table_data"]


def test_word_image_blocks():
    state = WordExtractorTool().run({"file_path": WORD_FILE, "document_id": "test-doc-003"}, {})
    images = [b for b in state["blocks"] if b["type"] == "image_caption"]
    assert images, "No image blocks found"
    for img in images:
        meta = img.get("metadata") or {}
        assert meta.get("raw_image_path")
        assert meta.get("pending_vision") is True

        