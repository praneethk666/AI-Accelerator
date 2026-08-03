from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.retrieval.get_page_context import GetPageContextTool


def _fake_store(blocks):
    store = MagicMock()
    store.get_blocks.return_value = blocks
    return store


def test_returns_all_blocks_on_the_requested_page_joined_in_order():
    blocks = [
        {"block_id": "b1", "type": "heading", "text": "Alarm code\n11H",
         "source_ref": {"page": 61}},
        {"block_id": "b2", "type": "heading", "text": "State when alarm occurred.",
         "source_ref": {"page": 61}},
        {"block_id": "b3", "type": "table", "text": "| Factor 1 | ... |",
         "source_ref": {"page": 61}},
        {"block_id": "b4", "type": "text", "text": "unrelated page 62 content",
         "source_ref": {"page": 62}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store(blocks)):
        result = GetPageContextTool().run(document_id="doc-1", page=61)

    assert result["document_id"] == "doc-1"
    assert result["page"] == 61
    assert "Alarm code" in result["content"]
    assert "11H" in result["content"]
    assert "State when alarm occurred." in result["content"]
    assert "Factor 1" in result["content"]
    assert "page 62" not in result["content"]  # other pages excluded


def test_page_as_string_is_coerced_to_int():
    blocks = [{"block_id": "b1", "type": "text", "text": "hello",
               "source_ref": {"page": 5}}]
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store(blocks)):
        result = GetPageContextTool().run(document_id="doc-1", page="5")
    assert result["page"] == 5
    assert "hello" in result["content"]


def test_missing_document_id_or_page_returns_error():
    assert "error" in GetPageContextTool().run(document_id="", page=1)
    assert "error" in GetPageContextTool().run(document_id="doc-1", page=None)


def test_no_blocks_for_document_returns_error():
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store([])):
        result = GetPageContextTool().run(document_id="doc-1", page=1)
    assert "error" in result


def test_page_not_found_lists_available_pages():
    blocks = [
        {"block_id": "b1", "type": "text", "text": "x", "source_ref": {"page": 1}},
        {"block_id": "b2", "type": "text", "text": "y", "source_ref": {"page": 3}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store(blocks)):
        result = GetPageContextTool().run(document_id="doc-1", page=99)
    assert "error" in result
    assert result["pages_available"] == [1, 3]


def test_slide_matching_in_get_page_context():
    blocks = [
        {"block_id": "b1", "type": "text", "text": "slide 4 content",
         "source_ref": {"slide": 4}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store(blocks)):
        result = GetPageContextTool().run(document_id="doc-1", page=4)
    assert result["page"] == 4
    assert "slide 4 content" in result["content"]


def test_sheet_matching_in_get_page_context():
    blocks = [
        {"block_id": "b1", "type": "table", "text": "sheet data",
         "source_ref": {"sheet": "Overview"}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore",
              return_value=_fake_store(blocks)):
        result = GetPageContextTool().run(document_id="doc-1", page="Overview")
    assert result["page"] == "Overview"
    assert "sheet data" in result["content"]


if __name__ == "__main__":
    test_returns_all_blocks_on_the_requested_page_joined_in_order()
    test_page_as_string_is_coerced_to_int()
    test_missing_document_id_or_page_returns_error()
    test_no_blocks_for_document_returns_error()
    test_page_not_found_lists_available_pages()
    test_slide_matching_in_get_page_context()
    test_sheet_matching_in_get_page_context()
    print("get_page_context tests passed")
