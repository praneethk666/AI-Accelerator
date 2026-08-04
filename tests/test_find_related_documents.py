"""Tests for FindRelatedDocumentsTool -- agent tool built on backend/categorize/
id_graph.py's exact drawing/CAD-sheet ID matching. Fills a real gap: existing
agent tools (search_documents, get_page_context, view_page_image) all operate
WITHIN one document's semantic content; this is the one that answers "what OTHER
documents mention this exact drawing number", which is what actually correlates
a manual, a CAD drawing, and a circuit diagram about the same physical component.
"""
from unittest.mock import MagicMock, patch

from backend.retrieval.find_related_documents import FindRelatedDocumentsTool


def _id_rows():
    return [
        {"document_id": "doc-cad", "block_id": "b1", "type": "text",
         "text": "SPINDLE ASSEMBLY MS03AAA789AB", "source_ref": {"page": 1}},
        {"document_id": "doc-manual", "block_id": "b2", "type": "text",
         "text": "See drawing MS03AAA789AB for seal replacement.", "source_ref": {"page": 42}},
    ]


def _fake_store(docs):
    store = MagicMock()
    store.list_documents.return_value = docs
    return store


def test_missing_identifier_returns_error():
    result = FindRelatedDocumentsTool().run(identifier="")
    assert "error" in result


def test_no_matches_returns_empty_with_note():
    with patch("backend.categorize.id_graph.find_documents_by_id", return_value=[]):
        result = FindRelatedDocumentsTool().run(identifier="MS03AAA789AB")
    assert result["documents"] == []
    assert "note" in result


def test_groups_mentions_by_document_and_enriches_with_filename():
    docs = [
        {"document_id": "doc-cad", "filename": "MS03AAA789AB-spindle assembly.pdf",
         "document_type": "cad_drawing"},
        {"document_id": "doc-manual", "filename": "Maintenance manual.pdf",
         "document_type": "manual"},
    ]
    with patch("backend.categorize.id_graph.find_documents_by_id", return_value=_id_rows()), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(docs)):
        result = FindRelatedDocumentsTool().run(identifier="MS03AAA789AB")

    assert result["identifier"] == "MS03AAA789AB"
    by_id = {d["document_id"]: d for d in result["documents"]}
    assert by_id["doc-cad"]["filename"] == "MS03AAA789AB-spindle assembly.pdf"
    assert by_id["doc-cad"]["document_type"] == "cad_drawing"
    assert by_id["doc-manual"]["filename"] == "Maintenance manual.pdf"
    assert by_id["doc-manual"]["mentions"][0]["page"] == 42


def test_mentions_capped_per_document():
    rows = [
        {"document_id": "doc-a", "block_id": f"b{i}", "type": "text",
         "text": f"mention {i}", "source_ref": {"page": i}}
        for i in range(10)
    ]
    with patch("backend.categorize.id_graph.find_documents_by_id", return_value=rows), \
         patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store([])):
        result = FindRelatedDocumentsTool().run(identifier="X")

    assert len(result["documents"][0]["mentions"]) == 5


def test_identifier_whitespace_stripped():
    with patch("backend.categorize.id_graph.find_documents_by_id",
              return_value=[]) as mock_find:
        FindRelatedDocumentsTool().run(identifier="  MS03AAA789AB  ")
    mock_find.assert_called_once_with("MS03AAA789AB")
