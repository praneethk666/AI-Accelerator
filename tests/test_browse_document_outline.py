"""Tests for BrowseDocumentOutlineTool -- agent tool exposing
backend/pipeline/outline_builder.py's document_outline table. Config-gated off by
default (query.agent.document_outline.enabled), same posture as
BrowseByEquipmentTool -- an empty/disabled result must always be a clear,
distinguishable message, never a silent empty list an agent could misread as
"this content doesn't exist".
"""
from unittest.mock import patch

from backend.retrieval.browse_document_outline import BrowseDocumentOutlineTool

_ENABLED_CFG = {"query": {"agent": {"document_outline": {"enabled": True}}}}
_DISABLED_CFG = {"query": {"agent": {"document_outline": {"enabled": False}}}}
_UNSET_CFG = {"query": {"agent": {}}}


def test_missing_document_id_returns_error():
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_ENABLED_CFG):
        result = BrowseDocumentOutlineTool().run(document_id="")
    assert "error" in result


def test_disabled_by_default_returns_clear_error_not_empty():
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_DISABLED_CFG):
        result = BrowseDocumentOutlineTool().run(document_id="doc-1")
    assert "error" in result
    assert "disabled" in result["error"].lower()
    assert "search_documents" in result["error"]


def test_disabled_when_config_key_entirely_absent():
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_UNSET_CFG):
        result = BrowseDocumentOutlineTool().run(document_id="doc-1")
    assert "error" in result


def test_no_outline_returns_empty_with_note():
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.pipeline.outline_builder.get_outline_children", return_value=[]):
        result = BrowseDocumentOutlineTool().run(document_id="doc-cad")
    assert result["children"] == []
    assert "note" in result
    assert "does NOT mean" in result["note"]


def test_returns_top_level_children_when_node_id_omitted():
    children = [{"node_id": "n0", "title": "1. CHANGING THE SETUP...", "level": 1,
                "page_start": 5, "page_end": 8}]
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.pipeline.outline_builder.get_outline_children", return_value=children) as mock_get:
        result = BrowseDocumentOutlineTool().run(document_id="doc-1")

    mock_get.assert_called_once_with("doc-1", None)
    assert result["children"] == children
    assert result["node_id"] is None


def test_descends_into_a_specific_node():
    children = [{"node_id": "n1", "title": "1.1 Replacing the Workpiece Holder", "level": 2,
                "page_start": 5, "page_end": 5}]
    with patch("backend.retrieval.browse_document_outline._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.pipeline.outline_builder.get_outline_children", return_value=children) as mock_get:
        result = BrowseDocumentOutlineTool().run(document_id="doc-1", node_id="n0")

    mock_get.assert_called_once_with("doc-1", "n0")
    assert result["node_id"] == "n0"
    assert result["children"][0]["title"] == "1.1 Replacing the Workpiece Holder"
