"""Tests for BrowseByEquipmentTool -- agent tool built on the machine/component/
doc_category chunk tags backend/chunking/chunk_tool.py::_structural_tags_from_block
propagates from folder_router.py. A metadata BROWSE (no embedding call), the
complementary half of cross-document correlation alongside FindRelatedDocumentsTool's
exact-ID matching -- this one answers "everything filed under this equipment", not
"what mentions this exact identifier".

Config-gated off by default (query.agent.equipment_browse.enabled) -- same posture
as query.answerer.image_ground, learned from a real incident where an always-on
doc_type/industry soft filter silently hid relevant chunks (search_documents.py).
"""
from unittest.mock import MagicMock, patch

from backend.retrieval.browse_by_equipment import BrowseByEquipmentTool

_ENABLED_CFG = {"query": {"agent": {"equipment_browse": {"enabled": True}}}}
_DISABLED_CFG = {"query": {"agent": {"equipment_browse": {"enabled": False}}}}
_UNSET_CFG = {"query": {"agent": {}}}


def _chunks():
    return [
        {"chunk_id": "c1", "document_id": "doc-manual", "text": "Replace the spindlehead bearing.",
         "source_ref": {"page": 12}, "machine": "120_CYLINDRICAL GRINDER", "component": "Spindlehead"},
        {"chunk_id": "c2", "document_id": "doc-cad", "text": "KE-MC000954-G spindlehead assembly drawing.",
         "source_ref": {"page": 1}, "machine": "120_CYLINDRICAL GRINDER", "component": "Spindlehead"},
    ]


def _fake_store(docs):
    store = MagicMock()
    store.list_documents.return_value = docs
    return store


def test_no_filters_returns_error():
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_ENABLED_CFG):
        result = BrowseByEquipmentTool().run()
    assert "error" in result


def test_disabled_by_default_returns_clear_error_not_empty():
    # Real design constraint: an empty result must never be indistinguishable from
    # "disabled" -- an agent seeing plain empty results could wrongly conclude the
    # equipment isn't in the corpus at all instead of that the tool is off.
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_DISABLED_CFG):
        result = BrowseByEquipmentTool().run(machine="120_CYLINDRICAL GRINDER")
    assert "error" in result
    assert "disabled" in result["error"].lower()
    assert "search_documents" in result["error"]


def test_disabled_when_config_key_entirely_absent():
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_UNSET_CFG):
        result = BrowseByEquipmentTool().run(machine="120_CYLINDRICAL GRINDER")
    assert "error" in result


def test_no_matches_returns_empty_with_note():
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.retrieval.vector_store.VectorStore.browse_by_filter", return_value=[]):
        result = BrowseByEquipmentTool().run(machine="NONEXISTENT")
    assert result["documents"] == []
    assert "note" in result
    assert "does NOT mean" in result["note"]


def test_groups_chunks_by_document_and_enriches_with_filename():
    docs = [
        {"document_id": "doc-manual", "filename": "Maintenance manual.pdf", "document_type": "manual"},
        {"document_id": "doc-cad", "filename": "KE-MC000954-G.pdf", "document_type": "cad_drawing"},
    ]
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.retrieval.vector_store.VectorStore.browse_by_filter", return_value=_chunks()), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(docs)):
        result = BrowseByEquipmentTool().run(machine="120_CYLINDRICAL GRINDER", component="Spindlehead")

    assert result["filters"] == {"machine": "120_CYLINDRICAL GRINDER", "component": "Spindlehead"}
    by_id = {d["document_id"]: d for d in result["documents"]}
    assert by_id["doc-manual"]["filename"] == "Maintenance manual.pdf"
    assert by_id["doc-cad"]["document_type"] == "cad_drawing"
    assert by_id["doc-manual"]["chunks"][0]["page"] == 12


def test_filters_passed_through_to_vector_store():
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.retrieval.vector_store.VectorStore.browse_by_filter", return_value=[]) as mock_browse:
        BrowseByEquipmentTool().run(doc_category="3.INSTRUCTION MANUAL")
    args, kwargs = mock_browse.call_args
    assert args[1] == {"doc_category": "3.INSTRUCTION MANUAL"}


def test_chunks_capped_per_document():
    chunks = [
        {"chunk_id": f"c{i}", "document_id": "doc-a", "text": f"chunk {i}",
         "source_ref": {"page": i}, "machine": "M"}
        for i in range(10)
    ]
    with patch("backend.retrieval.browse_by_equipment._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.retrieval.vector_store.VectorStore.browse_by_filter", return_value=chunks), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store([])):
        result = BrowseByEquipmentTool().run(machine="M")
    assert len(result["documents"][0]["chunks"]) == 5
