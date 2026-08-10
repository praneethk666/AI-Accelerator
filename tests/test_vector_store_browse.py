"""Tests for VectorStore.browse_by_filter / QdrantStore.scroll_by_filter -- the
payload-only "find everything tagged X" query (no embedding, no similarity
ranking) added 10-Aug for backend/retrieval/browse_by_equipment.py. Distinct from
search_dense/search_sparse, which both require a query vector.
"""
from unittest.mock import MagicMock, patch

from backend.retrieval.vector_store import VectorStore
from backend.storage.qdrant_store import QdrantStore


def _record(payload):
    r = MagicMock()
    r.payload = payload
    return r


def test_scroll_by_filter_flattens_chunk_id_into_payload():
    with patch("backend.storage.qdrant_store.QdrantClient") as MockClient:
        client = MockClient.return_value
        client.collection_exists.return_value = True
        client.scroll.return_value = (
            [_record({"chunk_id": "c1", "document_id": "d1", "machine": "M"})],
            None,
        )
        store = QdrantStore(dim=1024, collection="chunks")
        out = store.scroll_by_filter({"machine": "M"}, limit=50)

    assert out == [{"chunk_id": "c1", "document_id": "d1", "machine": "M"}]


def test_scroll_by_filter_passes_filter_and_limit_to_client():
    with patch("backend.storage.qdrant_store.QdrantClient") as MockClient:
        client = MockClient.return_value
        client.collection_exists.return_value = True
        client.scroll.return_value = ([], None)
        store = QdrantStore(dim=1024, collection="chunks")
        store.scroll_by_filter({"machine": "M", "component": "Spindlehead"}, limit=25)

    _, kwargs = client.scroll.call_args
    assert kwargs["limit"] == 25
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False
    assert kwargs["scroll_filter"] is not None  # _build_filter produced a real Filter


def test_scroll_by_filter_no_vectors_requested():
    # Real point of this method: never pull vector payloads for a metadata browse.
    with patch("backend.storage.qdrant_store.QdrantClient") as MockClient:
        client = MockClient.return_value
        client.collection_exists.return_value = True
        client.scroll.return_value = ([], None)
        QdrantStore(dim=1024, collection="chunks").scroll_by_filter({"machine": "M"})
    assert client.scroll.call_args.kwargs["with_vectors"] is False


def test_vector_store_browse_by_filter_hydrates_from_postgres():
    config = {"embeddings": {"dense_dim": 1024}, "database": {"qdrant_collection": "chunks"}}
    with patch("backend.retrieval.vector_store.QdrantStore") as MockStore, \
         patch("backend.retrieval.vector_store.PostgresStore") as MockPg:
        MockStore.return_value.scroll_by_filter.return_value = [
            {"chunk_id": "c1", "document_id": "d1"},
        ]
        MockPg.return_value.get_chunks_by_ids.return_value = [
            {"chunk_id": "c1", "document_id": "d1", "text": "hydrated text"},
        ]
        out = VectorStore.browse_by_filter(config, {"machine": "M"}, limit=50)

    assert out[0]["text"] == "hydrated text"
    assert out[0]["_score"] == 0.0  # no similarity score for a metadata browse
    MockStore.return_value.close.assert_called_once()


def test_vector_store_browse_by_filter_empty_result_no_postgres_call():
    config = {"embeddings": {"dense_dim": 1024}, "database": {"qdrant_collection": "chunks"}}
    with patch("backend.retrieval.vector_store.QdrantStore") as MockStore, \
         patch("backend.retrieval.vector_store.PostgresStore") as MockPg:
        MockStore.return_value.scroll_by_filter.return_value = []
        out = VectorStore.browse_by_filter(config, {"machine": "NONEXISTENT"})

    assert out == []
    MockPg.return_value.get_chunks_by_ids.assert_not_called()
