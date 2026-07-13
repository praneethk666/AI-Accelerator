"""Dense-search adapter — thin layer over the canonical QdrantStore.

Retrieval reads through this static interface; the actual storage is
backend.storage.qdrant_store (dense leg) + backend.storage.postgres_store
(text/tag hydration). One wire format, one writer (IndexTool), one reader (here).

  VectorStore.search(query_vector, config, top_k, filters) -> list[chunk dict]

filters keys map to flat Qdrant payload fields (tags are flattened on write):
    {"industry": "automotive"}     -> exact match
    {"document_id": ["uuid1", ...]} -> any-of (document scope)
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.storage.postgres_store import PostgresStore
from backend.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class VectorStore:
    @staticmethod
    def search(
        query_vector: list[float],
        config: dict,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Dense cosine search, then hydrate full chunk rows from Postgres.

        Qdrant returns chunk_ids + scores (text lives only in Postgres); we fetch
        the rows by id and preserve Qdrant's score order.
        """
        dim        = config["embeddings"]["dense_dim"]
        collection = config["database"]["qdrant_collection"]

        store = QdrantStore(dim, collection)
        try:
            hits = store.search_dense(query_vector, filters=filters, top_n=top_k)
        finally:
            store.close()

        return _hydrate(hits)


def _hydrate(hits: list[dict]) -> list[dict]:
    """Fetch chunk text/tags from Postgres for the hit chunk_ids, in hit order."""
    chunk_ids = [h["chunk_id"] for h in hits if h.get("chunk_id")]
    if not chunk_ids:
        return []
    pg = PostgresStore()
    try:
        rows = pg.get_chunks_by_ids(chunk_ids)
    finally:
        pg.close()
    by_id = {str(r["chunk_id"]): r for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]
