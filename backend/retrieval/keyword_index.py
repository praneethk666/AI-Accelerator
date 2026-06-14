"""Sparse/BM25-search adapter — thin layer over the canonical QdrantStore.

The sparse (BM25) leg of hybrid retrieval. Sparse vectors live in Qdrant as a
named "sparse" vector (IDF modifier, server-side scoring) — written by IndexTool,
queried here. No in-memory index to build or keep warm.

  KeywordIndex.search(query, config, top_k, filters) -> list[chunk dict]

The query is encoded with the BM25 *query* embedding (query vs passage matters
for BM25), then Qdrant scores it against the stored passage vectors. Hits are
hydrated with text/tags from Postgres, in score order.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.models import get_sparse_model
from backend.retrieval.vector_store import _hydrate
from backend.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class KeywordIndex:
    @staticmethod
    def search(
        query: str,
        config: dict,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        sparse = get_sparse_model(config)
        q = next(iter(sparse.query_embed([query])))

        dim        = config["embeddings"]["dense_dim"]
        collection = config["database"]["qdrant_collection"]

        store = QdrantStore(dim, collection)
        try:
            hits = store.search_sparse(
                indices=q.indices.tolist(),
                values=q.values.tolist(),
                filters=filters,
                top_n=top_k,
            )
        finally:
            store.close()

        return _hydrate(hits)
