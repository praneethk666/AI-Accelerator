"""
backend/retrieval/keyword_index.py
──────────────────────────────────────────────────────────────────────────────
FastEmbed sparse keyword index — the sparse leg of hybrid retrieval.

The index is built once from the chunk corpus and cached in memory.
For the benchmark we build it from a list of Chunk objects; in production
Karthii's ingestion step would populate an external index.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastembed import SparseEmbedding, SparseTextEmbedding

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class KeywordIndex:
    """Singleton sparse index over the loaded chunk corpus."""

    _instance: Optional["KeywordIndex"] = None

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._sparse_model: Optional[SparseTextEmbedding] = None
        self._chunk_vectors: list[dict[int, float]] = []

    @classmethod
    def get(cls) -> "KeywordIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def build_from_pg(self, document_scope: Optional[list[str]] = None) -> None:
        """
        Production build — loads chunks from Postgres.
        Joins documents.status = 'ready' so in-progress ingestion
        jobs don't pollute the index.
        Re-call after new documents finish ingestion.
        """
        from backend.retrieval.pg_store import PGStore
        chunks = PGStore.get().fetch_all_ready_chunks(document_scope)
        self.build(chunks)
        logger.info("KeywordIndex.build_from_pg: %d chunks loaded", len(chunks))

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build the sparse index from a list of Chunk objects.
        Call once per corpus; re-call if the corpus changes.
        """
        self._chunks = chunks
        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        texts = [c.get("text") or "" for c in chunks]
        embeddings = list(self._sparse_model.passage_embed(texts))
        self._chunk_vectors = [_sparse_to_dict(embedding) for embedding in embeddings]
        logger.info("KeywordIndex built: %d chunks", len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Sparse keyword search.

        Parameters
        ----------
        query    : raw query string (will be tokenized internally)
        top_k    : number of results to return
        filters  : optional metadata filter dict (applied post-scoring)

        Returns
        -------
        list[Chunk] ordered best-first
        """
        if self._sparse_model is None or not self._chunks:
            logger.warning("KeywordIndex is empty — call build() first")
            return []

        query_vec = _sparse_to_dict(next(self._sparse_model.query_embed(query)))
        scores = [
            _dot_product(query_vec, chunk_vec)
            for chunk_vec in self._chunk_vectors
        ]

        scored = [
            (scores[i], self._chunks[i])
            for i in range(len(self._chunks))
            if _passes_filter(self._chunks[i], filters)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def _sparse_to_dict(embedding: SparseEmbedding) -> dict[int, float]:
    """Convert a FastEmbed sparse vector into a plain index -> weight map."""
    return {
        int(index): float(value)
        for index, value in zip(embedding.indices, embedding.values)
    }


def _dot_product(left: dict[int, float], right: dict[int, float]) -> float:
    """Compute sparse dot product for FastEmbed sparse vectors."""
    if len(left) > len(right):
        left, right = right, left

    return sum(weight * right.get(index, 0.0) for index, weight in left.items())


def _passes_filter(chunk: Chunk, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    tags = chunk.get("tags") or {}
    for k, v in filters.items():
        if v is not None and tags.get(k) != v:
            return False
    return True
