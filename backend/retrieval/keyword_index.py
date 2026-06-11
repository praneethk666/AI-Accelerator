"""
backend/retrieval/keyword_index.py
──────────────────────────────────────────────────────────────────────────────
FastEmbed sparse keyword index — the sparse leg of hybrid retrieval.

The index is built once from the chunk corpus and cached as class-level state.
config is always passed at call time — never stored during init.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastembed import SparseEmbedding

from backend.core.models import get_sparse_model
from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class KeywordIndex:
    """Singleton sparse index over the loaded chunk corpus."""

    _chunks: list[Chunk] = []
    _sparse_model = None
    _chunk_vectors: list[dict[int, float]] = []

    @classmethod
    def get(cls, config: dict) -> "type[KeywordIndex]":
        if cls._sparse_model is None:
            cls._sparse_model = get_sparse_model(config)
        return cls

    @classmethod
    def build_from_pg(cls, config: dict, document_scope: Optional[list[str]] = None) -> None:
        """
        Production build — loads chunks from Postgres.
        Joins documents.status = 'ready' so in-progress ingestion
        jobs don't pollute the index.
        Re-call after new documents finish ingestion.
        """
        from backend.retrieval.pg_store import PGStore
        chunks = PGStore.fetch_all_ready_chunks(config, document_scope)
        cls.build(config, chunks)
        logger.info("KeywordIndex.build_from_pg: %d chunks loaded", len(chunks))

    @classmethod
    def build(cls, config: dict, chunks: list[Chunk]) -> None:
        """
        Build the sparse index from a list of Chunk objects.
        Call once per corpus; re-call if the corpus changes.
        """
        if cls._sparse_model is None:
            cls._sparse_model = get_sparse_model(config)
        cls._chunks = chunks
        texts = [c["text"] or "" for c in chunks]
        embeddings = list(cls._sparse_model.passage_embed(texts))
        cls._chunk_vectors = [_sparse_to_dict(embedding) for embedding in embeddings]
        logger.info("KeywordIndex built: %d chunks", len(chunks))

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int,
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
        if cls._sparse_model is None or not cls._chunks:
            logger.warning("KeywordIndex is empty — call build() first")
            return []

        query_vec = _sparse_to_dict(next(cls._sparse_model.query_embed(query)))
        scores = [
            _dot_product(query_vec, chunk_vec)
            for chunk_vec in cls._chunk_vectors
        ]

        scored = [
            (scores[i], cls._chunks[i])
            for i in range(len(cls._chunks))
            if _passes_filter(cls._chunks[i], filters)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def _sparse_to_dict(embedding: SparseEmbedding) -> dict[int, float]:
    return {
        int(index): float(value)
        for index, value in zip(embedding.indices, embedding.values)
    }


def _dot_product(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(index, 0.0) for index, weight in left.items())


def _passes_filter(chunk: Chunk, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    tags = chunk["tags"] or {}
    for k, v in filters.items():
        if v is not None and tags[k] != v:
            return False
    return True