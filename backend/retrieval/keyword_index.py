"""
backend/retrieval/keyword_index.py
───────────────────────────────────
BM25 keyword search — the sparse leg of hybrid retrieval.
Uses rank_bm25 (pure-Python, no external service needed for the benchmark).
In production this would be backed by Elasticsearch/OpenSearch.

The index is built once from the chunk corpus and cached in memory.
For the benchmark we build it from a list of Chunk objects; in production
Karthii's ingestion step would populate an external index.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class KeywordIndex:
    """Singleton BM25 index over the loaded chunk corpus."""

    _instance: Optional["KeywordIndex"] = None

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    @classmethod
    def get(cls) -> "KeywordIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from a list of Chunk objects.
        Call once per corpus; re-call if the corpus changes.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 required: pip install rank-bm25") from e

        self._chunks = chunks
        tokenized = [_tokenize(c.get("text") or "") for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("KeywordIndex built: %d chunks", len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        BM25 keyword search.

        Parameters
        ----------
        query    : raw query string (will be tokenized internally)
        top_k    : number of results to return
        filters  : optional metadata filter dict (applied post-scoring)

        Returns
        -------
        list[Chunk] ordered best-first
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("KeywordIndex is empty — call build() first")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (score, chunk) and apply metadata filters
        scored = [
            (scores[i], self._chunks[i])
            for i in range(len(self._chunks))
            if _passes_filter(self._chunks[i], filters)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric — simple but effective for BM25."""
    return re.findall(r"\w+", text.lower())


def _passes_filter(chunk: Chunk, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    tags = chunk.get("tags") or {}
    for key, val in filters.items():
        if val is not None and tags.get(key) != val:
            return False
    return True