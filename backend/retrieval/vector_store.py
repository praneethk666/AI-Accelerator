"""
backend/retrieval/vector_store.py
──────────────────────────────────
Thin adapter over Qdrant (the vector DB in Karthii's docker-compose stack).
Retrieval code calls VectorStore.get().search(...) — swapping the back-end
means changing only this file.

Connection settings come from environment variables (never hardcoded):
    QDRANT_HOST   default "localhost"
    QDRANT_PORT   default 6333
    QDRANT_COLLECTION  default "chunks"
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Singleton adapter. Call VectorStore.get() everywhere."""

    _instance: Optional["VectorStore"] = None

    def __init__(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        except ImportError as e:
            raise ImportError("qdrant-client required: pip install qdrant-client") from e

        self._client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )
        self._collection = os.getenv("QDRANT_COLLECTION", "chunks")
        logger.info("VectorStore connected: %s:%s / %s",
                    os.getenv("QDRANT_HOST", "localhost"),
                    os.getenv("QDRANT_PORT", 6333),
                    self._collection)

    @classmethod
    def get(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Dense cosine search.

        Parameters
        ----------
        query_vector : the query embedding
        top_k        : number of results
        filters      : dict of metadata key→value to filter on
                       (e.g. {"industry": "automotive", "doc_type": "circuit_diagram"})

        Returns
        -------
        list[Chunk] — ordered best-first
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=f"tags.{k}", match=MatchValue(value=v))
                for k, v in filters.items()
                if v is not None
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_hit_to_chunk(h) for h in hits]


# ── helpers ───────────────────────────────────────────────────────────────────

def _hit_to_chunk(hit) -> Chunk:
    """Convert a Qdrant ScoredPoint back into the shared Chunk schema."""
    payload = hit.payload or {}
    return Chunk(
        chunk_id=str(hit.id),
        document_id=payload.get("document_id", ""),
        text=payload.get("text", ""),
        tags=payload.get("tags", {}),
        source_ref=payload.get("source_ref", {}),
        embedding=None,   # don't round-trip the vector; caller doesn't need it
    )