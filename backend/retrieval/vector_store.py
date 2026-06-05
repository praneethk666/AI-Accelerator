"""
backend/retrieval/vector_store.py
──────────────────────────────────────────────────────────────────────────────
Qdrant adapter — the dense vector search back-end.

Connection settings are read from config YAML:
    config["database"]["qdrant_url"]
    config["database"]["qdrant_collection"]

Fallbacks keep the adapter usable in tests or partial configs.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Singleton Qdrant adapter. Call VectorStore.get() everywhere."""

    _instance: Optional["VectorStore"] = None

    def __init__(self, config: Optional[dict] = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("qdrant-client required: pip install qdrant-client") from e

        config = config or {}
        database = config.get("database", {})
        qdrant_url = database.get("qdrant_url")
        collection = database.get("qdrant_collection", "chunks")

        if qdrant_url:
            self._client = QdrantClient(url=qdrant_url)
            parsed = urlparse(qdrant_url)
            host = parsed.hostname or qdrant_url
            port = parsed.port or ""
        else:
            host = "localhost"
            port = 6333
            self._client = QdrantClient(host=host, port=port)

        self._collection = collection
        logger.info("VectorStore -> %s:%s / %s", host, port, self._collection)

    @classmethod
    def get(cls, config: Optional[dict] = None) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Dense cosine search.

        filters : e.g. {"industry": "automotive", "doc_type": "circuit_diagram"}
                  keys are matched against chunk tags stored as Qdrant payload fields.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=f"tags.{k}", match=MatchValue(value=v))
                for k, v in filters.items() if v is not None
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


def _hit_to_chunk(hit) -> Chunk:
    payload = hit.payload or {}
    return Chunk(
        chunk_id=str(hit.id),
        document_id=payload.get("document_id", ""),
        text=payload.get("text", ""),
        tags=payload.get("tags", {}),
        source_ref=payload.get("source_ref", {}),
        embedding=None,
    )
