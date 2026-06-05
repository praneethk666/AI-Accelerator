"""
backend/retrieval/vector_store.py
──────────────────────────────────
Qdrant adapter for dense vector search.
 
Split storage pattern (matches init_db.sql design):
  Qdrant  — stores vectors + minimal payload (chunk_id, tags for filtering)
  Postgres — stores full text, source_ref, table_data, image_path
 
So search() does:
  1. Ask Qdrant for top-k matching chunk_ids
  2. Call PGStore.fetch_by_ids() to get full chunk data
  3. Return hydrated Chunk list
 
This avoids duplicating large text blobs in both stores.
 
Connection settings from environment:
    QDRANT_HOST        default "localhost"
    QDRANT_PORT        default 6333
    QDRANT_COLLECTION  default "chunks"
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

from backend.core.schemas import Chunk
from backend.retrieval.pg_store import PGStore

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
 
        Step 1: Qdrant returns top-k chunk_ids + scores
        Step 2: PGStore hydrates full Chunk objects from Postgres
 
        filters keys map to Qdrant payload fields stored as tags.*:
            {"industry": "automotive"}  →  tags.industry == "automotive"
            {"document_id": ["uuid1"]}  →  for document_scope filtering
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
 
        qdrant_filter = _build_qdrant_filter(filters)
 
        hits = self._client.search(
            collection_name=self._collection,
            query_vector   =query_vector,
            limit          =top_k,
            query_filter   =qdrant_filter,
            with_payload   =False,   # only need IDs; text comes from Postgres
            with_vectors   =False,
        )
 
        if not hits:
            return []
 
        chunk_ids = [str(h.id) for h in hits]
 
        # Hydrate from Postgres — preserves Qdrant score order
        chunk_map = {c["chunk_id"]: c for c in PGStore.get().fetch_by_ids(chunk_ids)}
        return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
            


def _build_qdrant_filter(filters: Optional[dict]):
    """
    Convert a flat filter dict to a Qdrant Filter object.
 
    Handles:
        {"industry": "automotive"}        → FieldCondition / MatchValue
        {"document_id": ["uuid1","uuid2"]}→ FieldCondition / MatchAny (list)
    """
    if not filters:
        return None
 
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
 
    conditions = []
    for key, val in filters.items():
        if val is None:
            continue
        field = f"tags.{key}" if key not in ("document_id",) else key
        if isinstance(val, list):
            conditions.append(FieldCondition(key=field, match=MatchAny(any=val)))
        else:
            conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))
 
    return Filter(must=conditions) if conditions else None