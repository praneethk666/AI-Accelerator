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

    @staticmethod
    def get_client(config: dict):

        from qdrant_client import QdrantClient

        qdrant_url = config["database"]["qdrant_url"]

        if qdrant_url:
            return QdrantClient(url=qdrant_url)

        return QdrantClient(
            host=config["database"]["qdrant_host"],
            port=config["database"]["qdrant_port"],
        )

    def search(
        query_vector: list[float],
        config: dict,
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
 
        client = VectorStore.get_client(config)
 
        hits = client.search(
        collection_name=config["database"]["qdrant_collection"],
        query_vector=query_vector,
        limit=top_k,
        query_filter=_build_qdrant_filter(filters),
        with_payload=False,
        with_vectors=False,
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
