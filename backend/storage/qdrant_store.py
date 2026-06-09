"""Qdrant vector store for chunk embeddings.

- one collection holds vectors + a tag payload (same tags as Postgres), keyed by
  chunk_id — one schema, category lives in the payload, never a collection per client
- write_chunk upserts; search does cosine similarity + an optional tag filter
- Postgres holds the text (source of truth); search here returns chunk_ids + scores,
  then text is fetched from Postgres by chunk_id
- endpoint comes from env (.env); never hardcode
"""

from __future__ import annotations

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


def url_from_env() -> str:
    """Qdrant endpoint from env, defaulting to the local docker-compose service."""
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def _point_id(chunk_id: str) -> str:
    # Qdrant point ids must be int or UUID; derive a stable UUID from any chunk_id
    # so upserts are idempotent. The real chunk_id is kept in the payload.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Thin wrapper over a Qdrant collection (vectors + tag payload)."""

    def __init__(
        self, dim: int, collection: str = "chunks", url: str | None = None
    ) -> None:
        self.dim = dim
        self.collection = collection
        self.client = QdrantClient(url=url or url_from_env())
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        # dim is config-driven (embeddings.dim); collection created once if absent
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def write_chunk(self, chunk: dict) -> None:
        """Upsert one chunk's vector + tag payload, keyed (stably) by chunk_id."""
        # tags flattened to top level so retrieval filters by plain keys (industry,
        # doc_type, ...); chunk_id/document_id kept explicit for the Postgres join.
        payload = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            **chunk.get("tags", {}),
        }
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=_point_id(chunk["chunk_id"]),
                    vector=chunk["vector"],
                    payload=payload,
                )
            ],
        )

    def search(
        self, query_vector: list[float], filters: dict | None = None, top_n: int = 5
    ) -> list[dict]:
        """Nearest chunks by cosine similarity, narrowed by exact tag matches.

        Returns chunk_id + score + tags; fetch the text from Postgres by chunk_id.
        """
        qfilter = None
        if filters:
            qfilter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )
        hits = self.client.query_points(
            self.collection,
            query=query_vector,
            query_filter=qfilter,
            limit=top_n,
            with_payload=True,
        ).points
        return [
            {"chunk_id": h.payload.get("chunk_id"), "score": h.score, "tags": h.payload}
            for h in hits
        ]

    def close(self) -> None:
        self.client.close()
