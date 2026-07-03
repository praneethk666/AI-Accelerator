"""Qdrant vector store for chunk embeddings (hybrid: dense + sparse/BM25).

- one collection holds a NAMED dense vector ("dense") + a NAMED sparse vector
  ("sparse", BM25) + a tag payload — one schema, category lives in the payload,
  never a collection per client
- write_chunk upserts both legs; search_dense / search_sparse each return
  chunk_ids + scores, then text is hydrated from Postgres by chunk_id
- the sparse vector uses Qdrant's IDF modifier so BM25 scoring is server-side
- point id is a stable uuid5(chunk_id); the real chunk_id rides in the payload
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
    FilterSelector,
    MatchAny,
    MatchValue,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DENSE = "dense"
SPARSE = "sparse"


def url_from_env() -> str:
    """Qdrant endpoint from env, defaulting to the local docker-compose service."""
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def _point_id(chunk_id: str) -> str:
    # Qdrant point ids must be int or UUID; derive a stable UUID from any chunk_id
    # so upserts are idempotent. The real chunk_id is kept in the payload.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Thin wrapper over a Qdrant collection (named dense + sparse vectors)."""

    def __init__(
        self, dim: int, collection: str = "chunks", url: str | None = None
    ) -> None:
        self.dim = dim
        self.collection = collection
        self.client = QdrantClient(url=url or url_from_env())
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        # dim is config-driven (embeddings.dense_dim); collection created once.
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE: VectorParams(size=self.dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    # IDF modifier -> Qdrant computes BM25 IDF server-side
                    SPARSE: SparseVectorParams(
                        index=SparseIndexParams(), modifier=Modifier.IDF
                    ),
                },
            )

    def write_chunk(self, chunk: dict) -> None:
        """Upsert one chunk's dense + sparse vectors + tag payload (keyed by chunk_id)."""
        # tags flattened to top level so retrieval filters by plain keys (industry,
        # doc_type, ...); chunk_id/document_id kept explicit for the Postgres join.
        payload = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            **chunk.get("tags", {}),
        }

        vector: dict = {DENSE: chunk["vector"]}
        sparse = chunk.get("sparse_vector")
        if sparse and sparse.get("indices"):
            vector[SPARSE] = SparseVector(
                indices=sparse["indices"], values=sparse["values"]
            )

        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=_point_id(chunk["chunk_id"]),
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search_dense(
        self, query_vector: list[float], filters: dict | None = None, top_n: int = 5
    ) -> list[dict]:
        """Nearest chunks by cosine similarity on the dense leg, tag-filtered.

        Returns chunk_id + score + tags; fetch the text from Postgres by chunk_id.
        """
        hits = self.client.query_points(
            self.collection,
            query=query_vector,
            using=DENSE,
            query_filter=_build_filter(filters),
            limit=top_n,
            with_payload=True,
        ).points
        return _as_results(hits)

    def search_sparse(
        self, indices: list[int], values: list[float],
        filters: dict | None = None, top_n: int = 5,
    ) -> list[dict]:
        """BM25 search on the sparse leg (pass a query's sparse indices/values)."""
        hits = self.client.query_points(
            self.collection,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE,
            query_filter=_build_filter(filters),
            limit=top_n,
            with_payload=True,
        ).points
        return _as_results(hits)

    # back-compat alias: dense search was the original `search`
    search = search_dense

    def delete_by_document(self, document_id: str) -> None:
        """Delete every point belonging to a document (dense + sparse share one
        point). Makes re-ingestion idempotent — point ids come from regenerated
        chunk_ids, so stale points must be cleared before re-indexing."""
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(
                filter=_build_filter({"document_id": document_id})
            ),
        )

    def close(self) -> None:
        self.client.close()


def _build_filter(filters: dict | None) -> Filter | None:
    # tags are flattened to top-level payload keys (industry, doc_type, ...);
    # document_id is also top-level. A list value -> MatchAny (e.g. doc scope),
    # a scalar -> MatchValue.
    if not filters:
        return None
    conditions = []
    for key, val in filters.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(val))))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
    return Filter(must=conditions) if conditions else None


def _as_results(hits) -> list[dict]:
    return [
        {"chunk_id": h.payload.get("chunk_id"), "score": h.score, "tags": h.payload}
        for h in hits
    ]
