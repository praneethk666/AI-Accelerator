"""index tool — write chunks to both stores. Replaces IndexStub.

- Postgres holds text + tags (source of truth); Qdrant holds the vector + tag
  payload — same chunk_id keys both, so retrieval can join them
- one connection per run (skeleton-simple); both closed in finally
- a store failure raises -> the graph catches it into state["errors"] (run continues)
- embed runs before index, so chunk["vector"] is present (no fallback embedding)
"""

from __future__ import annotations

from backend.core.tool import PipelineState
from backend.storage.postgres_store import PostgresStore
from backend.storage.qdrant_store import QdrantStore


class IndexTool:
    name = "index"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        dim = config.get("embeddings", {}).get("dense_dim", 768)
        collection = config.get("database", {}).get("qdrant_collection", "chunks")
        pg = PostgresStore()
        vectors = QdrantStore(dim, collection)
        try:
            for chunk in state.get("chunks", []):
                pg.write_chunk(chunk)
                vectors.write_chunk(chunk)
        finally:
            pg.close()
            vectors.close()
        return state
