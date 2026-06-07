"""index tool — write chunks to Postgres+pgvector. Replaces IndexStub.

- one connection per run (skeleton-simple); closed in finally
- a DB failure raises -> the graph catches it into state["errors"] (run continues)
"""

from __future__ import annotations

from backend.core.tool import PipelineState
from backend.embeddings.local_embedder import DEFAULT_DIM, LocalEmbedder
from backend.storage.postgres_store import PostgresStore


class IndexTool:
    name = "index"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        dim = config.get("embeddings", {}).get("dim", DEFAULT_DIM)
        store = PostgresStore(dim)
        try:
            fallback = LocalEmbedder(dim)  # embed step should run first; guard anyway
            for chunk in state.get("chunks", []):
                vector = chunk.get("embedding") or fallback.embed(chunk.get("text", ""))
                store.write_chunk(chunk, vector)
        finally:
            store.close()
        return state
