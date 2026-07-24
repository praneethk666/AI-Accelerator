"""index tool — write chunks to both stores. Replaces IndexStub.

- Postgres holds text + tags (source of truth); Qdrant holds the vector + tag
  payload — same chunk_id keys both, so retrieval can join them
- one connection per run (skeleton-simple); both closed in finally
- a store failure raises -> the graph catches it into state["errors"] (run continues)
- embed runs before index, so chunk["vector"] is present (no fallback embedding)
"""

import logging

from backend.core.tool import PipelineState
from backend.storage.postgres_store import PostgresStore
from backend.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IndexTool:
    name = "index"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        dim = config.get("embeddings", {}).get("dense_dim", 768)
        collection = config.get("database", {}).get("qdrant_collection", "chunks")
        chunks = state.get("chunks", [])

        # Text always lands in Postgres (source of truth). A chunk only goes to
        # Qdrant if it has a vector — if embed partially failed, we skip the
        # vector write (and note it) rather than KeyError the whole step.
        skipped = 0
        # QdrantStore() makes a live network call in __init__; construct it INSIDE the
        # try that owns pg.close() so a Qdrant outage can't leak the Postgres connection.
        pg = PostgresStore()
        try:
            if not pg.document_exists(state.get("document_id")):
                logger.warning("Document %s was deleted mid-ingestion; aborting index step.", state.get("document_id"))
                return state
                
            vectors = QdrantStore(dim, collection)
            try:
                for chunk in chunks:
                    pg.write_chunk(chunk)
                    if chunk.get("vector"):
                        vectors.write_chunk(chunk)
                    else:
                        skipped += 1
            finally:
                vectors.close()
        finally:
            pg.close()

        if skipped:
            msg = f"index: {skipped}/{len(chunks)} chunks had no vector (embed failed?) — text stored, not vector-indexed"
            logger.warning(msg)
            state.setdefault("errors", []).append(msg)
        return state
