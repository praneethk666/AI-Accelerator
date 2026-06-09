"""embed tool — chunk text -> embedding vector. Replaces EmbedStub.

- reads embeddings.dim/model from config (config-driven, swappable)
- writes chunk["vector"] in place (matches the Chunk.vector schema field)
"""

from __future__ import annotations

from backend.core.tool import PipelineState
from backend.embeddings.local_embedder import DEFAULT_DIM, LocalEmbedder


class EmbedTool:
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        dim = config.get("embeddings", {}).get("dim", DEFAULT_DIM)
        embedder = LocalEmbedder(dim)  # model switch goes here later
        for chunk in state.get("chunks", []):
            chunk["vector"] = embedder.embed(chunk.get("text", ""))
        return state
