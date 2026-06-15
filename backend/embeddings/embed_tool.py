"""Convert chunk text into dense and sparse vectors.

Reads chunks from state["chunks"].
Writes vector and sparse_vector into each chunk for later indexing.
Uses models from backend.core.models (cached singletons).
"""

from backend.core.tool import Tool, PipelineState
from backend.core.models import get_dense_model, get_sparse_model


class EmbedTool(Tool):
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        try:
            chunks = state.get("chunks", [])
            if not chunks:
                return state

            dense_model = get_dense_model(config)
            sparse_model = get_sparse_model(config)

            texts = [chunk["text"] for chunk in chunks]

            # Batch encode dense vectors
            dense_vectors = dense_model.encode(texts, normalize_embeddings=True)

            # Batch encode sparse vectors
            sparse_results = list(sparse_model.embed(texts))

            # Assign back
            for i, chunk in enumerate(chunks):
                vec = dense_vectors[i]
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                chunk["vector"] = vec

                sparse = sparse_results[i]
                chunk["sparse_vector"] = {
                    "indices": sparse.indices.tolist() if hasattr(sparse.indices, "tolist") else list(sparse.indices),
                    "values": sparse.values.tolist() if hasattr(sparse.values, "tolist") else list(sparse.values),
                }

            state["chunks"] = chunks
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool": self.name,
                "level": "error",
                "message": str(e),
                "block_id": None,
            })
        return state