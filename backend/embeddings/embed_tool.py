"""embed tool — chunk text -> dense + sparse vectors. Replaces the hash stub.

- dense: nomic-embed-text-v1.5 (768-dim) via get_dense_model; documents carry the
  "search_document: " instruction prefix nomic expects (queries use the query
  prefix in retrieval). Vectors are L2-normalized for cosine search.
- sparse: Qdrant/bm25 via get_sparse_model -> {"indices", "values"} for the BM25
  leg of hybrid retrieval (stored as a named sparse vector in Qdrant).
- writes chunk["vector"] + chunk["sparse_vector"] in place (Chunk schema fields).

Models load once (singletons in core.models). One bad chunk must not kill the
batch — encoding is per-batch, but the tool is wrapped by the graph's try/except.
"""

from __future__ import annotations

from backend.core.models import DENSE_DOCUMENT_PREFIX, get_dense_model, get_sparse_model
from backend.core.tool import PipelineState


class EmbedTool:
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        dense = get_dense_model(config)
        sparse = get_sparse_model(config)

        texts = [(c.get("text") or "") for c in chunks]

        # dense: batch-encode with the nomic document prefix, normalized for cosine
        dense_vecs = dense.encode(
            [DENSE_DOCUMENT_PREFIX + t for t in texts],
            normalize_embeddings=True,
        )
        # sparse: BM25 passage embeddings (passage vs query matters for BM25/IDF)
        sparse_vecs = list(sparse.passage_embed(texts))

        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs):
            chunk["vector"] = dvec.tolist() if hasattr(dvec, "tolist") else list(dvec)
            chunk["sparse_vector"] = {
                "indices": svec.indices.tolist(),
                "values": svec.values.tolist(),
            }
        return state
