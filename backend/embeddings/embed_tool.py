"""embed tool — chunk text -> dense + sparse vectors. Replaces the hash stub.

- dense: config-selected embedder via get_dense_model (default BAAI/bge-m3, 1024-dim);
  documents carry the model-specific document prefix from config (empty for bge-m3,
  "search_document: " for nomic). Vectors are L2-normalized for cosine search.
- sparse: Qdrant/bm25 via get_sparse_model -> {"indices", "values"} for the BM25
  leg of hybrid retrieval (stored as a named sparse vector in Qdrant).
- writes chunk["vector"] + chunk["sparse_vector"] in place (Chunk schema fields).

Models load once (singletons in core.models). One bad chunk must not kill the
batch — encoding is per-batch, but the tool is wrapped by the graph's try/except.
"""

from __future__ import annotations

from backend.core.models import (
    get_dense_document_prefix,
    get_dense_model,
    get_sparse_model,
)
from backend.core.tool import PipelineState


class EmbedTool:
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        dense = get_dense_model(config)
        sparse = get_sparse_model(config)

        # Contextual retrieval: embed each chunk's enrichment (summary + keywords)
        # together with its text, so both vectors carry that context and match more
        # queries. The stored/displayed chunk["text"] is unchanged — only the
        # embedding INPUT is augmented.
        inputs = [_embed_input(c) for c in chunks]

        # dense: batch-encode with the (model-specific, config-driven) document prefix,
        # normalized for cosine. Empty prefix for bge-m3; nomic sets "search_document: ".
        # batch_size caps PEAK memory per sub-batch (validated live: on a 105-page,
        # 640-chunk document, MPS ran out of memory here — docling/YOLO/OCR/chunking
        # models loaded earlier in this same process had already consumed most of
        # Apple's MPS ceiling by the time embed ran, so bge-m3 had little headroom
        # left; a smaller batch_size plus freeing the MPS cache first gives it room).
        ecfg = config.get("embeddings") or {}
        batch_size = int(ecfg.get("dense_batch_size", 16))
        _free_mps_cache()
        doc_prefix = get_dense_document_prefix(config)
        dense_vecs = dense.encode(
            [doc_prefix + t for t in inputs],
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        # sparse: BM25 passage embeddings (passage vs query matters for BM25/IDF)
        sparse_vecs = list(sparse.passage_embed(inputs))

        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs):
            chunk["vector"] = dvec.tolist() if hasattr(dvec, "tolist") else list(dvec)
            chunk["sparse_vector"] = {
                "indices": svec.indices.tolist(),
                "values": svec.values.tolist(),
            }
        return state


def _free_mps_cache() -> None:
    """Best-effort: release PyTorch's cached (unfreed) MPS allocations from earlier
    steps in this process (docling/YOLO/OCR/chunking models) before bge-m3 asks for
    its own. No-op on CPU/CUDA or if torch isn't available — MPS-specific quirk
    where cached memory isn't returned to the pool between calls by default."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _embed_input(chunk: dict) -> str:
    """Embedding input = enrichment context (summary + keywords) prepended to the
    chunk text. Improves recall without changing the stored text."""
    tags = chunk.get("tags") or {}
    text = chunk.get("text") or ""
    summary = (tags.get("summary") or "").strip()
    keywords = tags.get("keywords") or []
    ctx = ""
    if summary:
        ctx += summary + "\n"
    if keywords:
        ctx += "Keywords: " + ", ".join(str(k) for k in keywords) + "\n"
    return (ctx + text).strip() or text
