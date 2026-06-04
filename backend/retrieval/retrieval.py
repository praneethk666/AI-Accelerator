"""
backend/retrieval/retrieval.py
──────────────────────────────
Five RAG retrieval methods, all behind the same signature:

    retrieve(query, config) -> list[Chunk]

Methods
-------
1. naive          — pure dense (cosine) vector search
2. hybrid         — dense + BM25 keyword, score-fused (RRF)
3. hybrid_rerank  — hybrid + cross-encoder reranker
4. hyde           — HyDE: embed a hypothetical answer, then dense search
5. enriched       — hybrid_rerank on chunks that carry topic/keyword tags
                    (same algorithm as 3, but chunk corpus has richer metadata)

The active method is chosen by config["method"]; all other knobs
(top_k, model names, weights, …) also come from config, never hardcoded.

How it plugs in
---------------
Karthii's ingestion step writes chunks + embeddings to the vector store and
the relational DB.  This module only *reads* from those stores at query time.
Swap the storage back-end by changing the `VectorStore` / `KeywordIndex`
factories below without touching any retrieval logic.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional
import os
from backend.core.schemas import Chunk
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.keyword_index import KeywordIndex
from backend.retrieval.embedder import embed_text
from backend.retrieval.reranker import rerank
from backend.retrieval.llm_client import generate_hypothetical_answer

logger = logging.getLogger(__name__)


# ── public entry point ──────────────────────────────────────────────────────

def retrieve(query: str, config: dict) -> dict:
    """
    Retrieve the top-k most relevant chunks for *query*.

    Parameters
    ----------
    query   : the user's question
    config  : dict with at least:
        method          : str  — naive | hybrid | hybrid_rerank | hyde | enriched
        top_k           : int  — number of chunks to return
        filters         : dict — optional metadata filters (industry, doc_type, …)
        embedding_model : str  — sentence-transformers model id
        reranker_model  : str  — cross-encoder model id (for hybrid_rerank / enriched)
        dense_weight    : float — RRF weight for dense leg (hybrid methods)
        sparse_weight   : float — RRF weight for sparse leg (hybrid methods)
        hyde_model      : str  — LLM used to generate hypothetical answer

    Returns
    -------
    dict with keys:
        chunks   : list[Chunk]   — retrieved chunks (ordered by score desc)
        latency_ms : float       — wall-clock time for this call
        method   : str           — which method was used
    """
    method = config.get("method", "naive")
    start = time.perf_counter()

    if method == "naive":
        chunks = _naive(query, config)
    elif method == "hybrid":
        chunks = _hybrid(query, config)
    elif method == "hybrid_rerank":
        chunks = _hybrid_rerank(query, config)
    elif method == "hyde":
        chunks = _hyde(query, config)
    elif method == "enriched":
        # Same algo as hybrid_rerank but relies on enriched chunk metadata.
        # The "enriched" flag is a corpus property, not a retrieval algorithm change.
        # We keep it as a separate method name so the benchmark can compare
        # unenriched vs enriched corpora on the same algorithm.
        chunks = _hybrid_rerank(query, config)
    else:
        raise ValueError(f"Unknown retrieval method: {method!r}")

    latency_ms = (time.perf_counter() - start) * 1000
    logger.debug("retrieve method=%s top=%d latency=%.1f ms", method, len(chunks), latency_ms)

    return {
        "chunks": chunks,
        "latency_ms": round(latency_ms, 2),
        "method": method,
    }


# ── method implementations ───────────────────────────────────────────────────

def _naive(query: str, config: dict) -> list[Chunk]:
    """Dense cosine similarity search only."""
    top_k = config.get("top_k", 5)
    filters = config.get("filters", {})
    q_emb = embed_text(query, config["embedding_model"])
    store = VectorStore.get()
    return store.search(q_emb, top_k=top_k, filters=filters)


def _hybrid(query: str, config: dict) -> list[Chunk]:
    """
    Dense + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF).
    RRF score = Σ  1 / (k + rank_i)  for each retrieval leg.
    """
    top_k = config.get("top_k", 5)
    dense_weight = config.get("dense_weight", 0.6)
    sparse_weight = config.get("sparse_weight", 0.4)
    filters = config.get("filters", {})

    q_emb = embed_text(query, config["embedding_model"])
    store = VectorStore.get()
    idx = KeywordIndex.get()

    dense_hits = store.search(q_emb, top_k=top_k * 2, filters=filters)
    sparse_hits = idx.search(query, top_k=top_k * 2, filters=filters)

    return _rrf_fuse(dense_hits, sparse_hits,
                     w_dense=dense_weight, w_sparse=sparse_weight,
                     top_k=top_k)


def _hybrid_rerank(query: str, config: dict) -> list[Chunk]:
    """
    Hybrid retrieval followed by a cross-encoder reranker.
    Fetch a wider candidate set, then rerank and return top_k.
    """
    top_k = config.get("top_k", 5)
    candidate_k = config.get("candidate_k", top_k * 4)
    reranker_model = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Widen the candidate pool before reranking
    wide_config = {**config, "top_k": candidate_k}
    candidates = _hybrid(query, wide_config)

    reranked = rerank(query, candidates, model_name=reranker_model)
    return reranked[:top_k]


def _hyde(query: str, config: dict) -> list[Chunk]:
    """
    HyDE — Hypothetical Document Embeddings.
    1. Ask the LLM for a hypothetical answer to the query.
    2. Embed that hypothetical answer (not the query itself).
    3. Search with that embedding — it's closer to real answer passages.
    """
    top_k = config.get("top_k", 5)
    filters = config.get("filters", {})
    hyde_model = config.get("hyde_model",os.getenv("HYDE_MODEL", "llama-3.1-8b-instant"))

    hypothetical = generate_hypothetical_answer(query, model=hyde_model)
    hyp_emb = embed_text(hypothetical, config["embedding_model"])

    store = VectorStore.get()
    return store.search(hyp_emb, top_k=top_k, filters=filters)


# ── RRF fusion helper ────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_hits: list[Chunk],
    sparse_hits: list[Chunk],
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    top_k: int = 5,
    k: int = 60,
) -> list[Chunk]:
    """
    Reciprocal Rank Fusion.
    score(d) = w_dense * 1/(k + rank_dense) + w_sparse * 1/(k + rank_sparse)
    Chunks that only appear in one leg get a score of 0 for the missing leg.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense_hits, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + w_dense * (1.0 / (k + rank))
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse_hits, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + w_sparse * (1.0 / (k + rank))
        chunk_map[cid] = chunk

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in ranked[:top_k]]