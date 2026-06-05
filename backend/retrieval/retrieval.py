"""
backend/retrieval/retrieval.py
──────────────────────────────
RetrievalTool — implements the Tool Protocol from backend/core/tool.py.

  run(state, config)
    reads  : state["sub_questions"]    list[str]  — from query-decomposer
             state["document_scope"]   list[str]  — optional doc_id filter
    writes : state["retrieved_chunks"] list[Chunk] — flat, deduped, best-first
    errors : appends to state["errors"] on failure; never raises

Five methods (config["query"]["retrieval"]["method"]):
  naive          — dense cosine search
  hybrid         — dense + BM25, fused with RRF
  hybrid_rerank  — hybrid + cross-encoder reranker
  hyde           — embed a hypothetical answer, then dense search
  enriched       — hybrid_rerank on topic/keyword-tagged corpus

All knobs live under config["query"]["retrieval"][...].
LLM, embedder, reranker come from backend/core/models.py factories:
  get_llm(config)          — LangChain chat model (Groq or Gemini)
  get_dense_model(config)  — SentenceTransformer
  get_reranker(config)     — CrossEncoder
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from backend.core.tool import PipelineState   # Tool is a Protocol — implement, don't subclass
from backend.core.schemas import Chunk
from backend.core.models import get_dense_model, get_reranker
from backend.core.llm_client import get_llm
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.keyword_index import KeywordIndex

logger = logging.getLogger(__name__)


class RetrievalTool:
    """
    Implements the Tool Protocol (backend/core/tool.py).

    Tool is a Protocol, not a base class — Python structural subtyping means
    this class satisfies it as long as it has `name: str` and
    `run(self, state, config) -> PipelineState`.

    State contract (matches tool.py PipelineState exactly):
        READS  state["sub_questions"]    list[str]   queries to retrieve for
               state["document_scope"]   list[str]   doc ids to restrict search (optional)
        WRITES state["retrieved_chunks"] list[Chunk] flat, deduped, scored desc
        ERRORS state["errors"]           list        append only, never raise
    """

    name: str = "retrieval"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        """
        Retrieve relevant chunks for all sub_questions and write
        a single flat, deduplicated list to state["retrieved_chunks"].

        De-duplication: if the same chunk_id appears across multiple
        sub-question result sets, keep the copy with the highest rank
        (i.e. first occurrence after per-method ordering).
        """
        sub_questions: list[str] = state.get("sub_questions") or []
        if not sub_questions:
            logger.warning("RetrievalTool: no sub_questions in state — skipping")
            state["retrieved_chunks"] = []
            return state

        retrieval_cfg = config.get("query", {}).get("retrieval", {})

        # Build filters: document_scope from state + any config-level filters
        base_filters: dict = dict(retrieval_cfg.get("filters") or {})
        doc_scope: list[str] = state.get("document_scope") or []
        # (doc_scope filtering is applied inside VectorStore.search via payload match)

        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for query in sub_questions:
            try:
                result = _retrieve_one(
                    query=query,
                    retrieval_cfg=retrieval_cfg,
                    full_config=config,
                    extra_filters=base_filters,
                    doc_scope=doc_scope,
                )
                logger.debug(
                    "RetrievalTool q=%r method=%s n=%d %.1fms",
                    query[:60], result["method"],
                    len(result["chunks"]), result["latency_ms"],
                )
                for chunk in result["chunks"]:
                    cid = chunk.get("chunk_id", "")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)

            except Exception as exc:
                logger.error("RetrievalTool failed for %r: %s", query[:60], exc)
                errors: list = state.get("errors") or []
                errors.append({
                    "tool":  "retrieval",
                    "query": query,
                    "error": str(exc),
                })
                state["errors"] = errors

        state["retrieved_chunks"] = all_chunks
        return state


# ── internal dispatcher ───────────────────────────────────────────────────────

def _retrieve_one(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    extra_filters: Optional[dict] = None,
    doc_scope: Optional[list[str]] = None,
) -> dict:
    """
    Run one query through the chosen method.

    Returns
    -------
    dict: { "chunks": list[Chunk], "latency_ms": float, "method": str }
    """
    method  = retrieval_cfg.get("method", "hybrid_rerank")
    filters = dict(extra_filters or {})
    # doc_scope is passed through to vector_store for payload filtering
    if doc_scope:
        filters["document_id"] = doc_scope   # VectorStore handles list vs scalar

    start = time.perf_counter()

    if method == "naive":
        chunks = _naive(query, retrieval_cfg, full_config, filters)
    elif method == "hybrid":
        chunks = _hybrid(query, retrieval_cfg, full_config, filters)
    elif method == "hybrid_rerank":
        chunks = _hybrid_rerank(query, retrieval_cfg, full_config, filters)
    elif method == "hyde":
        chunks = _hyde(query, retrieval_cfg, full_config, filters)
    elif method == "enriched":
        # "enriched" = same algorithm as hybrid_rerank but run on a corpus that
        # was tagged with topic/keywords before embedding. The algorithm itself
        # doesn't change; the benchmark compares unenriched vs enriched corpora.
        chunks = _hybrid_rerank(query, retrieval_cfg, full_config, filters)
    else:
        raise ValueError(
            f"Unknown retrieval method: {method!r}. "
            "Valid: naive | hybrid | hybrid_rerank | hyde | enriched"
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return {"chunks": chunks, "latency_ms": round(latency_ms, 2), "method": method}


# ── method implementations ────────────────────────────────────────────────────

def _naive(query: str, cfg: dict, full_config: dict, filters: dict) -> list[Chunk]:
    """Dense cosine similarity search only."""
    embedder = get_dense_model(full_config)
    q_emb    = embedder.encode(query, normalize_embeddings=True).tolist()
    return VectorStore.get().search(
        q_emb, top_k=cfg.get("top_k", 5), filters=filters
    )


def _hybrid(query: str, cfg: dict, full_config: dict, filters: dict) -> list[Chunk]:
    """Dense + BM25, scores fused with Reciprocal Rank Fusion."""
    top_k         = cfg.get("top_k", 5)
    dense_weight  = cfg.get("dense_weight", 0.6)
    sparse_weight = cfg.get("sparse_weight", 0.4)

    embedder    = get_dense_model(full_config)
    q_emb       = embedder.encode(query, normalize_embeddings=True).tolist()
    dense_hits  = VectorStore.get().search(q_emb, top_k=top_k * 2, filters=filters)
    sparse_hits = KeywordIndex.get().search(query, top_k=top_k * 2, filters=filters)

    return _rrf_fuse(
        dense_hits, sparse_hits,
        w_dense=dense_weight, w_sparse=sparse_weight,
        top_k=top_k,
    )


def _hybrid_rerank(query: str, cfg: dict, full_config: dict, filters: dict) -> list[Chunk]:
    """Hybrid + cross-encoder reranker. Fetches a wider candidate pool first."""
    top_k       = cfg.get("top_k", 5)
    candidate_k = cfg.get("candidate_k", top_k * 4)

    candidates = _hybrid(query, {**cfg, "top_k": candidate_k}, full_config, filters)

    reranker = get_reranker(full_config)
    pairs    = [(query, c.get("text") or "") for c in candidates]
    scores   = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


def _hyde(query: str, cfg: dict, full_config: dict, filters: dict) -> list[Chunk]:
    """
    HyDE (Hypothetical Document Embeddings).
    Ask the LLM for a hypothetical answer, embed that instead of the raw query.
    The hypothetical text sits closer in embedding space to real answer passages.
    """
    top_k = cfg.get("top_k", 5)

    llm          = get_llm(full_config)
    hypothetical = llm.invoke(
        "Write a short factual paragraph that directly answers this question. "
        "Reply with ONLY the paragraph, no preamble.\n\nQuestion: " + query
    ).content

    embedder = get_dense_model(full_config)
    hyp_emb  = embedder.encode(hypothetical, normalize_embeddings=True).tolist()

    return VectorStore.get().search(hyp_emb, top_k=top_k, filters=filters)


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_hits:  list[Chunk],
    sparse_hits: list[Chunk],
    w_dense:  float = 0.6,
    w_sparse: float = 0.4,
    top_k: int  = 5,
    k:     int  = 60,
) -> list[Chunk]:
    """
    Reciprocal Rank Fusion.
    score(d) = w_dense/(k + rank_dense) + w_sparse/(k + rank_sparse)
    Chunks in only one leg still get a partial score for that leg.
    """
    scores:    dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense_hits, start=1):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + w_dense * (1.0 / (k + rank))
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse_hits, start=1):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + w_sparse * (1.0 / (k + rank))
        chunk_map[cid] = chunk

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in ranked[:top_k]]