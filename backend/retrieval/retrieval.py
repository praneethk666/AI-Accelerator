"""
backend/retrieval/retrieval.py
──────────────────────────────
RetrievalTool — implements the Tool Protocol from backend/core/tool.py.

  run(state, config)
    READS  state["sub_questions"]    list[str]   — from query-decomposer
           state["document_scope"]   list[str]   — doc_ids to restrict search
           state["skip_retrieval"]   bool        — set by adaptive_router
    WRITES state["retrieved_chunks"] list[Chunk] — flat, deduped, best-first
    ERRORS state["errors"]           list        — append only, never raise

PostgreSQL integration (scripts/init_db.sql):
  - document_scope  → filters Qdrant + BM25 to only chunks in those docs
  - skip_retrieval  → if True, loads ALL chunks for scope from Postgres
                      directly (no vector search) so adaptive_router can
                      put the full document in context
  - industry filter → auto-populated from documents.industry in Postgres
                      when document_scope is a single known document

Five methods (config["query"]["retrieval"]["method"]):
  naive | hybrid | hybrid_rerank | hyde | enriched
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from backend.core.tool import PipelineState
from backend.core.schemas import Chunk
from backend.core.models import get_dense_model, get_reranker
from backend.core.llm_client import get_llm
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.keyword_index import KeywordIndex
from backend.retrieval.pg_store import PGStore

logger = logging.getLogger(__name__)


class RetrievalTool:
    """
    Implements the Tool Protocol (backend/core/tool.py).
    name + run(state, config) → satisfies the Protocol structurally.

    State contract (tool.py PipelineState):
        READS  sub_questions    list[str]
               document_scope  list[str]   ← doc_id filter, empty = all docs
               skip_retrieval  bool        ← True = bypass vector search
        WRITES retrieved_chunks list[Chunk] ← flat, deduped
        ERRORS errors           list
    """

    name: str = "retrieval"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        sub_questions: list[str] = state.get("sub_questions") or []
        doc_scope:     list[str] = state.get("document_scope") or []
        skip_retrieval: bool     = state.get("skip_retrieval") or False

        # ── skip_retrieval path (adaptive_router said corpus fits in context) ─
        if skip_retrieval:
            if not doc_scope:
                logger.warning("skip_retrieval=True but document_scope is empty")
                state["retrieved_chunks"] = []
                return state
            chunks = PGStore.get().fetch_chunks_for_scope(doc_scope)
            logger.info(
                "RetrievalTool skip_retrieval: loaded %d chunks for scope %s",
                len(chunks), doc_scope,
            )
            state["retrieved_chunks"] = chunks
            return state

        if not sub_questions:
            logger.warning("RetrievalTool: no sub_questions in state — skipping")
            state["retrieved_chunks"] = []
            return state

        retrieval_cfg = config.get("query", {}).get("retrieval", {})

        # Auto-populate industry filter from Postgres when scope = 1 doc
        base_filters: dict = dict(retrieval_cfg.get("filters") or {})
        if doc_scope:
            base_filters["document_id"] = doc_scope
            if len(doc_scope) == 1 and "industry" not in base_filters:
                industry = PGStore.get().get_document_industry(doc_scope[0])
                if industry:
                    base_filters["industry"] = industry

        all_chunks: list[Chunk] = []
        seen_ids:   set[str]   = set()

        for query in sub_questions:
            try:
                result = _retrieve_one(
                    query=query,
                    retrieval_cfg=retrieval_cfg,
                    full_config=config,
                    filters=base_filters,
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
                errors.append({"tool": "retrieval", "query": query, "error": str(exc)})
                state["errors"] = errors

        state["retrieved_chunks"] = all_chunks
        return state


# ── dispatcher ────────────────────────────────────────────────────────────────

def _retrieve_one(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    filters: Optional[dict] = None,
) -> dict:
    method = retrieval_cfg.get("method", "hybrid_rerank")
    start  = time.perf_counter()

    if method == "naive":
        chunks = _naive(query, retrieval_cfg, full_config, filters)
    elif method == "hybrid":
        chunks = _hybrid(query, retrieval_cfg, full_config, filters)
    elif method == "hybrid_rerank":
        chunks = _hybrid_rerank(query, retrieval_cfg, full_config, filters)
    elif method == "hyde":
        chunks = _hyde(query, retrieval_cfg, full_config, filters)
    elif method == "enriched":
        chunks = _hybrid_rerank(query, retrieval_cfg, full_config, filters)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Valid: naive | hybrid | hybrid_rerank | hyde | enriched"
        )

    return {
        "chunks":     chunks,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "method":     method,
    }


# ── methods ───────────────────────────────────────────────────────────────────

def _naive(query, cfg, full_config, filters):
    embedder = get_dense_model(full_config)
    q_emb    = embedder.encode(query, normalize_embeddings=True).tolist()
    return VectorStore.get().search(q_emb, top_k=cfg.get("top_k", 5), filters=filters)


def _hybrid(query, cfg, full_config, filters):
    top_k         = cfg.get("top_k", 5)
    dense_weight  = cfg.get("dense_weight", 0.6)
    sparse_weight = cfg.get("sparse_weight", 0.4)

    embedder    = get_dense_model(full_config)
    q_emb       = embedder.encode(query, normalize_embeddings=True).tolist()
    dense_hits  = VectorStore.get().search(q_emb, top_k=top_k * 2, filters=filters)
    sparse_hits = KeywordIndex.get().search(query, top_k=top_k * 2, filters=filters)

    return _rrf_fuse(dense_hits, sparse_hits,
                     w_dense=dense_weight, w_sparse=sparse_weight, top_k=top_k)


def _hybrid_rerank(query, cfg, full_config, filters):
    top_k       = cfg.get("top_k", 5)
    candidate_k = cfg.get("candidate_k", top_k * 4)
    candidates  = _hybrid(query, {**cfg, "top_k": candidate_k}, full_config, filters)

    reranker = get_reranker(full_config)
    scores   = reranker.predict([(query, c.get("text") or "") for c in candidates])
    ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


def _hyde(query, cfg, full_config, filters):
    top_k = cfg.get("top_k", 5)
    llm   = get_llm(full_config)
    hyp   = llm.invoke(
        "Write a short factual paragraph directly answering this question. "
        "Reply with ONLY the paragraph.\n\nQuestion: " + query
    ).content
    embedder = get_dense_model(full_config)
    hyp_emb  = embedder.encode(hyp, normalize_embeddings=True).tolist()
    return VectorStore.get().search(hyp_emb, top_k=top_k, filters=filters)


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(dense_hits, sparse_hits, w_dense=0.6, w_sparse=0.4, top_k=5, k=60):
    scores:    dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    for rank, chunk in enumerate(dense_hits, 1):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + w_dense * (1.0 / (k + rank))
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(sparse_hits, 1):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + w_sparse * (1.0 / (k + rank))
        chunk_map[cid] = chunk
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in ranked[:top_k]]