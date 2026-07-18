"""
backend/retrieval/retrieval.py
──────────────────────────────
RetrievalTool — implements the Tool Protocol from backend/core/tool.py.

  run(state, config)
    READS  state["sub_questions"]    list[str]   — decomposed sub-questions
           state["document_scope"]   list[str]   — doc_ids to restrict search
    WRITES state["retrieved_chunks"] list[Chunk] — flat, deduped, best-first
    ERRORS state["errors"]           list        — append only, never raise

Five methods (config["query"]["retrieval"]["method"]):
  naive | hybrid | hybrid_rerank | hyde | enriched
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from backend.core.tool import PipelineState
from backend.core.schemas import Chunk
from backend.core.models import DENSE_QUERY_PREFIX, get_dense_model, get_reranker
from backend.core import usage
from backend.core.llm_client import get_llm, clean_message_content
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.keyword_index import KeywordIndex

logger = logging.getLogger(__name__)


class RetrievalTool:
    """
    Implements the Tool Protocol (backend/core/tool.py).

    State contract:
        READS  sub_questions    list[str]   ← decomposed sub-questions
               document_scope  list[str]   ← doc_id filter, empty = all docs
        WRITES retrieved_chunks list[Chunk]
        ERRORS errors           list
    """

    name: str = "retrieval"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        sub_questions: list[str] = state["sub_questions"] or []
        doc_scope:     list[str] = state["document_scope"] or []

        if not sub_questions:
            logger.warning("RetrievalTool: no sub_questions in state — skipping")
            state["retrieved_chunks"] = []
            return state

        retrieval_cfg = config["query"]["retrieval"]
        filters       = {"document_id": doc_scope} if doc_scope else None

        all_chunks: list[Chunk] = []
        seen_ids:   set[str]   = set()

        for query in sub_questions:
            try:
                result = _retrieve_one(
                    query=query,
                    retrieval_cfg=retrieval_cfg,
                    full_config=config,
                    filters=filters,
                )
                logger.debug(
                    "RetrievalTool q=%r method=%s n=%d %.1fms",
                    query[:60], result["method"],
                    len(result["chunks"]), result["latency_ms"],
                )
                for chunk in result["chunks"]:
                    cid = chunk["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)

            except Exception as exc:
                logger.error("RetrievalTool failed for %r: %s", query[:60], exc)
                errors: list = state["errors"] or []
                errors.append({"tool": "retrieval", "query": query, "error": str(exc)})
                state["errors"] = errors

        state["retrieved_chunks"] = all_chunks
        return state


# ── dispatcher ────────────────────────────────────────────────────────────────

def _retrieve_one(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    filters: Optional[dict],
) -> dict:
    method = retrieval_cfg["method"]
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

def _embed_query(embedder, query: str) -> list[float]:
    # nomic queries MUST carry the query prefix (different from the document
    # prefix used at index time) — otherwise dense recall silently degrades.
    return embedder.encode(
        DENSE_QUERY_PREFIX + query, normalize_embeddings=True
    ).tolist()


def _naive(query, cfg, full_config, filters):
    top_k    = cfg["top_n"]
    embedder = get_dense_model(full_config)
    q_emb    = _embed_query(embedder, query)
    return VectorStore.search(q_emb, full_config, top_k=top_k, filters=filters)


def _hybrid(query, cfg, full_config, filters):
    top_k         = cfg["top_n"]
    dense_weight  = cfg["dense_weight"]
    sparse_weight = cfg["sparse_weight"]
    candidate_k   = cfg["candidate_k"]

    embedder    = get_dense_model(full_config)
    q_emb       = _embed_query(embedder, query)
    dense_hits  = VectorStore.search(q_emb, full_config, top_k=candidate_k, filters=filters)
    sparse_hits = KeywordIndex.search(query, full_config, top_k=candidate_k, filters=filters)

    return _rrf_fuse(dense_hits, sparse_hits,
                     w_dense=dense_weight, w_sparse=sparse_weight,
                     top_k=top_k, k=candidate_k)


def _hybrid_rerank(query, cfg, full_config, filters):
    candidate_k  = cfg["candidate_k"]
    rerank_top_k = cfg["rerank_top_k"]
    candidates   = _hybrid(query, cfg, full_config, filters)[:candidate_k]
    if not candidates:
        return []

    reranker = get_reranker(full_config)
    scores   = reranker.predict([(query, c["text"] or "") for c in candidates])
    ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:rerank_top_k]]


def _hyde(query, cfg, full_config, filters):
    top_k    = cfg["top_n"]
    llm      = get_llm(full_config)
    response = llm.invoke(
        "Write a short factual paragraph directly answering this question. "
        "Reply with ONLY the paragraph.\n\nQuestion: " + query
    )
    usage.record_from_message("hyde", response)
    hyp      = clean_message_content(response.content)
    embedder = get_dense_model(full_config)
    hyp_emb  = _embed_query(embedder, hyp)
    return VectorStore.search(hyp_emb, full_config, top_k=top_k, filters=filters)


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(dense_hits, sparse_hits, w_dense, w_sparse, top_k, k):
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