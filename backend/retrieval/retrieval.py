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
from backend.core.models import get_dense_query_prefix, get_dense_model, get_reranker
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
        # HARD filter = explicit document_id scope (a choice the user/agent made — respect it).
        # SOFT filter = doc_type / industry (a hint the agent inferred). Both are ANDed by the
        # store, but the soft filter must never HIDE the answer: if it yields zero chunks we
        # retry the sub-question without it (keeping the hard scope). Empty -> whole corpus.
        hard_filters: dict = {"document_id": doc_scope} if doc_scope else {}
        soft_filters: dict = {}
        if state.get("doc_type"):
            soft_filters["doc_type"] = state["doc_type"]
        if state.get("industry"):
            soft_filters["industry"] = state["industry"]

        all_chunks: list[Chunk] = []
        seen_ids:   set[str]   = set()

        for query in sub_questions:
            try:
                result = _retrieve_one(
                    query=query,
                    retrieval_cfg=retrieval_cfg,
                    full_config=config,
                    filters={**hard_filters, **soft_filters} or None,
                )
                if soft_filters and not result["chunks"]:
                    logger.info(
                        "RetrievalTool: soft filter %s returned 0 for %r — retrying without it",
                        soft_filters, query[:50],
                    )
                    result = _retrieve_one(
                        query=query, retrieval_cfg=retrieval_cfg, full_config=config,
                        filters=(hard_filters or None),
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

def _embed_query(embedder, query: str, full_config: dict) -> list[float]:
    # The query prefix is model-specific and config-driven (empty for bge-m3;
    # "search_query: " for nomic). It MUST differ from the document prefix used at
    # index time for instruction-tuned models, or dense recall silently degrades.
    prefix = get_dense_query_prefix(full_config)
    return embedder.encode(
        prefix + query, normalize_embeddings=True
    ).tolist()


def _naive(query, cfg, full_config, filters):
    top_k    = cfg["top_n"]
    embedder = get_dense_model(full_config)
    q_emb    = _embed_query(embedder, query, full_config)
    return VectorStore.search(q_emb, full_config, top_k=top_k, filters=filters)


def _hybrid(query, cfg, full_config, filters, fuse_top_k=None):
    # fuse_top_k lets the reranker path request the FULL candidate pool (candidate_k)
    # instead of the answer-facing top_n. Default (None) = top_n for direct hybrid use.
    top_k         = fuse_top_k if fuse_top_k is not None else cfg["top_n"]
    dense_weight  = cfg["dense_weight"]
    sparse_weight = cfg["sparse_weight"]
    candidate_k   = cfg["candidate_k"]

    embedder    = get_dense_model(full_config)
    q_emb       = _embed_query(embedder, query, full_config)
    dense_hits  = VectorStore.search(q_emb, full_config, top_k=candidate_k, filters=filters)
    sparse_hits = KeywordIndex.search(query, full_config, top_k=candidate_k, filters=filters)

    return _rrf_fuse(dense_hits, sparse_hits,
                     w_dense=dense_weight, w_sparse=sparse_weight,
                     top_k=top_k, k=candidate_k)


def _hybrid_rerank(query, cfg, full_config, filters):
    candidate_k  = cfg["candidate_k"]
    rerank_top_k = cfg["rerank_top_k"]
    # Fuse the FULL candidate_k pool (not top_n) so the reranker actually sees the
    # wide pool it was configured for. Previously _hybrid returned top_n=20 and the
    # [:candidate_k] slice was a no-op, so a right-doc chunk ranked 21-80 by RRF was
    # dropped BEFORE the cross-encoder could rescore it — the many-docs failure mode.
    candidates   = _hybrid(query, cfg, full_config, filters, fuse_top_k=candidate_k)
    if not candidates:
        return []

    try:
        reranker = get_reranker(full_config)
        scores   = reranker.predict([(query, c["text"] or "") for c in candidates])
        ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # Optional relevance gate: drop candidates below a calibrated cross-encoder score
        # so an all-wrong-doc result returns nothing (answerer then refuses) instead of
        # forcing 5 irrelevant chunks in. Unset by default — bge-reranker scores are
        # logits (can be negative); calibrate on real queries before enabling.
        min_score = cfg.get("rerank_min_score")
        if min_score is not None:
            ranked = [(s, c) for s, c in ranked if s >= min_score]

        # Stamp the reranker score so it overrides the RRF _score from _hybrid.
        result = []
        for score, chunk in ranked[:rerank_top_k]:
            c = dict(chunk)
            c["_score"] = float(score)
            result.append(c)
        return result
    except Exception as exc:
        logger.warning(
            "Reranker failed (rate limited or API error), falling back to standard hybrid RRF results: %s",
            exc,
        )
        fallback_limit = cfg.get("top_n", 20)
        return candidates[:fallback_limit]



def _hyde(query, cfg, full_config, filters):
    top_k    = cfg["top_n"]
    llm      = get_llm(full_config)
    response = llm.invoke(
        "Write a short factual paragraph directly answering this question. "
        "Reply with ONLY the paragraph.\n\nQuestion: " + query
    )
    usage.record_from_message("hyde", response, model=full_config["llm"]["model"], provider=full_config["llm"]["provider"],)
    hyp      = clean_message_content(response.content)
    embedder = get_dense_model(full_config)
    hyp_emb  = _embed_query(embedder, hyp, full_config)
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
    # Stamp the RRF fusion score onto each chunk as _score.
    result = []
    for cid in ranked[:top_k]:
        c = dict(chunk_map[cid])
        c["_score"] = scores[cid]
        result.append(c)
    return result