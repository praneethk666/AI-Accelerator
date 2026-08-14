"""
backend/retrieval/retrieval.py
──────────────────────────────
RetrievalTool — implements the Tool Protocol from backend/core/tool.py.

  run(state, config)
    READS  state["sub_questions"]    list[str]   — decomposed sub-questions
           state["document_scope"]   list[str]   — doc_ids to restrict search
           state["user_roles"]       list[str]   — JWT roles for RBAC filtering
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
               user_roles      list[str]   ← JWT roles for RBAC post-filter
        WRITES retrieved_chunks list[Chunk]
        ERRORS errors           list
    """

    name: str = "retrieval"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        sub_questions: list[str] = state.get("sub_questions") or []
        doc_scope:     list[str] = state.get("document_scope") or []
        # RBAC: user roles from JWT, set by the API layer before calling the pipeline.
        user_roles:    list[str] = state.get("user_roles") or []

        if not sub_questions:
            raw_q = state.get("standalone_query") or state.get("query")
            if raw_q:
                sub_questions = [raw_q]
            else:
                logger.warning("RetrievalTool: no sub_questions or query in state — skipping")
                state["retrieved_chunks"] = []
                return state

        # Safety net: always search the raw, unmodified user query too, so a
        # query_planner mis-rewrite (e.g. acronym hallucination) can never fully
        # starve retrieval of the literal terms the user actually typed.
        # Case-insensitive dedup avoids a redundant search when the planner's
        # rewrite already matches the raw query modulo case. Bounded to a sane
        # max length so a pathological/garbage query can't blow up the fan-out.
        raw_query = (state.get("query") or "").strip()
        _MAX_RAW_QUERY_CHARS = 500
        if raw_query and len(raw_query) <= _MAX_RAW_QUERY_CHARS:
            existing_lower = {q.strip().lower() for q in sub_questions}
            if raw_query.lower() not in existing_lower:
                sub_questions = sub_questions + [raw_query]
                logger.info(
                    "RetrievalTool: added raw query as safety-net search variant "
                    "(planner sub_questions did not include it verbatim): %r",
                    raw_query[:80],
                )

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
        _t0_total = time.perf_counter()

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

                new_chunk_ids = []
                for chunk in result["chunks"]:
                    cid = chunk["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)
                        new_chunk_ids.append(str(cid))

                logger.info(
                    "RetrievalTool q=%r method=%s n=%d new_ids=%s %.1fms",
                    query[:60], result["method"],
                    len(result["chunks"]), new_chunk_ids, result["latency_ms"],
                )

            except Exception as exc:
                logger.error("RetrievalTool failed for %r: %s", query[:60], exc)
                errors: list = state["errors"] or []
                errors.append({"tool": "retrieval", "query": query, "error": str(exc)})
                state["errors"] = errors

        # ── RBAC post-filter ─────────────────────────────────────────────────────────
        # Drop chunks whose allowed_roles list is set and doesn't intersect with
        # user_roles. A chunk with allowed_roles=None (or []) is PUBLIC — all users.
        if user_roles:
            user_roles_set = set(r.lower() for r in user_roles)
            pre_acl = len(all_chunks)
            all_chunks = [
                c for c in all_chunks
                if _rbac_allowed(c, user_roles_set)
            ]
            dropped = pre_acl - len(all_chunks)
            if dropped:
                logger.info("RetrievalTool RBAC: dropped %d chunks (user roles: %s)", dropped, user_roles)
        else:
            # No user roles in state — pass-through (unauthenticated / internal call).
            # ACL-restricted chunks are NOT dropped here; deploy with an auth middleware
            # that always injects user_roles before any real user request reaches this.
            pass

        # ── OTel span enrichment ─────────────────────────────────────────────────────
        try:
            from opentelemetry import trace as _otel_trace
            span = _otel_trace.get_current_span()
            if span and span.is_recording():
                chunk_ids_str = ",".join(str(c["chunk_id"]) for c in all_chunks[:50])
                span.set_attribute("retrieval.chunk_ids", chunk_ids_str)
                span.set_attribute("retrieval.chunk_count", len(all_chunks))
                # Stamp the index_version from the first chunk that carries it
                for c in all_chunks:
                    iv = (c.get("tags") or {}).get("index_version")
                    if iv:
                        span.set_attribute("retrieval.index_version", iv)
                        break
        except Exception:
            pass  # OTel is optional; never let it break retrieval

        # ── Fallback: Context Expansion (page-level) ───────────────────────────
        # If all retrieved chunks score below `fallback_threshold`, it means the
        # query terms exist in the document but are spread across different chunks
        # (e.g. "WORKHEAD" in one chunk, spare-parts table in another). Instead of
        # sending the LLM low-confidence fragments, we expand each partial hit to
        # its full page content from document_blocks and inject those as context.
        fb_cfg = retrieval_cfg
        if fb_cfg.get("fallback_enabled", False) and all_chunks:
            best_score = max(
                (float(c.get("_score") or 0) for c in all_chunks),
                default=0.0,
            )
            fb_threshold = float(fb_cfg.get("fallback_threshold", -1.0))
            if best_score < fb_threshold:
                logger.info(
                    "RetrievalTool: best score %.3f < threshold %.3f — triggering page-expansion fallback",
                    best_score, fb_threshold,
                )
                fb_top_k = int(fb_cfg.get("fallback_top_k", 2))
                expanded = _expand_chunks_to_pages(all_chunks[:fb_top_k])
                if expanded:
                    # Prepend expanded page-chunks; keep original chunks after for
                    # any non-expanded context that still contributes.
                    expanded_ids = {c["chunk_id"] for c in expanded}
                    remaining = [c for c in all_chunks if c["chunk_id"] not in expanded_ids]
                    all_chunks = expanded + remaining

        # Checkpoint 4: Token Budget Manager (greedy context window selection)
        tb_cfg = config.get("guardrails", {}).get("token_budget", {})
        if tb_cfg.get("enabled", True):
            from backend.guardrails.token_budget import TokenBudgetManager
            tb_mgr = TokenBudgetManager.from_config(config)
            selected_chunks, total_tokens, dropped_count = tb_mgr.select_chunks(
                all_chunks, budget_tokens=tb_cfg.get("max_context_tokens", 8000)
            )
            state["retrieved_chunks"] = selected_chunks
        else:
            state["retrieved_chunks"] = all_chunks

        # ── Persist audit record (best-effort, never block retrieval) ───────────
        try:
            final_chunks  = state["retrieved_chunks"]
            total_ms      = round((time.perf_counter() - _t0_total) * 1000, 1)
            raw_query_str = (state.get("query") or "").strip()
            # Infer index_version from the first chunk that carries it in tags
            _index_ver = None
            for _c in final_chunks:
                _iv = (_c.get("tags") or {}).get("index_version")
                if _iv:
                    _index_ver = _iv
                    break
            from backend.storage.postgres_store import PostgresStore as _PGS
            _pg = _PGS()
            try:
                _pg.write_query_audit(
                    session_id=state.get("session_id"),
                    query_text=raw_query_str or "(unknown)",
                    retrieved_chunk_ids=[str(c["chunk_id"]) for c in final_chunks],
                    latency_ms=total_ms,
                    index_version=_index_ver,
                    user_roles=user_roles or None,
                )
            finally:
                _pg.close()
        except Exception:
            logger.debug("RetrievalTool: query_audit write failed (non-fatal)", exc_info=True)

        return state



# ── RBAC helper ───────────────────────────────────────────────────────────────

def _rbac_allowed(chunk: dict, user_roles_set: set[str]) -> bool:
    """Return True if the user is allowed to see this chunk.

    A chunk is PUBLIC (allowed_roles is None or []) — any user can see it.
    A chunk is RESTRICTED if allowed_roles is a non-empty list; only users
    whose roles intersect with allowed_roles may see it.

    Roles are stored in both chunk["allowed_roles"] (top-level) and
    chunk["tags"]["allowed_roles"] (for Qdrant payload filtering later).
    We check both, preferring the top-level field.
    """
    roles = chunk.get("allowed_roles") or (chunk.get("tags") or {}).get("allowed_roles")
    if not roles:
        return True   # public chunk
    chunk_roles = set(r.lower() for r in roles)
    return bool(user_roles_set & chunk_roles)


# ── dispatcher ────────────────────────────────────────────────────────────────

def _retrieve_one(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    filters: Optional[dict],
) -> dict:
    method = retrieval_cfg["method"]
    start  = time.perf_counter()

    try:
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
    except Exception as exc:
        logger.warning(
            "Retrieval method %s failed (Qdrant down?): %s — falling back to keyword search",
            method, exc
        )
        try:
            chunks = KeywordIndex.search(query, full_config, top_k=retrieval_cfg.get("top_n", 20), filters=filters)
            method = "keyword_fallback"
        except Exception as k_exc:
            logger.critical("Keyword index fallback also failed: %s", k_exc)
            raise exc

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

    # Enforce degradation guardrails: reranker_max_pairs and reranker_max_tokens_per_pair
    g_cfg = full_config.get("guardrails", {})
    deg_cfg = g_cfg.get("degradation", {})
    
    max_pairs = deg_cfg.get("reranker_max_pairs")
    if max_pairs is not None:
        candidates = candidates[:max_pairs]
        
    max_tokens = deg_cfg.get("reranker_max_tokens_per_pair")
    pairs = []
    for c in candidates:
        text = c["text"] or ""
        if max_tokens is not None:
            # 1 token is roughly 4 characters
            char_limit = max_tokens * 4
            if len(text) > char_limit:
                text = text[:char_limit]
        pairs.append((query, text))

    try:
        reranker = get_reranker(full_config)
        scores   = reranker.predict(pairs)

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
    except Exception:
        logger.info("[Reranker] Rate limited or API key error — using standard hybrid RRF reranker.")
        fallback_limit = cfg.get("top_n", 20)
        return candidates[:fallback_limit]



def _hyde(query, cfg, full_config, filters):
    top_k    = cfg["top_n"]
    llm      = get_llm(full_config)
    hyde_prompt = (
        "Write a short factual paragraph directly answering this question. "
        "Reply with ONLY the paragraph.\n\nQuestion: " + query
    )
    response = llm.invoke(hyde_prompt)
    usage.record_from_message("hyde", response, prompt=hyde_prompt, model=full_config["llm"]["model"], provider=full_config["llm"]["provider"])
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


# ── Fallback helper ───────────────────────────────────────────────────────────

def _expand_chunks_to_pages(chunks: list[Chunk]) -> list[Chunk]:
    """Page-expansion fallback: replace each chunk with the full text of its page.

    Uses the same document_blocks store as answerer._expand_thin_chunks and
    get_page_context.GetPageContextTool — reading order raw extraction, not
    chunked text.  Returns a new list of synthetic page-chunks that inherit
    all metadata (document_id, source_ref, chunk_id) from the seed chunk but
    have their `text` replaced with the full reconstructed page text.
    Skips chunks with no source_ref page/slide; skips Excel sheets to avoid
    token bloat (same guard as answerer._expand_thin_chunks).
    """
    from backend.storage.postgres_store import PostgresStore

    # Deduplicate by (document_id, page) so two chunks on the same page
    # only trigger one DB fetch.
    seen_pages: set[tuple] = set()
    to_expand: list[tuple[tuple, Chunk]] = []
    for chunk in chunks:
        ref = chunk.get("source_ref") or {}
        # Skip Excel sheets (token bloat risk)
        filename = ref.get("filename") or ""
        if filename.lower().endswith((".xlsx", ".xls", ".xlsm")) or ref.get("sheet") is not None:
            continue
        doc_id = chunk.get("document_id")
        page_val = ref.get("page") or ref.get("slide")
        if not doc_id or page_val is None:
            continue
        key = (str(doc_id), page_val)
        if key not in seen_pages:
            seen_pages.add(key)
            to_expand.append((key, chunk))

    if not to_expand:
        return []

    store = None
    result: list[Chunk] = []
    try:
        store = PostgresStore()
        for (doc_id, page_val), seed_chunk in to_expand:
            try:
                blocks = store.get_blocks(doc_id)
            except Exception:
                logger.exception(
                    "_expand_chunks_to_pages: get_blocks failed (doc %s, page %s)",
                    doc_id, page_val,
                )
                continue
            page_blocks = [
                b for b in blocks
                if isinstance(b.get("source_ref"), dict) and (
                    b["source_ref"].get("page") == page_val or
                    b["source_ref"].get("slide") == page_val
                )
            ]
            # Build text from both prose and table_data rows (same as answerer)
            parts = []
            for b in page_blocks:
                t = (b.get("text") or "").strip()
                if t:
                    parts.append(t)
                td = b.get("table_data")
                if td and td.get("rows"):
                    headers = td.get("headers") or []
                    rows_str = "\n".join(
                        " | ".join(str(v) for v in row) for row in td["rows"]
                    )
                    parts.append(
                        f"Columns: {' | '.join(str(h) for h in headers)}\n{rows_str}"
                    )
            if not parts:
                continue
            # Build synthetic expanded chunk from seed metadata
            expanded = dict(seed_chunk)
            expanded["text"] = "\n\n".join(parts)
            logger.info(
                "_expand_chunks_to_pages: expanded doc=%s page=%s (%d chars)",
                doc_id, page_val, len(expanded["text"]),
            )
            result.append(expanded)
    finally:
        if store is not None:
            store.close()

    return result