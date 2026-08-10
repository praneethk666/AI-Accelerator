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

from backend.core import usage
from backend.core.llm_client import clean_message_content, get_llm
from backend.core.models import get_dense_model, get_dense_query_prefix, get_reranker
from backend.core.schemas import Chunk
from backend.core.tool import PipelineState
from backend.retrieval.keyword_index import KeywordIndex
from backend.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _harvest_subquery_worker(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    hard_filters: dict,
    soft_filters: dict,
    is_rerank_method: bool,
) -> dict:
    """Worker task executed in ThreadPoolExecutor for a single sub-question.

    Operates strictly on local variables and returns harvested chunks and optional error dict
    to the main thread to ensure thread safety.
    """
    start = time.perf_counter()
    try:
        if is_rerank_method:
            candidate_k = retrieval_cfg.get("candidate_k", 80)
            chunks = _hybrid(
                query, retrieval_cfg, full_config,
                filters={**hard_filters, **soft_filters} or None,
                fuse_top_k=candidate_k,
            )
            if soft_filters and not chunks:
                logger.info(
                    "RetrievalTool worker: soft filter %s returned 0 for %r — retrying without it",
                    soft_filters, query[:50],
                )
                chunks = _hybrid(
                    query, retrieval_cfg, full_config,
                    filters=(hard_filters or None),
                    fuse_top_k=candidate_k,
                )
        else:
            result = _retrieve_one(
                query=query,
                retrieval_cfg=retrieval_cfg,
                full_config=full_config,
                filters={**hard_filters, **soft_filters} or None,
            )
            if soft_filters and not result["chunks"]:
                logger.info(
                    "RetrievalTool worker: soft filter %s returned 0 for %r — retrying without it",
                    soft_filters, query[:50],
                )
                result = _retrieve_one(
                    query=query, retrieval_cfg=retrieval_cfg, full_config=full_config,
                    filters=(hard_filters or None),
                )
            chunks = result["chunks"]

        # Filter out Excel chunks from standard Vector/BM25 results if doing a global search
        if not hard_filters.get("document_id"):
            filtered_chunks = []
            for c in chunks:
                sr = c.get("source_ref")
                filename = ""
                if sr:
                    if hasattr(sr, "get"):
                        filename = sr.get("filename", "")
                    elif hasattr(sr, "filename"):
                        filename = getattr(sr, "filename", "")
                if not (isinstance(filename, str) and filename.lower().endswith((".xlsx", ".xls", ".csv"))):
                    filtered_chunks.append(c)
            chunks = filtered_chunks

        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.debug(
            "RetrievalTool worker harvested q=%r n=%d %.1fms",
            query[:60], len(chunks), latency,
        )
        return {
            "query": query,
            "chunks": chunks,
            "latency_ms": latency,
            "error": None,
        }
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.error("RetrievalTool worker failed for %r after %.1fms: %s", query[:60], latency, exc)
        return {
            "query": query,
            "chunks": [],
            "latency_ms": latency,
            "error": {"tool": "retrieval", "query": query, "error": str(exc)},
        }


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
        sub_questions: list[str] = state.get("sub_questions") or []
        doc_scope:     list[str] = state.get("document_scope") or []

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
        raw_query = (state.get("raw_user_prompt") or state.get("query") or "").strip()
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

        retrieval_cfg = dict(config["query"]["retrieval"])
        
        # Apply doc_type specific overrides from global.yaml (e.g. spreadsheet -> hybrid_local_rerank)
        doc_type = state.get("doc_type")
        overrides = retrieval_cfg.get("doc_type_overrides", {})
        if doc_type and doc_type in overrides:
            retrieval_cfg["method"] = overrides[doc_type]
            logger.info("RetrievalTool: overriding method to %s for doc_type %s", overrides[doc_type], doc_type)

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

        from concurrent.futures import ThreadPoolExecutor, as_completed

        method = retrieval_cfg.get("method", "hybrid_rerank")
        is_rerank_method = method in ("hybrid_rerank", "enriched")

        max_workers = max(1, min(len(sub_questions), int(retrieval_cfg.get("max_workers", 4))))
        results_by_index: list[dict | None] = [None] * len(sub_questions)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _harvest_subquery_worker,
                    query=query,
                    retrieval_cfg=retrieval_cfg,
                    full_config=config,
                    hard_filters=hard_filters,
                    soft_filters=soft_filters,
                    is_rerank_method=is_rerank_method,
                ): idx
                for idx, query in enumerate(sub_questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_index[idx] = future.result()
                except Exception as exc:
                    query = sub_questions[idx]
                    logger.error("Unhandled harvest worker failure for %r: %s", query[:60], exc)
                    results_by_index[idx] = {
                        "query": query,
                        "chunks": [],
                        "latency_ms": 0.0,
                        "error": {"tool": "retrieval", "query": query, "error": str(exc)},
                    }

        # ── Main-thread safe state merging (zero race conditions) ──
        all_chunks: list[Chunk] = []
        seen_ids:   set[str]   = set()
        errors_to_append: list[dict] = []

        for res in results_by_index:
            if not res:
                continue
            if res.get("error"):
                errors_to_append.append(res["error"])

            for chunk in res.get("chunks") or []:
                cid = chunk["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)

        if errors_to_append:
            existing_errors: list = state.get("errors") or []
            existing_errors.extend(errors_to_append)
            state["errors"] = existing_errors

        # ── Batch Reranking (executed ONCE after all sub-questions are harvested) ──
        if is_rerank_method and all_chunks:
            standalone_query = (
                state.get("standalone_query")
                or state.get("query")
                or state.get("raw_user_prompt")
                or (sub_questions[0] if sub_questions else "")
            )
            all_chunks = _rerank_candidates(standalone_query, all_chunks, retrieval_cfg, config)

        # ── Excel Keyword Injection ────────────────────────────────────────────
        # Vector/BM25 search often misses Excel rows because part numbers and
        # supplier names are not in the enrichment keywords. Do a direct Postgres
        # ILIKE scan on all Excel chunks for any significant word in the query
        # and prepend matching chunks so the LLM always sees them.
        raw_q_for_excel = state.get("standalone_query") or state.get("query") or ""
        excel_chunks = _excel_keyword_inject(raw_q_for_excel, doc_scope, config)
        if excel_chunks:
            logger.info(
                "RetrievalTool: Excel keyword injection found %d matching chunks for query %r",
                len(excel_chunks), raw_q_for_excel[:80],
            )
            for chunk in excel_chunks:
                cid = chunk["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.insert(0, chunk)  # prepend so LLM sees them first

        # ── Fallback: Context Expansion (page-level) ──────────────────────────
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
            selected_chunks, _total_tokens, _dropped_count = tb_mgr.select_chunks(
                all_chunks, budget_tokens=tb_cfg.get("max_context_tokens", 8000)
            )
            state["retrieved_chunks"] = selected_chunks
        else:
            state["retrieved_chunks"] = all_chunks
        return state



# ── Excel Keyword Injection helper ────────────────────────────────────────────

_EXCEL_STOP_WORDS = {
    "what", "which", "where", "when", "who", "how", "are", "is", "the",
    "a", "an", "of", "for", "in", "by", "to", "do", "and", "or", "not",
    "all", "any", "give", "show", "list", "tell", "me", "about", "from",
    "that", "this", "with", "has", "have", "be", "been", "was", "were",
    "parts", "part", "items", "item", "name", "names", "provided", "supply",
    "supplied", "make", "made", "number", "numbers", "components", "component",
}


def _excel_keyword_inject(
    query: str,
    doc_scope: list[str],
    config: dict,
    max_chunks: int = 200,
) -> list[dict]:
    """Direct PostgreSQL ILIKE scan on Excel chunks for any significant word in query.

    Extracts words >= 3 chars (excluding stop-words) from the query and fetches
    all Excel (.xlsx/.xls/.csv) chunks whose text matches ANY of those words.
    Returns them as Chunk-compatible dicts to be merged into retrieved_chunks.
    Silently returns [] on any error so it never breaks the normal pipeline.
    """
    import re
    # Normalize double/triple hyphens to single hyphens
    normalized_query = re.sub(r"-+", "-", query)
    
    words = re.findall(r"[A-Za-z0-9][\w\-\.]*", normalized_query)
    
    # Extract base alphanumeric tokens for hyphenated codes (e.g. MC000954 from KE-MC000954-G)
    expanded_words = []
    for w in words:
        expanded_words.append(w)
        if '-' in w:
            parts = w.split('-')
            # Add significant sub-parts (e.g. MC000954)
            expanded_words.extend([p for p in parts if len(p) >= 3])

    keywords = []
    for w in expanded_words:
        if len(w) >= 3 and w.lower() not in _EXCEL_STOP_WORDS and w not in keywords:
            keywords.append(w)

    if not keywords:
        return []

    try:
        import psycopg

        from backend.core.config import get_db_url

        dsn = get_db_url(config) if config else None
        if not dsn:
            import os
            dsn = os.getenv("POSTGRES_URL", "")
        if not dsn:
            return []

        # Build OR conditions checking both c.text AND c.table_data
        conditions = " OR ".join(["(c.text ILIKE %s OR c.table_data::text ILIKE %s)"] * len(keywords))
        params: list = []
        for kw in keywords:
            params.extend([f"%{kw}%", f"%{kw}%"])

        # Optionally scope to explicit document_ids
        scope_clause = ""
        if doc_scope:
            placeholders = ", ".join(["%s"] * len(doc_scope))
            scope_clause = f"AND c.document_id::text IN ({placeholders})"
            params.extend(doc_scope)

        sql = f"""
            SELECT
                c.chunk_id::text,
                c.document_id::text,
                c.text,
                c.token_count,
                c.tags,
                c.source_ref,
                c.table_data,
                c.image_path
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.filename ~* '\\.(xlsx|xls|csv)$'
            AND ({conditions})
            {scope_clause}
            LIMIT %s
        """
        params.append(max_chunks)

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        chunks = []
        for row in rows:
            chunks.append({
                "chunk_id":    row[0],
                "document_id": row[1],
                "text":        row[2] or "",
                "token_count": row[3],
                "tags":        row[4] or {},
                "source_ref":  row[5] or {},
                "table_data":  row[6],
                "image_path":  row[7],
                "_score":      0.0,   # no vector score — injected, not ranked
                "_excel_injected": True,
            })
        return chunks

    except Exception as exc:
        logger.warning("_excel_keyword_inject failed (non-fatal): %s", exc)
        return []


# ── dispatcher ────────────────────────────────────────────────────────────────

def _retrieve_one(
    query: str,
    retrieval_cfg: dict,
    full_config: dict,
    filters: dict | None,
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
        elif method == "hybrid_local_rerank":
            from backend.retrieval.hybrid_search import _hybrid_local_rerank
            chunks = _hybrid_local_rerank(query, retrieval_cfg, full_config, filters)
        elif method == "hyde":
            chunks = _hyde(query, retrieval_cfg, full_config, filters)
        elif method == "enriched":
            chunks = _hybrid_rerank(query, retrieval_cfg, full_config, filters)
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                "Valid: naive | hybrid | hybrid_rerank | hybrid_local_rerank | hyde | enriched"
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


def _rerank_candidates(query: str, candidates: list[dict], cfg: dict, full_config: dict) -> list[dict]:
    """Batch rerank candidate chunks against `query` using the configured reranker.

    Enforces degradation limits (max pairs, max tokens) and falls back to RRF candidate
    order if the reranker fails.
    """
    if not candidates:
        return []

    rerank_top_k = cfg.get("rerank_top_k", cfg.get("top_n", 5))

    # Enforce degradation guardrails: reranker_max_pairs and reranker_max_tokens_per_pair
    g_cfg = full_config.get("guardrails", {})
    deg_cfg = g_cfg.get("degradation", {})

    max_pairs = deg_cfg.get("reranker_max_pairs")
    if max_pairs is not None:
        candidates = candidates[:max_pairs]

    max_tokens = deg_cfg.get("reranker_max_tokens_per_pair")
    pairs = []
    for c in candidates:
        text = c.get("text") or ""
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
    except Exception as exc:
        logger.warning(
            "Reranker failed (rate limited or API error), falling back to standard hybrid RRF results: %s",
            exc,
        )
        fallback_limit = cfg.get("top_n", 20)
        return candidates[:fallback_limit]


def _hybrid_rerank(query, cfg, full_config, filters):
    candidate_k = cfg.get("candidate_k", 80)
    return _hybrid(query, cfg, full_config, filters, fuse_top_k=candidate_k)



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