"""
backend/retrieval/answerer.py
──────────────────────────────
AnswererTool — implements the Tool Protocol from backend/core/tool.py.

  run(state, config)
    READS  state["query"]             str         — raw user question
           state["retrieved_chunks"]  list[Chunk] — from RetrievalTool
           state["session_id"]        str         — for conversations table
    WRITES state["answer"]            str         — final answer text
           state["citations"]         list        — per-chunk citation dicts
    ERRORS state["errors"]            list        — append only, never raise

Chunk schema (backend/core/schemas.py):
    chunk_id    str
    document_id str
    text        str
    tags        dict   — industry, doc_type, topic, section, keywords
    source_ref  dict   — filename, page
    table_data  dict   — headers + rows (non-null for table chunks)
    image_path  str    — non-null for image caption chunks
    token_count int
    vector      list
    sparse_vector dict

Logs Q&A turn to PostgreSQL conversations table (scripts/init_db.sql):
    conversations(session_id UUID, turn INTEGER, question TEXT, answer TEXT)
"""
from __future__ import annotations

import logging
import re

from backend.core.tool import PipelineState
from backend.core.schemas import Chunk
from backend.core import usage
from backend.core.llm_client import get_llm_for, resolve_model_provider, clean_message_content
from backend.retrieval.pg_store import PGStore
from backend.core.tracing import record_handled_error

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You are a precise document-intelligence assistant for enterprise documents "
    "(manuals, reports, contracts, invoices, spreadsheets, presentations, and other technical or office documents).\n"
    "Rules:\n"
    "- Answer using ONLY the provided context passages. Never use outside knowledge "
    "or guess.\n"
    "- Cite every fact inline as [filename, p.N], using the passage you drew it from; "
    "if several support it, cite the most specific.\n"
    "- Copy exact values VERBATIM — part numbers, model names, measurements, torque "
    "specs, fault codes. Never paraphrase or round a number.\n"
    "- Do NOT add your own derivations, reformulations, or 'equivalent forms' of formulas "
    "unless that exact form appears verbatim in the source. If you present multiple forms, "
    "every one must be cited from a specific passage.\n"
    "- If a relevant table or figure caption is in the context, use it and cite it.\n"
    "- If the context partially answers or uses corresponding domain terms (e.g. danger/warning levels or safety procedures for incident/severity questions), answer what the context supports with citations and state plainly if specific aspects (like organizational escalation tiers) are not defined.\n"
    "- If the answer is not in the context, reply EXACTLY: "
    "'I could not find this in the provided documents.'\n"
    "- Be direct: lead with the answer, don't restate the question, no filler.\n"
    "- For mathematical formulas, use standard Markdown LaTeX syntax: '$$formula$$' for block/display equations and '$formula$' for inline equations. Never use single brackets '[ ... ]' for math blocks."
)


# Markers of a 'the docs don't answer this' reply — used to suppress a misleading
# full source list on a refusal (the answer drew on nothing).
_REFUSAL_MARKERS = (
    "could not find this in the provided",
    "no relevant passages found",
    "not in the provided documents",
    "don't have information",
)


def _looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in _REFUSAL_MARKERS)


def _filter_cited_citations(answer: str, citations: list[dict]) -> list[dict]:
    """
    Parse the LLM's answer to identify which sources were actually cited.
    Returns only the citations that match bracketed index numbers or filenames and pages.
    """
    import re

    answer_lower = answer.lower()

    # 1. Parse bracketed index citations, e.g. [1], [2, p.3], [1-3]
    cited_indices = set()
    brackets = re.findall(r'\[([^\]]+)\]', answer)
    for content in brackets:
        # Check for ranges first, e.g. 1-3
        for start, end in re.findall(r'\b(\d+)\s*-\s*(\d+)\b', content):
            try:
                s, e = int(start), int(end)
                if s <= e and e - s < 50:
                    cited_indices.update(range(s, e + 1))
            except ValueError:
                pass
        # Individual index numbers
        for num_str in re.findall(r'\b\d+\b', content):
            try:
                cited_indices.add(int(num_str))
            except ValueError:
                pass

    # Helper function to find numbers/ranges in a snippet of text
    def _expand_ranges_and_numbers(text: str) -> set[int]:
        numbers = set()
        # Find ranges like '3-5', '3 to 5', '3- 5'
        range_patterns = [
            r'\b(\d+)\s*-\s*(\d+)\b',
            r'\b(\d+)\s+to\s+(\d+)\b',
        ]
        for pattern in range_patterns:
            for start, end in re.findall(pattern, text):
                try:
                    s, e = int(start), int(end)
                    if s <= e and e - s < 100:
                        numbers.update(range(s, e + 1))
                except ValueError:
                    pass
        # Also find individual numbers
        for num_str in re.findall(r'\b\d+\b', text):
            try:
                numbers.add(int(num_str))
            except ValueError:
                pass
        return numbers

    # Generic terms to avoid false positives (e.g. matching "document" inside text to document.docx)
    GENERIC_FILE_WORDS = {
        "document", "doc", "report", "manual", "file", "presentation", "slide", 
        "sheet", "pdf", "word", "excel", "powerpoint", "ppt", "xlsx", "docx", "pages"
    }

    # 2. Filter citations
    filtered = []
    
    for idx, cit in enumerate(citations, start=1):
        is_cited = False
        
        # Check A: Cited by index [i]
        if idx in cited_indices:
            is_cited = True
            
        else:
            # Check B: Cited by name and optionally page/slide/sheet
            fname = cit.get("filename") or ""
            if fname:
                fname_clean = fname.lower()
                # Also try without extension
                fname_no_ext = fname_clean.rsplit('.', 1)[0] if '.' in fname_clean else fname_clean
                
                # To avoid false positive on very short file names, require name to be at least 3 chars
                if len(fname_no_ext) >= 3:
                    # Escape for regex
                    esc_fname = re.escape(fname_clean)
                    esc_fname_no_ext = re.escape(fname_no_ext)
                    
                    # Pattern matching with word boundary on both sides
                    # Matches either the exact filename or the filename without extension
                    # If it's a generic word, we require exact match with extension
                    if fname_no_ext in GENERIC_FILE_WORDS:
                        pattern = rf'\b{esc_fname}\b'
                    else:
                        pattern = rf'\b(?:{esc_fname}|{esc_fname_no_ext})\b'
                    
                    matches = list(re.finditer(pattern, answer_lower))
                    if matches:
                        # Filename was mentioned! Now check if specific pages are cited.
                        page_val = cit.get("page") or cit.get("slide")
                        sheet_val = cit.get("sheet")
                        
                        has_specific_citation = False
                        matches_any_page_spec = False
                        
                        for m in matches:
                            # Look in a window of 100 characters before and after the mention
                            start_pos = max(0, m.start() - 100)
                            end_pos = min(len(answer_lower), m.end() + 100)
                            context = answer_lower[start_pos:end_pos]
                            
                            # Does the context contain page/slide indicators?
                            has_page_word = bool(re.search(r'\b(?:page|pages|p|pg|slide|slides|sheet|sheets)\b|[()]', context))
                            
                            if has_page_word:
                                has_specific_citation = True
                                nums = _expand_ranges_and_numbers(context)
                                
                                # Check page/slide match
                                if page_val is not None:
                                    try:
                                        p_int = int(float(page_val))
                                        if p_int in nums:
                                            matches_any_page_spec = True
                                            break
                                    except (ValueError, TypeError):
                                        pass
                                        
                                # Check sheet match
                                if sheet_val and str(sheet_val).lower() in context:
                                    matches_any_page_spec = True
                                    break
                                    
                        if has_specific_citation:
                            if matches_any_page_spec:
                                is_cited = True
                        else:
                            # General mention without specific page references -> count as cited
                            is_cited = True
                            
        if is_cited:
            # Find earliest character position where this citation (index or page) appears in answer
            pos = 999999
            page_val = cit.get("page") or cit.get("slide")
            # Check [idx] mention
            m_idx = answer.find(f"[{idx}]")
            if m_idx != -1:
                pos = min(pos, m_idx)
            # Check p.X mention
            if page_val is not None:
                for p_pat in (f"p.{page_val}", f"p. {page_val}", f"page {page_val}"):
                    m_p = answer_lower.find(p_pat)
                    if m_p != -1:
                        pos = min(pos, m_p)
            cit["_pos"] = pos
            filtered.append(cit)
            
    # 3. Fallback: if parsing failed to extract any valid citation matches, 
    # keep all retrieved chunks to ensure the sources list is not empty.
    res = filtered if filtered else citations

    # Rank citations primarily by CONTENT OVERLAP with the final answer
    # (so the page/chunk containing the majority of the answer is ordered FIRST),
    # then by mention frequency, retrieval score, and position.
    def _content_overlap(ans: str, snip: str) -> float:
        if not ans or not snip:
            return 0.0
        ans_words = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', ans.lower()))
        snip_words = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', snip.lower()))
        if not ans_words or not snip_words:
            return 0.0
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "are", "these",
            "source", "page", "pages", "pdf", "docx", "xlsx", "manual", "document",
            "include", "includes", "including", "system", "into", "their", "will",
            "which", "also", "have", "been", "they", "were", "when", "what"
        }
        return float(len((ans_words & snip_words) - stop_words))

    for c in res:
        overlap = _content_overlap(answer, c.get("snippet") or "")
        page_val = c.get("page") or c.get("slide")
        mentions = 0
        if page_val is not None:
            for p_pat in (f"p.{page_val}", f"p. {page_val}", f"page {page_val}", f"page: {page_val}"):
                mentions += answer_lower.count(p_pat)
        c["_overlap"] = overlap
        c["_mentions"] = mentions
        c["_score_val"] = float(c.get("score") or 0.0)

    res.sort(
        key=lambda c: (
            c.get("_overlap", 0.0),
            c.get("_mentions", 0),
            c.get("_score_val", 0.0),
            -c.get("_pos", 999999),
        ),
        reverse=True
    )
    for c in res:
        c.pop("_pos", None)
        c.pop("_overlap", None)
        c.pop("_mentions", None)
        c.pop("_score_val", None)
    return res


# Same threshold enrich_chunks uses to decide a chunk is too short for LLM
# summarization — missing tags.summary is a ready-made "this chunk is thin"
# signal we don't have to invent (headings, bare labels like "Alarm code:
# 11H"). token_count is a fallback for the rare case summary is absent for
# another reason (summarize: false in config).
_THIN_TOKEN_FLOOR = 120


def _is_thin(chunk: dict) -> bool:
    tags = chunk.get("tags") or {}
    if not (tags.get("summary") or "").strip():
        return True
    if tags.get("chunk_type") in ("table", "list", "header"):
        return True
    return int(chunk.get("token_count") or 0) < _THIN_TOKEN_FLOOR


def _expand_thin_chunks(chunks: list[dict], max_pages: int = 5) -> list[dict]:
    """Auto-Merging Retrieval: a chunk too thin or fragmented to be a useful standalone answer
    source (see _is_thin, or multiple chunks from the same page/slide/sheet) is replaced with its FULL PAGE content from
    document_blocks — same escape hatch the get_page_context agent tool offers,
    but applied automatically rather than depending on an LLM to notice a
    citation looks fragmented and decide to ask for it.

    Caps how many DISTINCT pages/slides/sheets get fetched (max_pages) to bound DB round
    trips when many candidate chunks happen to be thin, and dedupes by
    (document_id, page/slide/sheet) since several thin chunks often share one page."""
    # Count how many retrieved chunks come from each page/slide/sheet to detect fragmented pages
    page_counts: dict[tuple, int] = {}
    for c in chunks:
        ref = c.get("source_ref") or {}
        doc_id = c.get("document_id")
        page_val = ref.get("page") or ref.get("slide") or ref.get("sheet")
        if doc_id and page_val is not None:
            key = (str(doc_id), page_val)
            page_counts[key] = page_counts.get(key, 0) + 1

    thin_ids = set()
    for c in chunks:
        ref = c.get("source_ref") or {}
        filename = ref.get("filename") or ""
        # Skip Excel chunks entirely from auto-expansion to prevent token bloat
        if filename.lower().endswith((".xlsx", ".xls", ".xlsm")) or ref.get("sheet") is not None:
            continue

        doc_id = c.get("document_id")
        page_val = ref.get("page") or ref.get("slide") or ref.get("sheet")
        key = (str(doc_id), page_val) if doc_id and page_val is not None else None
        if _is_thin(c) or (key and page_counts.get(key, 0) > 1):
            thin_ids.add(c["chunk_id"])

    if not thin_ids:
        return chunks

    from backend.storage.postgres_store import PostgresStore

    cache: dict[tuple, str] = {}
    store = None
    try:
        for chunk in chunks:
            if chunk["chunk_id"] not in thin_ids or len(cache) >= max_pages:
                continue
            ref = chunk.get("source_ref") or {}
            doc_id = chunk.get("document_id")
            page_val = ref.get("page") or ref.get("slide") or ref.get("sheet")
            if not doc_id or page_val is None:
                continue
            doc_id = str(doc_id)  # psycopg returns uuid.UUID for this column; keep
                                  # the cache key + get_blocks() param a plain str
            key = (doc_id, page_val)
            if key in cache:
                continue
            if store is None:
                store = PostgresStore()
            try:
                blocks = store.get_blocks(doc_id)
            except Exception:
                logger.exception("get_blocks failed expanding thin chunk (doc %s, page/slide/sheet %s)",
                                 doc_id, page_val)
                continue
            page_blocks = [
                b for b in blocks
                if isinstance(b.get("source_ref"), dict) and (
                    b["source_ref"].get("page") == page_val or
                    b["source_ref"].get("slide") == page_val or
                    b["source_ref"].get("sheet") == page_val
                )
            ]
            parts = [b.get("text") for b in page_blocks if (b.get("text") or "").strip()]
            if parts:
                cache[key] = "\n\n".join(parts)
    finally:
        if store is not None:
            store.close()

    if not cache:
        return chunks

    out = []
    for chunk in chunks:
        ref = chunk.get("source_ref") or {}
        doc_id = chunk.get("document_id")
        page_val = ref.get("page") or ref.get("slide") or ref.get("sheet")
        key = (str(doc_id) if doc_id else doc_id, page_val)
        if chunk["chunk_id"] in thin_ids and key in cache:
            c = dict(chunk)
            c["text"] = cache[key]
            out.append(c)
        else:
            out.append(chunk)
    return out


class AnswererTool:
    """
    Implements the Tool Protocol (backend/core/tool.py).

    State contract:
        READS  query             str         ← raw user question
               retrieved_chunks list[Chunk] ← from RetrievalTool
               session_id       str         ← conversation session
    WRITES answer            str
               citations        list
        ERRORS errors            list
    """

    name: str = "answerer"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        query:   str         = state["query"]
        standalone_query: str = state.get("standalone_query") or query
        chunks:  list[Chunk] = state["retrieved_chunks"] or []
        session_id: str      = state["session_id"]
        turn:    int         = len(state["conversation_history"] or []) + 1

        if not chunks:
            answer_text = "No relevant passages found in the provided documents."
            state["answer"]    = answer_text
            state["citations"] = []
            _log(session_id, turn, query, answer_text, config)
            return state

        chunks = _expand_thin_chunks(chunks)

        try:
            # Enforce the configured context budget so a burst of large bge-m3 chunks
            # (up to max_sub_questions x rerank_top_k) can't blow up cost/context on the
            # paid model. Approx 4 chars/token; 0/unset => no cap. Chunks arrive best-first.
            max_ctx_tokens = int((config.get("query") or {}).get("max_context_tokens") or 0)
            if not max_ctx_tokens:
                max_ctx_tokens = int((config.get("guardrails") or {}).get("token_budget", {}).get("max_context_tokens") or 20000)
            context_blocks = []
            used_tokens = 0
            for i, chunk in enumerate(chunks, start=1):
                ref   = chunk.get("source_ref") or {}
                label = _locator(ref)
                summary = (chunk.get("tags") or {}).get("summary")
                header = f"[{i}] ({label})" + (f" — {summary}" if summary else "")
                chunk_text = chunk.get("text") or ""
                block = f"{header}\n{chunk_text}"
                
                block_tokens = len(block) // 4
                if max_ctx_tokens and used_tokens + block_tokens > max_ctx_tokens:
                    # Truncate first block to fit budget if it is already too large on its own
                    if not context_blocks:
                        allowed_chars = max(0, max_ctx_tokens * 4 - len(header) - 50)
                        truncated_text = chunk_text[:allowed_chars]
                        block = f"{header}\n{truncated_text}\n... [truncated to fit token budget]"
                        context_blocks.append(block)
                        used_tokens += max_ctx_tokens
                    break
                context_blocks.append(block)
                used_tokens += block_tokens

            user_msg = (
                "Context:\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nQuestion: {standalone_query}"
            )

            # answering is reasoning-heavy. Resolution: query.answerer.model ->
            # llm.answer_model -> global llm.model.
            answerer_cfg  = config.get("query", {}).get("answerer") or {}
            model_name, provider_name = resolve_model_provider(
                config, answerer_cfg, default_model=config["llm"].get("answer_model")
            )
            llm = get_llm_for(config, answerer_cfg, default_model=config["llm"].get("answer_model"))
            prompt_messages = [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ]
            response = llm.invoke(prompt_messages)
            usage.record_from_message("answer", response, prompt=prompt_messages, model=model_name, provider=provider_name)
            answer_text = clean_message_content(response.content).strip()

            # Build citations — image_path and table_data are top-level chunk fields.
            # source_ref varies by file type (page for PDF, sheet for Excel, slide
            # for PPT), so read every locator field with .get and never assume page.
            citations = []
            for chunk in chunks:
                ref = chunk.get("source_ref") or {}
                tags = chunk.get("tags") or {}
                citations.append({
                    "filename":    _clean_filename(ref.get("filename") or "") or ref.get("filename"),
                    "page":        ref.get("page"),
                    "document_id": chunk.get("document_id"),
                    "score":       chunk.get("_score"),
                    "sheet":       ref.get("sheet"),
                    "slide":       ref.get("slide"),
                    "snippet":     (chunk.get("text") or ""),
                    "summary":     tags.get("summary"),
                    "keywords":    tags.get("keywords"),
                    "image_path":  chunk.get("image_path"),
                    "table_data":  chunk.get("table_data"),
                    "chunk_type":  tags.get("chunk_type"),
                    "section":     tags.get("section") or ref.get("section"),
                })

            # Don't attach a source list to a 'not found' answer — it drew on nothing.
            if _looks_like_refusal(answer_text):
                citations = []
            else:
                citations = _filter_cited_citations(answer_text, citations)

            state["answer"]    = answer_text
            state["citations"] = citations

            _log(session_id, turn, query, answer_text, config)

        except Exception as exc:
            logger.error("AnswererTool failed for query %r: %s", query[:60], exc)
            errors: list = state["errors"] or []
            errors.append({"tool": "answerer", "query": query, "error": str(exc)})
            state["errors"]    = errors
            state["answer"]    = "An error occurred while generating the answer."
            state["citations"] = []
            record_handled_error("answerer_failure", str(exc), **{"query": query[:100]})

        return state


_UUID_PREFIX_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
    re.IGNORECASE,
)


def _clean_filename(name: str) -> str:
    """Strip leading UUID prefix (e.g. 'abc123..._report.pdf' -> 'report.pdf')."""
    return _UUID_PREFIX_RE.sub('', name or '') or name


def _locator(ref: dict) -> str:
    """Human-readable source label that works for any file type:
    'report.pdf, p.3' / 'sheet.xlsx, Sheet1' / 'deck.pptx, slide 4'."""
    name = _clean_filename(ref.get("filename") or "source")
    if ref.get("page") is not None:
        return f"{name}, p.{ref['page']}"
    if ref.get("sheet"):
        return f"{name}, {ref['sheet']}"
    if ref.get("slide") is not None:
        return f"{name}, slide {ref['slide']}"
    return name


def _log(
    session_id: str,
    turn: int,
    question: str,
    answer: str,
    config: dict,
) -> None:
    """Write to conversations table — silently skip if session_id is missing."""
    if not session_id:
        return
    try:
        PGStore.log_conversation(config, session_id, turn, question, answer)
    except Exception as exc:
        logger.warning("Failed to log conversation to Postgres: %s", exc)