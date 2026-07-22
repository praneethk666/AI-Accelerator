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

from backend.core.tool import PipelineState
from backend.core.schemas import Chunk
from backend.core import usage
from backend.core.llm_client import get_llm_for, clean_message_content
from backend.retrieval.pg_store import PGStore
from backend.core.tracing import record_handled_error

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You are a precise document-intelligence assistant for technical and enterprise "
    "documents (service manuals, datasheets, reports).\n"
    "Rules:\n"
    "- Answer using ONLY the provided context passages. Never use outside knowledge "
    "or guess.\n"
    "- Cite every fact inline as [filename, p.N], using the passage you drew it from; "
    "if several support it, cite the most specific.\n"
    "- Copy exact values VERBATIM — part numbers, model names, measurements, torque "
    "specs, fault codes. Never paraphrase or round a number.\n"
    "- If a relevant table or figure caption is in the context, use it and cite it.\n"
    "- If the context only partially answers, answer what it supports and state plainly "
    "what is missing.\n"
    "- If the answer is not in the context, reply EXACTLY: "
    "'I could not find this in the provided documents.'\n"
    "- Be direct: lead with the answer, don't restate the question, no filler."
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


# Same threshold enrich_chunks uses to decide a chunk is too short for LLM
# summarization — missing tags.summary is a ready-made "this chunk is thin"
# signal we don't have to invent (headings, bare labels like "Alarm code:
# 11H"). token_count is a fallback for the rare case summary is absent for
# another reason (summarize: false in config).
_THIN_TOKEN_FLOOR = 20


def _is_thin(chunk: dict) -> bool:
    tags = chunk.get("tags") or {}
    if not (tags.get("summary") or "").strip():
        return True
    return int(chunk.get("token_count") or 0) < _THIN_TOKEN_FLOOR


def _expand_thin_chunks(chunks: list[dict], max_pages: int = 5) -> list[dict]:
    """Auto-Merging Retrieval: a chunk too thin to be a useful standalone answer
    source (see _is_thin) is replaced with its FULL PAGE content from
    document_blocks — same escape hatch the get_page_context agent tool offers,
    but applied automatically rather than depending on an LLM to notice a
    citation looks fragmented and decide to ask for it.

    Validated live, 21-Jul: an agent had get_page_context available and a
    citation for the fuller page sitting right in front of it, and answered
    from a shallower source instead — pure LLM judgment about "is this result
    good enough" isn't reliable enough on its own. This makes the expansion
    unconditional for thin chunks instead of optional.

    Caps how many DISTINCT pages get fetched (max_pages) to bound DB round
    trips when many candidate chunks happen to be thin, and dedupes by
    (document_id, page) since several thin chunks often share one page (e.g.
    a bare heading AND a bare label both orphaned on the same alarm-code page)."""
    thin_ids = {c["chunk_id"] for c in chunks if _is_thin(c)}
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
            doc_id, page = chunk.get("document_id"), ref.get("page")
            if not doc_id or page is None:
                continue
            doc_id = str(doc_id)  # psycopg returns uuid.UUID for this column; keep
                                  # the cache key + get_blocks() param a plain str
            key = (doc_id, page)
            if key in cache:
                continue
            if store is None:
                store = PostgresStore()
            try:
                blocks = store.get_blocks(doc_id)
            except Exception:
                logger.exception("get_blocks failed expanding thin chunk (doc %s, page %s)",
                                 doc_id, page)
                continue
            page_blocks = [
                b for b in blocks
                if isinstance(b.get("source_ref"), dict) and b["source_ref"].get("page") == page
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
        key = (str(doc_id) if doc_id else doc_id, ref.get("page"))
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
            context_blocks = []
            used_tokens = 0
            for i, chunk in enumerate(chunks, start=1):
                ref   = chunk.get("source_ref") or {}
                label = _locator(ref)
                summary = (chunk.get("tags") or {}).get("summary")
                header = f"[{i}] ({label})" + (f" — {summary}" if summary else "")
                block = f"{header}\n{chunk.get('text') or ''}"
                if max_ctx_tokens and context_blocks and used_tokens + len(block) // 4 > max_ctx_tokens:
                    break
                context_blocks.append(block)
                used_tokens += len(block) // 4

            user_msg = (
                "Context:\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nQuestion: {query}"
            )

            # answering is reasoning-heavy. Resolution: query.answerer.model ->
            # llm.answer_model -> global llm.model.
            llm      = get_llm_for(config, config.get("query", {}).get("answerer"),
                                   default_model=config["llm"].get("answer_model"))
            response = llm.invoke([
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ])
            usage.record_from_message("answer", response)
            answer_text = clean_message_content(response.content).strip()

            # Build citations — image_path and table_data are top-level chunk fields.
            # source_ref varies by file type (page for PDF, sheet for Excel, slide
            # for PPT), so read every locator field with .get and never assume page.
            citations = []
            for chunk in chunks:
                ref = chunk.get("source_ref") or {}
                tags = chunk.get("tags") or {}
                citations.append({
                    "filename":    ref.get("filename"),
                    "page":        ref.get("page"),
                    "document_id": chunk.get("document_id"),
                    "score":       chunk.get("_score"),
                    "sheet":       ref.get("sheet"),
                    "slide":       ref.get("slide"),
                    "snippet":     (chunk.get("text") or "")[:200],
                    "summary":     tags.get("summary"),
                    "keywords":    tags.get("keywords"),
                    "image_path":  chunk.get("image_path"),
                    "table_data":  chunk.get("table_data"),
                })

            # Don't attach a source list to a 'not found' answer — it drew on nothing.
            if _looks_like_refusal(answer_text):
                citations = []

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


def _locator(ref: dict) -> str:
    """Human-readable source label that works for any file type:
    'report.pdf, p.3' / 'sheet.xlsx, Sheet1' / 'deck.pptx, slide 4'."""
    name = ref.get("filename") or "source"
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