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
    "- If the context only partially answers, answer what it supports and state plainly "
    "what is missing.\n"
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
    Returns only the citations that match bracketed index numbers or filenames.
    """
    import re
    
    # 1. Extract all text content within brackets, e.g. [1], [2, p.3], [Vinod-Nerella.pdf, p.1]
    brackets = re.findall(r'\[([^\]]+)\]', answer)
    if not brackets:
        # If the LLM did not generate any citations, fall back to returning all retrieved chunks
        return citations
        
    cited_indices = set()
    cited_filenames = set()
    
    for content in brackets:
        # Extract individual integers (index references like [1], [1, 2])
        for num_str in re.findall(r'\b\d+\b', content):
            cited_indices.add(int(num_str))
            
        # Match by filename or cleaned filename (case-insensitive)
        for cit in citations:
            fname = cit.get("filename") or ""
            if fname and fname.lower() in content.lower():
                cited_filenames.add(fname.lower())
                
    # 2. Filter the citation list
    filtered = []
    for idx, cit in enumerate(citations, start=1):
        is_cited = False
        
        # Check if cited by index [i]
        if idx in cited_indices:
            is_cited = True
            
        # Check if cited by filename reference
        elif cit.get("filename") and cit["filename"].lower() in cited_filenames:
            is_cited = True
            
        if is_cited:
            filtered.append(cit)
            
    # 3. Fallback: if parsing failed to extract any valid citation matches, 
    # keep all retrieved chunks to ensure the sources list is not empty.
    return filtered if filtered else citations


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


def _select_grounding_targets(chunks: list[dict], top_k: int) -> list[dict]:
    """Top-ranked, DISTINCT-page PDF citations to attach a page IMAGE for —
    image-grounded answering (28-Jul): the answer LLM can visually cross-check
    extracted text against the real page, catching a case like the real bug
    found tonight (a table's "Indication" icon column misread by the table-OCR
    engine on 9/11 rows while every other column was correct).

    chunks arrive best-first (already reranked). Only chunks with a PDF page
    locator qualify (source_ref.page) — Excel/PPT citations have no page-image
    cache. Dedups by (document_id, page) so two top-ranked chunks from the same
    page don't burn two image slots."""
    seen: set[tuple] = set()
    targets: list[dict] = []
    for chunk in chunks:
        if len(targets) >= top_k:
            break
        ref = chunk.get("source_ref") or {}
        doc_id, page = chunk.get("document_id"), ref.get("page")
        if not doc_id or page is None:
            continue
        key = (str(doc_id), page)
        if key in seen:
            continue
        seen.add(key)
        targets.append({"document_id": str(doc_id), "page": page, "source_ref": ref})
    return targets


def _load_grounding_images(targets: list[dict]) -> list[dict]:
    """Load the already-rendered page JPEG (document_pages, from ingestion —
    see backend/pipeline/page_images.py) for each target and base64-encode it.

    Fails OPEN, per-image: a document ingested before this feature existed (no
    document_pages row), a missing/unreadable file, or a DB error just means
    that ONE image is skipped — never raises, so a page-image problem can never
    break answering itself (mirrors _expand_thin_chunks's own catch-and-skip
    shape for get_blocks above)."""
    if not targets:
        return []
    import base64

    from backend.pipeline.page_images import physical_path
    from backend.storage.postgres_store import PostgresStore

    out: list[dict] = []
    store = None
    try:
        store = PostgresStore()
        for t in targets:
            try:
                row = store.get_page_image(t["document_id"], t["page"])
                if not row:
                    continue
                with open(physical_path(row["image_path"]), "rb") as f:
                    data = f.read()
                out.append({
                    "label": _locator(t["source_ref"]),
                    "b64": base64.b64encode(data).decode("ascii"),
                })
            except Exception:
                logger.warning("image-ground: failed to load page image (doc %s, page %s)",
                                t["document_id"], t["page"], exc_info=True)
    except Exception:
        logger.warning("image-ground: PostgresStore unavailable, answering without images",
                        exc_info=True)
    finally:
        if store is not None:
            store.close()
    return out


def _build_user_content(user_msg: str, images: list[dict]):
    """Plain string (byte-identical to the pre-image-grounding prompt) when
    there's nothing to attach — zero behavior/cost change when the feature is
    off or no image could be loaded. With images, an OpenAI-style multimodal
    content-block list: LangChain's ChatOpenAI (the active provider) accepts
    this shape natively (backend/core/vision_client.py's openai path already
    proves it works against this exact API)."""
    if not images:
        return user_msg
    labels = ", ".join(img["label"] for img in images)
    text = (
        user_msg
        + f"\n\n[The actual source page image is attached below for: {labels}. "
        "If the image shows a value that conflicts with the extracted text above, "
        "trust the image — extracted text can misread icons, symbols, or dense "
        "table cells.]"
    )
    content: list[dict] = [{"type": "text", "text": text}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img['b64']}"},
        })
    return content


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
                + f"\n\nQuestion: {standalone_query}"
            )

            # answering is reasoning-heavy. Resolution: query.answerer.model ->
            # llm.answer_model -> global llm.model.
            answerer_cfg  = config.get("query", {}).get("answerer") or {}
            model_name, provider_name = resolve_model_provider(
                config, answerer_cfg, default_model=config["llm"].get("answer_model")
            )
            llm = get_llm_for(config, answerer_cfg, default_model=config["llm"].get("answer_model"))

            # Image-grounded answering (28-Jul): only target citations that
            # actually survived the context-budget cut above, not the full
            # pre-truncation `chunks` list.
            context_chunks = chunks[: len(context_blocks)]
            image_cfg = answerer_cfg.get("image_ground") or {}
            grounding_images: list[dict] = []
            if image_cfg.get("enabled"):
                targets = _select_grounding_targets(
                    context_chunks, int(image_cfg.get("top_k") or 1))
                grounding_images = _load_grounding_images(targets)

            content = _build_user_content(user_msg, grounding_images)
            messages = [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": content},
            ]
            try:
                response = llm.invoke(messages)
            except Exception:
                if not grounding_images:
                    raise
                # The provider/model may reject the multimodal content shape
                # (e.g. a non-vision-capable model swapped in) -- fail open by
                # retrying once as plain text rather than losing the answer.
                logger.warning("image-ground: multimodal answer call failed, "
                                "retrying text-only", exc_info=True)
                response = llm.invoke([
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ])
            usage.record_from_message("answer", response, model=model_name, provider=provider_name)
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