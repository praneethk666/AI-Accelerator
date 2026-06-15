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
from backend.core.llm_client import get_llm
from backend.retrieval.pg_store import PGStore

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You are a precise document-intelligence assistant. "
    "Answer the user's question using ONLY the context passages provided. "
    "For every fact add an inline citation like [filename, p.N]. "
    "If the answer is not in the context say: "
    "'I could not find this in the provided documents.' "
    "Do not invent information."
)


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

        try:
            context_blocks = []
            for i, chunk in enumerate(chunks, start=1):
                ref   = chunk.get("source_ref") or {}
                label = _locator(ref)
                context_blocks.append(f"[{i}] ({label})\n{chunk.get('text') or ''}")

            user_msg = (
                "Context:\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nQuestion: {query}"
            )

            llm      = get_llm(config)
            response = llm.invoke([
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ])
            answer_text = (response.content or "").strip()

            # Build citations — image_path and table_data are top-level chunk fields.
            # source_ref varies by file type (page for PDF, sheet for Excel, slide
            # for PPT), so read every locator field with .get and never assume page.
            citations = []
            for chunk in chunks:
                ref = chunk.get("source_ref") or {}
                citations.append({
                    "filename":   ref.get("filename"),
                    "page":       ref.get("page"),
                    "sheet":      ref.get("sheet"),
                    "slide":      ref.get("slide"),
                    "snippet":    (chunk.get("text") or "")[:200],
                    "image_path": chunk.get("image_path"),
                    "table_data": chunk.get("table_data"),
                })

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