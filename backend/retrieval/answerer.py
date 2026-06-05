"""
backend/retrieval/answerer.py
──────────────────────────────
Generates a grounded, cited answer from retrieved chunks and logs
the Q&A turn to PostgreSQL (conversations table in init_db.sql).

Schema used:
    conversations(session_id UUID, turn INTEGER, question TEXT, answer TEXT)

Called after RetrievalTool — reads state["retrieved_chunks"] + state["query"]
and writes state["answer"] + state["citations"].
"""
from __future__ import annotations

import logging
from typing import Optional

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


def answer(
    query: str,
    chunks: list[Chunk],
    config: dict,
    session_id: Optional[str] = None,
    turn: int = 1,
    max_tokens: int = 512,
) -> dict:
    """
    Generate grounded answer and log to Postgres.

    Parameters
    ----------
    query      : user's question (state["standalone_query"] or state["query"])
    chunks     : state["retrieved_chunks"]
    config     : full pipeline config
    session_id : state["session_id"] — used for conversations table
    turn       : conversation turn number
    max_tokens : max LLM output tokens

    Returns
    -------
    dict: { answer, sources, citations, model }
    """
    if not chunks:
        answer_text = "No relevant passages found in the provided documents."
        _log(session_id, turn, query, answer_text)
        return {"answer": answer_text, "sources": [], "citations": [], "model": "n/a"}

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        ref   = chunk.get("source_ref") or {}
        label = f"{ref.get('filename', 'unknown')}, p.{ref.get('page', '?')}"
        context_blocks.append(f"[{i}] ({label})\n{chunk.get('text', '')}")

    user_msg = "Context:\n\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {query}"

    llm      = get_llm(config)
    response = llm.invoke([
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ])
    answer_text = (response.content or "").strip()

    # Build citations list (matches state["citations"] shape in tool.py)
    citations = []
    for chunk in chunks:
        ref  = chunk.get("source_ref") or {}
        meta = chunk.get("metadata") or {}
        citations.append({
            "filename":   ref.get("filename"),
            "page":       ref.get("page"),
            "snippet":    (chunk.get("text") or "")[:200],
            "image_path": meta.get("image_path"),
            "table_data": meta.get("table_data"),
        })

    # Log to Postgres conversations table
    _log(session_id, turn, query, answer_text)

    return {
        "answer":    answer_text,
        "sources":   [c.get("source_ref", {}) for c in chunks],
        "citations": citations,
        "model":     getattr(llm, "model_name", str(llm)),
    }


def _log(session_id: Optional[str], turn: int, question: str, answer: str) -> None:
    """Write to conversations table — silently skip if session_id is missing."""
    if not session_id:
        return
    try:
        PGStore.get().log_conversation(session_id, turn, question, answer)
    except Exception as exc:
        logger.warning("Failed to log conversation to Postgres: %s", exc)