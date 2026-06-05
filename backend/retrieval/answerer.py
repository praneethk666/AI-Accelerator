"""
backend/retrieval/answerer.py
──────────────────────────────
Generates a grounded, cited answer from retrieved chunks.
LLM comes from get_llm(config) — provider and model are set in global.yaml,
not hardcoded here.

Separate from retrieval so each can be tested in isolation.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.schemas import Chunk
from backend.core.llm_client import get_llm

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
    max_tokens: int = 512,
) -> dict:
    """
    Parameters
    ----------
    query     : the user's question
    chunks    : retrieved passages (output of RetrievalTool)
    config    : full pipeline config — LLM provider/model read from here
    max_tokens: max answer tokens

    Returns
    -------
    dict: { answer, sources, model }
    """
    if not chunks:
        return {"answer": "No relevant passages found.", "sources": [], "model": "n/a"}

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        ref   = chunk.get("source_ref") or {}
        label = f"{ref.get('filename', 'unknown')}, p.{ref.get('page', '?')}"
        context_blocks.append(f"[{i}] ({label})\n{chunk.get('text', '')}")

    user_msg = f"Context:\n\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {query}"

    llm      = get_llm(config)
    response = llm.invoke([
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ])
    answer_text = response.content or ""

    return {
        "answer":  answer_text.strip(),
        "sources": [c.get("source_ref", {}) for c in chunks],
        "model":   getattr(llm, "model_name", str(llm)),
    }