"""
backend/retrieval/answerer.py
──────────────────────────────
Takes retrieved chunks and produces a grounded, cited answer.
This is separate from retrieval so each can be tested independently.

The LLM is instructed to answer ONLY from the provided context and to cite
the source_ref of every fact it uses. Hallucination is reduced, not
eliminated — always inspect citations in the returned answer.
"""

from __future__ import annotations

import logging
import os

from groq import Groq

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = """\
You are a precise document-intelligence assistant.

Answer the user's question using ONLY the context passages provided.

For every fact in your answer, add an inline citation like [filename, p.N].

If the answer cannot be found in the context, say:
"I could not find this in the provided documents."

Do not invent information not present in the context.
"""


def answer(
    query: str,
    chunks: list[Chunk],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 512,
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Parameters
    ----------
    query      : the user's question
    chunks     : retrieved passages (from retrieve())
    model      : LLM model id (from config)
    max_tokens : max tokens in the answer

    Returns
    -------
    dict with keys:
        answer     : str
        sources    : list[dict]
        model      : str
    """

    if not chunks:
        return {
            "answer": "No relevant passages were found in the documents.",
            "sources": [],
            "model": model,
        }

    context_blocks = []

    for i, chunk in enumerate(chunks, start=1):
        ref = chunk.get("source_ref") or {}

        label = (
            f"{ref.get('filename', 'unknown')}, "
            f"p.{ref.get('page', '?')}"
        )

        context_blocks.append(
            f"[{i}] ({label})\n{chunk.get('text', '')}"
        )

    context_text = "\n\n".join(context_blocks)

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}"
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": _ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0.1,
        max_completion_tokens=max_tokens,
    )

    answer_text = response.choices[0].message.content or ""

    return {
        "answer": answer_text.strip(),
        "sources": [c.get("source_ref", {}) for c in chunks],
        "model": model,
    }