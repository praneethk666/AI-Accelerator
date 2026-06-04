"""
backend/retrieval/llm_client.py
────────────────────────────────
Thin LLM wrapper used only for HyDE (hypothetical document generation).

Uses Groq API.

Model name comes from config["hyde_model"], never hardcoded here.
"""

from __future__ import annotations

import logging
import os

from groq import Groq

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Write a short, factual paragraph that would directly answer the following question. "
    "Do not say 'I don't know' — produce the best hypothetical answer you can. "
    "Reply with ONLY the answer paragraph, no preamble."
)


def generate_hypothetical_answer(
    query: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate a hypothetical answer for the query.

    Used by HyDE retrieval:
    embed this text instead of the raw query.

    Parameters
    ----------
    query : the user's question
    model : Groq model id

    Returns
    -------
    str — a short passage that looks like an answer
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": _HYDE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.3,
        max_completion_tokens=200,
    )

    text = response.choices[0].message.content or ""

    logger.debug(
        "HyDE hypothetical answer (%d chars): %.80s…",
        len(text),
        text,
    )

    return text.strip()