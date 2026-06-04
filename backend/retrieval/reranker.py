"""
backend/retrieval/reranker.py
──────────────────────────────
Cross-encoder reranker.  Takes a query + candidate chunks, scores every
(query, chunk_text) pair with a cross-encoder, and re-orders by that score.

Model is loaded lazily and cached.  Model name comes from config.
Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"  (fast, good enough to demo).
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)

_MODELS: dict[str, object] = {}


def rerank(
    query: str,
    candidates: list[Chunk],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[Chunk]:
    """
    Re-rank *candidates* using a cross-encoder relevance model.

    Parameters
    ----------
    query      : the original user query
    candidates : list of Chunk objects to rerank (typically 20-40)
    model_name : HuggingFace cross-encoder model id

    Returns
    -------
    list[Chunk] — same chunks, re-ordered best-first by cross-encoder score
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, c.get("text") or "") for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked]


# ── internals ─────────────────────────────────────────────────────────────────

def _get_model(model_name: str):
    if model_name not in _MODELS:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from e
        logger.info("Loading reranker: %s", model_name)
        _MODELS[model_name] = CrossEncoder(model_name)
    return _MODELS[model_name]