"""
backend/retrieval/embedder.py
─────────────────────────────
Thin wrapper around sentence-transformers.
Cached per model name so we don't reload weights on every call.
Model choice comes from config, never hardcoded here.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import — keeps startup fast if embedder isn't used yet.
_BACKENDS: dict[str, object] = {}


def embed_text(text: str, model_name: str) -> list[float]:
    """
    Return a unit-norm embedding vector for *text* using *model_name*.

    Parameters
    ----------
    text       : the string to embed (query or document chunk)
    model_name : a sentence-transformers model id, e.g.
                 "sentence-transformers/all-MiniLM-L6-v2"

    Returns
    -------
    list[float] — dense embedding (length depends on model)
    """
    model = _get_model(model_name)
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str], model_name: str) -> list[list[float]]:
    """
    Embed a list of texts in one batch (more efficient than looping).
    Returns a list of unit-norm vectors in the same order.
    """
    model = _get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ── internals ────────────────────────────────────────────────────────────────

def _get_model(model_name: str):
    """Load and cache a SentenceTransformer model by name."""
    if model_name not in _BACKENDS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from e
        logger.info("Loading embedding model: %s", model_name)
        _BACKENDS[model_name] = SentenceTransformer(model_name)
    return _BACKENDS[model_name]