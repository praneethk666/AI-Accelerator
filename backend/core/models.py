"""Shared local model loaders. Every tool that needs a model calls these.

Models are heavy — bge-large is ~1.3 GB. Loading the same model twice
(e.g. once in chunk_tool and once in embed_tool) wastes memory and slows
startup. These module-level singletons load each model exactly once.

Usage:
    from backend.core.models import get_dense_model, get_sparse_model, get_reranker

    dense  = get_dense_model(config)   # SentenceTransformer, 1024-dim
    sparse = get_sparse_model(config)  # fastembed BM25
    rerank = get_reranker(config)      # CrossEncoder for reranking

Config keys used:
    config["embeddings"]["dense_model"]    — default: BAAI/bge-large-en-v1.5
    config["embeddings"]["sparse_model"]   — default: Qdrant/bm25
    config["embeddings"]["reranker_model"] — default: BAAI/bge-reranker-large
"""
from __future__ import annotations

_dense_model = None
_sparse_model = None
_reranker = None


def get_dense_model(config: dict):
    """Return the shared SentenceTransformer (bge-large, 1024-dim).

    NOTE on bge-large query vs document encoding:
      - Indexing documents (embed_tool): encode text directly, no prefix.
      - Encoding a query (retrieval_tool): prepend the instruction prefix:
        "Represent this sentence for searching relevant passages: " + query
    """
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = config["embeddings"]["dense_model"]
        _dense_model = SentenceTransformer(model_name)
    return _dense_model


def get_sparse_model(config: dict):
    """Return the shared fastembed BM25 sparse encoder."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        model_name = config["embeddings"]["sparse_model"]
        _sparse_model = SparseTextEmbedding(model_name)
    return _sparse_model


def get_reranker(config: dict):
    """Return the shared CrossEncoder reranker (bge-reranker-large)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        model_name = config["embeddings"]["reranker_model"]
        _reranker = CrossEncoder(model_name)
    return _reranker
