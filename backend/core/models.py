"""Shared local model loaders. Every tool that needs a model calls these.

Models are heavy. Loading the same model twice (e.g. once in chunk_tool and
once in embed_tool) wastes memory and slows startup. These module-level
singletons load each model exactly once.

Usage:
    from backend.core.models import get_dense_model, get_sparse_model, get_reranker

    dense  = get_dense_model(config)   # SentenceTransformer, 768-dim (nomic)
    sparse = get_sparse_model(config)  # fastembed BM25
    rerank = get_reranker(config)      # CrossEncoder for reranking

Config keys used:
    config["embeddings"]["dense_model"]    — default: nomic-ai/nomic-embed-text-v1.5
    config["embeddings"]["sparse_model"]   — default: Qdrant/bm25
    config["embeddings"]["reranker_model"] — default: BAAI/bge-reranker-large
"""
from __future__ import annotations

# nomic-embed-text-v1.5 expects task-instruction prefixes (it is a Matryoshka,
# instruction-tuned model). Documents and queries MUST use different prefixes:
#   embed_tool (indexing):   DENSE_DOCUMENT_PREFIX + text
#   retrieval (querying):    DENSE_QUERY_PREFIX + query
DENSE_DOCUMENT_PREFIX = "search_document: "
DENSE_QUERY_PREFIX = "search_query: "

DEFAULT_DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"

_dense_model = None
_sparse_model = None
_reranker = None


def warm_up(config: dict | None = None) -> None:
    """Initialize torch (load the dense model) BEFORE paddle (paddleocr) is ever
    imported. paddle imported first corrupts torch's tensor allocator ->
    "Tensor holds no memory" crash at embed time. Call this at process start (API
    startup / pipeline entry); it's idempotent and a no-op once the model loaded.
    """
    cfg = config or {"embeddings": {"dense_model": DEFAULT_DENSE_MODEL}}
    try:
        get_dense_model(cfg)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("model warm_up failed: %s", exc)


def get_dense_model(config: dict):
    """Return the shared SentenceTransformer (nomic-embed-text-v1.5, 768-dim).

    nomic is distributed with custom modelling code, so trust_remote_code=True
    is required. See DENSE_DOCUMENT_PREFIX / DENSE_QUERY_PREFIX for the
    instruction prefixes documents vs queries must carry.
    """
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = config["embeddings"]["dense_model"]
        _dense_model = SentenceTransformer(model_name, trust_remote_code=True)
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
