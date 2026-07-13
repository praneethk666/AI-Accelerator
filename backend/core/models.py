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

    Also warms the sparse + reranker models here, NOT lazily on a request. All
    three are singletons (module-level, loaded once) — if they only load on
    first use, that first use is whatever HTTP request happens to need them,
    and a slow/stuck model download (first-run HuggingFace fetch) blocks that
    request. Since this API runs a single worker with an async event loop, a
    blocking model load in a request handler freezes the ENTIRE process — every
    endpoint, including /health — not just the slow request. Warming at startup
    moves that risk to a place where it's visible in the startup log instead of
    silently hanging a user's first query.

    Each model gets a bounded timeout so a stuck download can't hang startup
    forever — it logs a clear warning and the API still starts; that model just
    stays unloaded until something calls it directly (same risk as before, but
    now you had a chance to notice at startup). Note: a timed-out load's
    background thread is NOT killed (Python can't force-stop a thread) — it may
    still finish later and populate the singleton on its own.
    """
    import logging
    logger = logging.getLogger(__name__)
    cfg = config or {"embeddings": {"dense_model": DEFAULT_DENSE_MODEL}}

    def _try(name: str, loader, timeout_s: float) -> None:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(loader, cfg)
            try:
                future.result(timeout=timeout_s)
            except FutureTimeout:
                logger.warning(
                    "model warm_up: %s did not finish within %ss (likely a slow/stuck "
                    "download) — API is starting anyway; the first request that needs "
                    "it may be slow or hang. Check network access to huggingface.co, "
                    "or pre-download the model.", name, timeout_s,
                )
            except Exception as exc:
                logger.warning("model warm_up: %s failed: %s", name, exc)

    _try("dense (nomic)", get_dense_model, 120)
    _try("sparse (bm25)", get_sparse_model, 60)
    _try("reranker (bge-reranker-large)", get_reranker, 60)


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
