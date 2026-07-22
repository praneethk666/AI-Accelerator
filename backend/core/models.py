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

# Dense instruction prefixes are MODEL-SPECIFIC and live in config, not code:
#   nomic-embed-text-v1.5 REQUIRES different doc/query prefixes (it is instruction-tuned);
#   bge-m3 (the current default) takes NO prefix and a wrong one silently degrades recall.
# embed_tool (indexing) uses get_dense_document_prefix(); retrieval uses get_dense_query_prefix().
# The constants below are the nomic defaults, kept only for reference/back-compat.
DENSE_DOCUMENT_PREFIX = "search_document: "
DENSE_QUERY_PREFIX = "search_query: "

DEFAULT_DENSE_MODEL = "BAAI/bge-m3"


def get_dense_document_prefix(config: dict) -> str:
    """Instruction prefix prepended to each DOCUMENT before dense encoding.
    Config-driven so it swaps with the model. Default "" (bge-m3 / prefix-free models)."""
    return (config.get("embeddings") or {}).get("dense_document_prefix", "") or ""


def get_dense_query_prefix(config: dict) -> str:
    """Instruction prefix prepended to each QUERY before dense encoding.
    Must differ from the document prefix for instruction-tuned models (nomic).
    Default "" (bge-m3 / prefix-free models)."""
    return (config.get("embeddings") or {}).get("dense_query_prefix", "") or ""

import threading

_dense_model = None
_sparse_model = None
_reranker = None
# Guards singleton construction so concurrent ingests (FastAPI background tasks in the
# threadpool) can't both observe None and double-load a ~2.3GB model when warm_up was
# skipped/timed out. Double-checked locking; loads happen once.
_model_lock = threading.Lock()


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

    _try("dense embedder", get_dense_model, 300)   # bge-m3 first-run download is ~2.3GB
    _try("sparse (bm25)", get_sparse_model, 60)
    _try("reranker", get_reranker, 120)


def get_dense_model(config: dict):
    """Return the shared SentenceTransformer dense embedder (default BAAI/bge-m3, 1024-dim).

    trust_remote_code=True is kept so instruction-tuned models distributed with
    custom code (e.g. nomic) still load if swapped in via config. See
    get_dense_document_prefix / get_dense_query_prefix for the model-specific
    instruction prefixes (empty for bge-m3).
    """
    global _dense_model
    if _dense_model is None:
        with _model_lock:
            if _dense_model is None:
                from sentence_transformers import SentenceTransformer
                model_name = config["embeddings"]["dense_model"]
                _dense_model = SentenceTransformer(model_name, trust_remote_code=True)
    return _dense_model


def get_sparse_model(config: dict):
    """Return the shared fastembed BM25 sparse encoder."""
    global _sparse_model
    if _sparse_model is None:
        with _model_lock:
            if _sparse_model is None:
                from fastembed import SparseTextEmbedding
                model_name = config["embeddings"]["sparse_model"]
                _sparse_model = SparseTextEmbedding(model_name)
    return _sparse_model


def get_reranker(config: dict):
    """Return the shared CrossEncoder reranker (default BAAI/bge-reranker-v2-m3, multilingual)."""
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                model_name = config["embeddings"]["reranker_model"]
                _reranker = CrossEncoder(model_name)
    return _reranker
