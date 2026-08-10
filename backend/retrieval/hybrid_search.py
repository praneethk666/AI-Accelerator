import logging

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from backend.retrieval.keyword_index import KeywordIndex
from backend.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Cache the cross-encoder model to avoid reloading it on every query
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def _hybrid_local_rerank(query: str, cfg: dict, full_config: dict, filters: dict | None) -> list[dict]:
    # 1. Fetch dense candidates from VectorStore (Qdrant)
    candidate_k = cfg.get("candidate_k", 80)
    rerank_top_k = cfg.get("rerank_top_k", 5)
    
    from backend.core.models import get_dense_model
    from backend.retrieval.retrieval import _embed_query
    
    embedder = get_dense_model(full_config)
    q_emb = _embed_query(embedder, query, full_config)
    dense_hits = VectorStore.search(q_emb, full_config, top_k=candidate_k, filters=filters)
    
    # 2. Fetch sparse hits from Postgres Keyword Index
    sparse_hits = KeywordIndex.search(query, full_config, top_k=candidate_k, filters=filters)
    
    # Combine candidates uniquely
    seen = set()
    candidates = []
    for hit in dense_hits + sparse_hits:
        if hit["chunk_id"] not in seen:
            seen.add(hit["chunk_id"])
            candidates.append(hit)
            
    if not candidates:
        return []
        
    # 3. Apply rank-bm25 scoring on candidates
    tokenized_corpus = [ (c["text"] or "").lower().split(" ") for c in candidates ]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 4. Rerank with CrossEncoder
    cross_encoder = get_cross_encoder()
    pairs = [(query, c["text"] or "") for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    
    # Combine scores or just use CrossEncoder
    # We sort by CrossEncoder score since it's the most accurate
    ranked = sorted(zip(ce_scores, bm25_scores, candidates), key=lambda x: x[0], reverse=True)
    
    result = []
    for ce_score, bm_score, chunk in ranked[:rerank_top_k]:
        c = dict(chunk)
        # Use CrossEncoder score as the primary _score
        c["_score"] = float(ce_score)
        c["_bm25_score"] = float(bm_score)
        result.append(c)
        
    return result
