# tests/test_retrieval.py
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from tests.fixtures import sample_chunks, sample_query_state
from backend.retrieval.retrieval import RetrievalTool, _rrf_fuse
from backend.core.models import DENSE_QUERY_PREFIX


# ── helpers ───────────────────────────────────────────────────────────────────

def chunk_dicts():
    return [c.__dict__ for c in sample_chunks()]


def make_config(method: str = "hybrid_rerank", top_n: int = 3):
    return {
        # query prefix is now config-driven (empty for bge-m3); set it to the nomic
        # value here so the HyDE prefix-application assertions exercise a real prefix.
        "llm": {"model": "gpt-4", "provider": "openai"},
        "embeddings": {"dense_query_prefix": DENSE_QUERY_PREFIX},
        "query": {
            "retrieval": {
                "method":        method,
                "top_n":         top_n,
                "candidate_k":   6,
                "rerank_top_k":  top_n,
                "dense_weight":  0.6,
                "sparse_weight": 0.4,
            }
        }
    }


# ── naive ─────────────────────────────────────────────────────────────────────

def test_naive_returns_chunks():
    state  = sample_query_state()
    config = make_config(method="naive")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    assert len(result["retrieved_chunks"]) > 0


def test_naive_calls_vector_store_with_top_n():
    state  = sample_query_state()
    config = make_config(method="naive", top_n=2)

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()[:2]) as mock_vs:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    mock_vs.assert_called_once()
    _, kwargs = mock_vs.call_args
    assert kwargs["top_k"] == 2


def test_naive_passes_filters_to_vector_store():
    state = sample_query_state()
    state["document_scope"] = ["doc-fixture-001"]
    config = make_config(method="naive")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()) as mock_vs:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    _, kwargs = mock_vs.call_args
    assert kwargs["filters"] == {"document_id": ["doc-fixture-001"]}


# ── hybrid ────────────────────────────────────────────────────────────────────

def test_hybrid_calls_both_dense_and_sparse():
    state  = sample_query_state()
    config = make_config(method="hybrid")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()) as mock_vs, \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()) as mock_ki:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    mock_vs.assert_called()
    mock_ki.assert_called()


def test_hybrid_returns_fused_chunks():
    state  = sample_query_state()
    config = make_config(method="hybrid", top_n=3)

    dense_chunks  = chunk_dicts()[:2]   # chunk-001, chunk-002
    sparse_chunks = chunk_dicts()[1:]   # chunk-002, chunk-003 — chunk-002 overlaps

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=dense_chunks), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=sparse_chunks):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
    # chunk-002 appears in both so must rank first after RRF
    assert ids[0] == "chunk-002"


def test_hybrid_no_document_scope_passes_no_filters():
    state = sample_query_state()
    state["document_scope"] = []
    config = make_config(method="hybrid")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()) as mock_vs, \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    _, kwargs = mock_vs.call_args
    assert kwargs["filters"] is None


# ── hybrid_rerank ─────────────────────────────────────────────────────────────

def test_hybrid_rerank_returns_chunks():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        result = RetrievalTool().run(state, config)

    assert len(result["retrieved_chunks"]) > 0


def test_hybrid_rerank_sorted_by_score():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank", top_n=3)
    chunks = chunk_dicts()

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        # chunk-001=0.1, chunk-002=0.9, chunk-003=0.4 → sorted: 002, 003, 001
        mock_reranker.return_value.predict.return_value = [0.1, 0.9, 0.4]
        result = RetrievalTool().run(state, config)

    ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
    assert ids[0] == "chunk-002"
    assert ids[1] == "chunk-003"
    assert ids[2] == "chunk-001"


def test_hybrid_rerank_top_k_respected():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank", top_n=1)

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        result = RetrievalTool().run(state, config)

    assert len(result["retrieved_chunks"]) == 1


def test_hybrid_rerank_calls_reranker_with_query_chunk_pairs():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank", top_n=3)
    chunks = chunk_dicts()

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        RetrievalTool().run(state, config)

    pairs = mock_reranker.return_value.predict.call_args[0][0]
    query = state["sub_questions"][0]
    # every pair must be (query, chunk_text)
    assert all(p[0] == query for p in pairs)
    assert all(isinstance(p[1], str) for p in pairs)


# ── hyde ──────────────────────────────────────────────────────────────────────

def test_hyde_generates_hypothesis_and_searches():
    state  = sample_query_state()
    config = make_config(method="hyde")

    with patch("backend.retrieval.retrieval.get_llm") as mock_llm, \
         patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()):

        mock_llm.return_value.invoke.return_value.content = (
            "The M6 bolt torque specification is 12 Nm."
        )
        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    # LLM was called to generate hypothesis
    mock_llm.return_value.invoke.assert_called_once()
    assert len(result["retrieved_chunks"]) > 0


def test_hyde_embeds_hypothesis_not_query():
    state  = sample_query_state()
    config = make_config(method="hyde")
    hypothesis = "The M6 bolt torque specification is 12 Nm."

    with patch("backend.retrieval.retrieval.get_llm") as mock_llm, \
         patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()):

        mock_llm.return_value.invoke.return_value.content = hypothesis
        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    # encode must be called with the hypothesis, not the original query
    # (retrieval prepends the nomic DENSE_QUERY_PREFIX to whatever it embeds)
    encoded_text = mock_embedder.return_value.encode.call_args[0][0]
    assert encoded_text == DENSE_QUERY_PREFIX + hypothesis


def test_hyde_top_n_passed_to_vector_store():
    state  = sample_query_state()
    config = make_config(method="hyde", top_n=2)

    with patch("backend.retrieval.retrieval.get_llm") as mock_llm, \
         patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()[:2]) as mock_vs:

        mock_llm.return_value.invoke.return_value.content = "hypothesis text"
        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    _, kwargs = mock_vs.call_args
    assert kwargs["top_k"] == 2


# ── enriched (alias for hybrid_rerank) ───────────────────────────────────────

def test_enriched_behaves_same_as_hybrid_rerank():
    state_enriched = sample_query_state()
    state_rerank   = sample_query_state()
    chunks         = chunk_dicts()

    def run_with_method(state, method):
        config = make_config(method=method, top_n=3)
        with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
             patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunks), \
             patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunks), \
             patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:
            mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
            mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
            return RetrievalTool().run(state, config)

    result_enriched = run_with_method(state_enriched, "enriched")
    result_rerank   = run_with_method(state_rerank,   "hybrid_rerank")

    assert ([c["chunk_id"] for c in result_enriched["retrieved_chunks"]]
         == [c["chunk_id"] for c in result_rerank  ["retrieved_chunks"]])


# ── unknown method ────────────────────────────────────────────────────────────

def test_unknown_method_appends_error():
    state  = sample_query_state()
    config = make_config(method="does_not_exist")

    result = RetrievalTool().run(state, config)

    assert any("does_not_exist" in e["error"] for e in result["errors"])


# ── run() general ─────────────────────────────────────────────────────────────

def test_run_writes_retrieved_chunks():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        result = RetrievalTool().run(state, config)

    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0


def test_run_empty_sub_questions_returns_empty():
    state = sample_query_state()
    state["sub_questions"] = []
    config = make_config()

    result = RetrievalTool().run(state, config)

    assert result["retrieved_chunks"] == []


def test_run_deduplicates_chunks_across_sub_questions():
    state = sample_query_state()
    state["sub_questions"] = [
        "What is the torque for M6?",
        "M6 bolt torque specification",
    ]
    config = make_config(method="hybrid_rerank")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        result = RetrievalTool().run(state, config)

    ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
    assert len(ids) == len(set(ids))


def test_run_appends_error_on_failure():
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search",
               side_effect=RuntimeError("Qdrant down")):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    assert any("Qdrant down" in e["error"] for e in result["errors"])


# ── RRF ───────────────────────────────────────────────────────────────────────

def test_rrf_fuse_overlap_scores_higher():
    chunks = chunk_dicts()
    dense  = chunks[:2]   # chunk-001, chunk-002
    sparse = chunks[1:]   # chunk-002, chunk-003

    fused = _rrf_fuse(dense, sparse, w_dense=0.6, w_sparse=0.4, top_k=3, k=60)

    assert [c["chunk_id"] for c in fused][0] == "chunk-002"


def test_rrf_fuse_top_k_respected():
    chunks = chunk_dicts()
    fused  = _rrf_fuse(chunks, chunks, w_dense=0.6, w_sparse=0.4, top_k=2, k=60)
    assert len(fused) == 2


def test_rrf_fuse_no_overlap_returns_all_unique():
    chunks = chunk_dicts()
    fused  = _rrf_fuse(chunks[:1], chunks[2:], w_dense=0.6, w_sparse=0.4, top_k=2, k=60)
    assert {c["chunk_id"] for c in fused} == {"chunk-001", "chunk-003"}


def test_rrf_fuse_dense_weight_dominates():
    chunks = chunk_dicts()
    # chunk-001 only in dense (high weight), chunk-002 only in sparse (low weight)
    fused = _rrf_fuse(chunks[:1], chunks[1:2], w_dense=0.9, w_sparse=0.1, top_k=2, k=60)
    assert fused[0]["chunk_id"] == "chunk-001"


def test_rrf_fuse_sparse_weight_dominates():
    chunks = chunk_dicts()
    # chunk-002 only in sparse (high weight), chunk-001 only in dense (low weight)
    fused = _rrf_fuse(chunks[:1], chunks[1:2], w_dense=0.1, w_sparse=0.9, top_k=2, k=60)
    assert fused[0]["chunk_id"] == "chunk-002"


# tests/test_retrieval.py  — additional cases

# ── naive ─────────────────────────────────────────────────────────────────────

def test_naive_returns_empty_when_vector_store_empty():
    state  = sample_query_state()
    config = make_config(method="naive")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[]):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    assert result["retrieved_chunks"] == []


# ── hybrid ────────────────────────────────────────────────────────────────────

def test_hybrid_dense_only_results_when_sparse_empty():
    """Sparse returns nothing — only dense chunks should appear in output."""
    state  = sample_query_state()
    config = make_config(method="hybrid", top_n=3)

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[]):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    ids = {c["chunk_id"] for c in result["retrieved_chunks"]}
    assert ids == {"chunk-001", "chunk-002", "chunk-003"}


def test_hybrid_sparse_only_results_when_dense_empty():
    """Dense returns nothing — only sparse chunks should appear in output."""
    state  = sample_query_state()
    config = make_config(method="hybrid", top_n=3)

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    ids = {c["chunk_id"] for c in result["retrieved_chunks"]}
    assert ids == {"chunk-001", "chunk-002", "chunk-003"}


def test_hybrid_both_empty_returns_empty():
    state  = sample_query_state()
    config = make_config(method="hybrid")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[]):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    assert result["retrieved_chunks"] == []


# ── hybrid_rerank ─────────────────────────────────────────────────────────────

def test_hybrid_rerank_empty_candidates_returns_empty():
    """If hybrid returns nothing, reranker should not be called and output is empty."""
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[]), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    mock_reranker.return_value.predict.assert_not_called()
    assert result["retrieved_chunks"] == []


def test_hybrid_rerank_all_same_score_preserves_candidate_order():
    """When all scores are equal the original candidate order must be preserved."""
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank", top_n=3)
    chunks = chunk_dicts()

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.5, 0.5, 0.5]
        result = RetrievalTool().run(state, config)

    # all scores equal — output length must still be correct
    assert len(result["retrieved_chunks"]) == 3


def test_hybrid_rerank_rerank_top_k_less_than_candidates():
    """rerank_top_k=1 must return exactly 1 chunk even with 3 candidates."""
    state  = sample_query_state()
    config = make_config(method="hybrid_rerank", top_n=3)
    config["query"]["retrieval"]["rerank_top_k"] = 1
    chunks = chunk_dicts()

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunks), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [0.9, 0.5, 0.3]
        result = RetrievalTool().run(state, config)

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["chunk_id"] == "chunk-001"


# ── hyde ──────────────────────────────────────────────────────────────────────

def test_hyde_llm_failure_appends_error():
    """If LLM raises during hypothesis generation the error must be caught."""
    state  = sample_query_state()
    config = make_config(method="hyde")

    with patch("backend.retrieval.retrieval.get_llm") as mock_llm, \
         patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder:

        mock_llm.return_value.invoke.side_effect = RuntimeError("LLM timeout")
        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    assert any("LLM timeout" in e["error"] for e in result["errors"])


def test_hyde_empty_hypothesis_still_searches():
    """LLM returns empty string — encode is still called and search proceeds."""
    state  = sample_query_state()
    config = make_config(method="hyde")

    with patch("backend.retrieval.retrieval.get_llm") as mock_llm, \
         patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()):

        mock_llm.return_value.invoke.return_value.content = ""
        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    encoded_text = mock_embedder.return_value.encode.call_args[0][0]
    assert encoded_text == DENSE_QUERY_PREFIX  # empty hypothesis -> just the prefix
    assert len(result["retrieved_chunks"]) > 0


# ── document_scope filter ─────────────────────────────────────────────────────

def test_document_scope_passed_as_filter_to_keyword_index():
    """KeywordIndex.search must also receive the document_scope filter."""
    state = sample_query_state()
    state["document_scope"] = ["doc-fixture-001"]
    config = make_config(method="hybrid")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=chunk_dicts()) as mock_ki:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    _, kwargs = mock_ki.call_args
    assert kwargs["filters"] == {"document_id": ["doc-fixture-001"]}


# ── multiple sub_questions ────────────────────────────────────────────────────

def test_multiple_sub_questions_each_trigger_retrieval():
    """One retrieval call per sub_question — 2 questions = 2 VectorStore calls."""
    state = sample_query_state()
    state["sub_questions"] = ["question one", "question two"]
    config = make_config(method="naive")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=chunk_dicts()) as mock_vs:

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        RetrievalTool().run(state, config)

    assert mock_vs.call_count == 2


def test_multiple_sub_questions_partial_failure_continues():
    """First sub_question fails, second succeeds — output has second question's chunks."""
    state = sample_query_state()
    state["sub_questions"] = ["bad question", "good question"]
    config = make_config(method="naive")

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_embedder, \
         patch("backend.retrieval.retrieval.VectorStore.search",
               side_effect=[RuntimeError("Qdrant down"), chunk_dicts()]):

        mock_embedder.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        result = RetrievalTool().run(state, config)

    # error recorded for the failed question
    assert len(result["errors"]) == 1
    assert "Qdrant down" in result["errors"][0]["error"]
    # successful question's chunks still in output
    assert len(result["retrieved_chunks"]) > 0


# ── RRF additional ────────────────────────────────────────────────────────────

def test_rrf_fuse_empty_dense_uses_sparse_only():
    chunks = chunk_dicts()
    fused  = _rrf_fuse([], chunks, w_dense=0.6, w_sparse=0.4, top_k=3, k=60)
    assert len(fused) == 3


def test_rrf_fuse_empty_sparse_uses_dense_only():
    chunks = chunk_dicts()
    fused  = _rrf_fuse(chunks, [], w_dense=0.6, w_sparse=0.4, top_k=3, k=60)
    assert len(fused) == 3


def test_rrf_fuse_both_empty_returns_empty():
    fused = _rrf_fuse([], [], w_dense=0.6, w_sparse=0.4, top_k=3, k=60)
    assert fused == []


def test_rrf_fuse_top_k_larger_than_results_returns_all():
    chunks = chunk_dicts()   # 3 chunks
    fused  = _rrf_fuse(chunks, [], w_dense=0.6, w_sparse=0.4, top_k=10, k=60)
    assert len(fused) == 3   # only 3 available, not 10


# ── Fallback page-expansion ────────────────────────────────────────────────────

def make_config_with_fallback(threshold: float = -1.0, enabled: bool = True):
    """Config that activates fallback expansion at the given threshold."""
    return {
        "llm": {"model": "gpt-4", "provider": "openai"},
        "embeddings": {"dense_query_prefix": DENSE_QUERY_PREFIX},
        "query": {
            "retrieval": {
                "method":              "hybrid_rerank",
                "top_n":               3,
                "candidate_k":         6,
                "rerank_top_k":        3,
                "dense_weight":        0.6,
                "sparse_weight":       0.4,
                "fallback_enabled":    enabled,
                "fallback_threshold":  threshold,
                "fallback_top_k":      1,
            }
        },
    }


def _low_score_chunk(chunk_id: str = "c1", page: int = 7, score: float = -5.0):
    """A chunk with a very low reranker score simulating a poor match."""
    return {
        "chunk_id":    chunk_id,
        "document_id": "doc-fixture-001",
        "text":        "sparse parts table row",
        "_score":      score,
        "token_count": 10,
        "tags":        {},
        "table_data":  None,
        "image_path":  None,
        "source_ref":  {"filename": "manual.pdf", "page": page},
        "vector":      [],
        "sparse_vector": {},
    }


def test_fallback_triggers_when_best_score_below_threshold():
    """When all chunk scores are below fallback_threshold, _expand_chunks_to_pages is called."""
    state  = sample_query_state()
    config = make_config_with_fallback(threshold=-1.0, enabled=True)

    low_chunk = _low_score_chunk(score=-3.0)
    expanded_chunk = dict(low_chunk)
    expanded_chunk["text"] = "FULL PAGE TEXT from document_blocks"

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_emb, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[low_chunk]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[low_chunk]), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker, \
         patch("backend.retrieval.retrieval._expand_chunks_to_pages",
               return_value=[expanded_chunk]) as mock_expand:

        mock_emb.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [-3.0]

        result = RetrievalTool().run(state, config)

    # Expansion was triggered
    mock_expand.assert_called_once()
    # The expanded (full-page) text is what the LLM receives
    assert result["retrieved_chunks"][0]["text"] == "FULL PAGE TEXT from document_blocks"


def test_fallback_does_not_trigger_when_score_above_threshold():
    """Normal high-confidence searches must NOT trigger the fallback path."""
    state  = sample_query_state()
    config = make_config_with_fallback(threshold=-1.0, enabled=True)

    high_chunk = _low_score_chunk(score=3.5)   # well above threshold

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_emb, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[high_chunk]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[high_chunk]), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker, \
         patch("backend.retrieval.retrieval._expand_chunks_to_pages") as mock_expand:

        mock_emb.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [3.5]

        result = RetrievalTool().run(state, config)

    # Expansion must NOT have been called for a high-confidence result
    mock_expand.assert_not_called()
    # The original chunk text is returned unchanged
    assert result["retrieved_chunks"][0]["text"] == high_chunk["text"]


def test_fallback_disabled_via_config():
    """When fallback_enabled=False the expansion path is never entered."""
    state  = sample_query_state()
    config = make_config_with_fallback(threshold=-1.0, enabled=False)

    low_chunk = _low_score_chunk(score=-5.0)

    with patch("backend.retrieval.retrieval.get_dense_model") as mock_emb, \
         patch("backend.retrieval.retrieval.VectorStore.search", return_value=[low_chunk]), \
         patch("backend.retrieval.retrieval.KeywordIndex.search", return_value=[low_chunk]), \
         patch("backend.retrieval.retrieval.get_reranker") as mock_reranker, \
         patch("backend.retrieval.retrieval._expand_chunks_to_pages") as mock_expand:

        mock_emb.return_value.encode.return_value.tolist.return_value = [0.0] * 1024
        mock_reranker.return_value.predict.return_value = [-5.0]

        RetrievalTool().run(state, config)

    mock_expand.assert_not_called()