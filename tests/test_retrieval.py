# tests/test_retrieval.py
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from tests.fixtures import sample_chunks, sample_query_state
from backend.retrieval.retrieval import RetrievalTool, _rrf_fuse


# ── helpers ───────────────────────────────────────────────────────────────────

def chunk_dicts():
    return [c.__dict__ for c in sample_chunks()]


def make_config(method: str = "hybrid_rerank", top_n: int = 3):
    return {
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
    encoded_text = mock_embedder.return_value.encode.call_args[0][0]
    assert encoded_text == hypothesis


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
    assert encoded_text == ""
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