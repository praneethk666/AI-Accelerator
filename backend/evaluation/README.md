# RAG Evaluation & Benchmarking Subsystem

The **Evaluation Module** (`backend/evaluation/`) provides testing, validation, and benchmarking frameworks to evaluate RAG answer quality, retrieval precision, and agent reasoning.

---

## 1. Key Capabilities & Features

- **RAG Triad Metrics**:
  - **Context Relevance**: Measures how relevant retrieved context chunks are to the user question.
  - **Groundedness / Faithfulness**: Measures whether claims in the generated response are factually supported by source chunks without hallucination.
  - **Answer Relevance**: Measures whether the synthesized answer directly answers the original prompt.
- **Automated Synthetic Dataset Generation**:
  - Automatically generates question-context-answer ground truth pairs from ingested document corpora.
- **Regression Benchmarking Harness**:
  - Evaluates retrieval precision (Hit Rate @ K, MRR @ K, NDCG @ K) and latency across embedding models and reranker providers.

---

## 2. Dependencies & Testing

- **backend.core.llm_client**: LLM judge completions.
- **Verification**:
  ```powershell
  pytest tests/test_smoke.py
  ```
