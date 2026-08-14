# Automated Pytest Test Suite

The **Tests Module** (`tests/`) houses the comprehensive automated test suite for the **AI-Accelerator** platform, spanning unit tests, pipeline graph integration, retrieval & ranking algorithms, safety guardrails, agent tool execution, and end-to-end ingestion pipelines.

---

## 1. Test Suite Catalog by Functional Subsystem

### 1. Core Architecture & Configuration
| Test File | Verification Scope |
|---|---|
| [`test_smoke.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_smoke.py) | Fast offline sanity check validating core imports, schemas, and default configurations without database dependencies. |
| [`test_schemas.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_schemas.py) | Validates `NormalizedBlock`, `Chunk`, `SourceRef`, and dictionary serialization contracts (`as_dicts`). |
| [`test_config.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_config.py) | Tests dynamic YAML loading, environment variable interpolation `${VAR}`, and database URL resolution. |
| [`test_graph.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_graph.py) | Exercises LangGraph `StateGraph` compilation, step ordering, route gating, and graceful error handling. |
| [`test_registry.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_registry.py) | Verifies defensive tool registration and uninstalled optional dependency handling. |

### 2. Extraction & Ingestion Pipelines
| Test File | Verification Scope |
|---|---|
| [`test_category_handler.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_category_handler.py) | Tests document type, industry, route mapping, grounding checks, and PDF kind classification. |
| [`test_docling_scanned.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_docling_scanned.py) | Exercises IBM Docling layout parsing, TableFormer recovery, and hybrid page routing. |
| [`test_excel_ppt_tools.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_excel_ppt_tools.py) | Validates spreadsheet bilingual headers, table parsing, and PowerPoint slide shape traversal. |
| [`test_word_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_word_tool.py) | Tests Word XML paragraph parsing, heading hierarchy, and embedded image extraction. |
| [`test_image_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_image_tool.py) | Tests standalone image format sniffing and vision captioning hand-off. |
| [`test_chunking.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_chunking.py) | Exercises sliding window token splits, Chonkie semantic boundaries, table row splitting, and dot-leader collapsing. |
| [`test_enrichment.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_enrichment.py) | Tests batch chunk summary and keyword tagging with offline TF-IDF fallback. |
| [`test_auto_ingestion.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_auto_ingestion.py) & [`test_gdrive_ingestion.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_gdrive_ingestion.py) | Validates directory watcher and Google Drive synchronization ingestion workflows. |
| [`test_ingest.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_ingest.py) | End-to-end ingestion pipeline integration test. |

### 3. Embeddings, Retrieval & Reranking
| Test File | Verification Scope |
|---|---|
| [`test_jina_embeddings.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_jina_embeddings.py) & [`test_openai_embeddings.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_openai_embeddings.py) | Validates cloud embedding client batching, normalization, and token usage accounting. |
| [`test_jina_reranker.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_jina_reranker.py) | Tests Jina Reranker v2 API client, multi-key rotation on 429 errors, and local BGE-reranker fallback. |
| [`test_retrieval.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_retrieval.py) & [`test_query.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_query.py) | Exercises hybrid dense+sparse BM25 search, Reciprocal Rank Fusion, and soft filter recovery. |
| [`test_tiered_procedure_search.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_tiered_procedure_search.py) & [`test_guided_procedure_multi_file.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_guided_procedure_multi_file.py) | Tests Tier 1 procedure extraction, multi-file procedure query support, and sequential step aggregations. |
| [`test_answerer_expand.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_answerer_expand.py) | Tests page context expansion for fragmented chunks and citation matching. |

### 4. Agent, Safety Guardrails & Persistence
| Test File | Verification Scope |
|---|---|
| [`test_agent_executor.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_agent_executor.py) | Tests LangGraph agent loop, tool dispatching, search short-circuiting, and human-in-the-loop write approvals. |
| [`test_context_manager.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_context_manager.py) | Tests sliding-window conversation memory caching. |
| [`test_guardrails.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_guardrails.py) | Exercises 3-stage safety guardrails: Input prompt injection checks, Indian PII redaction, chunk scanning, and output masking. |
| [`test_sql_read_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_sql_read_tool.py) & [`test_search_documents.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_search_documents.py) & [`test_get_page_context.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_get_page_context.py) | Validates individual agent tools. |
| [`test_storage.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_storage.py) & [`test_conversation_store.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/test_conversation_store.py) | Tests PostgreSQL relational writes, Qdrant vector index synchronization, and conversation persistence. |

---

## 2. Test Fixtures & Utilities

- [`fixtures.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/fixtures.py): Shared mock documents, configuration dictionaries, and state objects.
- [`stub_tools.py`](file:///d:/AI-Acc-updated/AI-Accelerator/tests/stub_tools.py): Deterministic mock tool stubs for fast unit testing.

---

## 3. Running the Test Suite

```powershell
# Run the entire test suite
pytest

# Run fast offline tests (skipping database dependencies)
pytest -m "not needs_db"

# Run specific test modules
pytest tests/test_smoke.py tests/test_agent_executor.py tests/test_guardrails.py tests/test_retrieval.py
```
