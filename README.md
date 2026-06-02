# AI-Accelerator

Document Intelligence + Enterprise RAG Accelerator.

Ingest documents (PDF, Excel, PPT, images) -> extract -> enrich -> chunk -> embed
-> store -> answer questions with citations. **Everything is a tool; the pipeline
is assembled from tools via a config-driven graph.** Self-hosted data plane;
vision + LLM are external APIs.

See the Intern Brief and the Phase 1 Plan (shared separately) for full context.
Coding standards: `CONTRIBUTING.md`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_smoke.py      # should print: smoke test passed
```

## Layout
```
backend/core        shared contracts: schemas, Tool interface, config  (do not fork)
backend/pipeline    graph wiring (LangGraph)
backend/extraction  pdf / excel / ppt / ocr handlers
backend/vision      visual enrichment (vision API)
backend/categorize  document categorization & routing
backend/chunking    splitting
backend/enrichment  text/chunk tagging
backend/embeddings  embedding model
backend/storage     relational + vector + object storage (ingestion)
backend/retrieval   retrieval + answering
backend/api         HTTP endpoints
backend/evaluation  golden set + metrics
frontend            upload + test UI
config              pipeline profiles
tests               tests
```
