# AI-Accelerator — project memory for Claude Code

## Why this exists
A reusable **Document Intelligence + Enterprise RAG Accelerator**: ingest enterprise documents
(PDF, Excel, PPT, images, and engineering content like CAD / circuit diagrams) and answer
questions grounded in them, **with citations**. This is not a one-off chatbot — it's a
configurable engine meant to be reused per client and industry.

## What it does (the pipeline)
Upload → **categorize & route** (early, decides how to process) → **page_profile** (PDF per-page
x-ray) → **extract** → **visual enrichment** (vision model on significant visuals) → **chunk** →
**text/chunk enrichment** (topic/section/keyword tags) → **embed** → **store** → **retrieve** → **answer**.

- **Everything is a tool**; the pipeline is assembled from tools by **config** (a graph).
  Adding a capability = add a tool node + a config entry — no edits to other tools.
- **Hosting:** the data plane (files, extraction, OCR, embeddings, vectors, databases) is
  **self-hosted**; the **vision model and LLM are external APIs**. Keep what we send out
  minimal, downscaled, cached, and logged.

## Core design rules (do not violate)
- Tools never call each other — they read from and write to a shared `PipelineState`
  (`backend/core/tool.py`).
- Build to the shared contracts in `backend/core/schemas.py` (`NormalizedBlock`, `PageProfile`,
  `Chunk`). Do **not** fork them; changing a contract is a team decision.
- **Category = metadata/tags + route + prompt + retrieval filter. One schema — never a separate
  database per category.**
- Fail gracefully: append problems to `state["errors"]`; never crash the whole document.
- Vision enrichment is **gated** (significant visuals / scanned pages / low-OCR), not run on every page.
- **Agent vs LangGraph:** LangGraph is the runtime/skeleton. Use a **deterministic graph for
  ingestion**; use an **agent (an LLM-decision node inside the graph)** for query-time decisions
  and tool/connector selection (databases, inventory, comms).

## Layout
`backend/core` (shared contracts) · `pipeline` (graph) · `extraction` (pdf/excel/ppt/ocr) ·
`vision` · `categorize` · `chunking` · `enrichment` · `embeddings` · `storage` · `retrieval` ·
`api` · `evaluation` · `frontend` · `config` · `tests`.

## Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_smoke.py      # smoke test — should print "smoke test passed"
# docker compose up             # local stack (fill in services first)
```

## Conventions
Coding standards live in `CONTRIBUTING.md` (black, ruff, type hints, naming, branch/PR rules).
Secrets go in a local `.env` (gitignored) — **never commit keys; this repo is public.**
Branch as `feat/<name>-<task>` and PR into `main`.

## More context
Full plan and the team brief are in `docs/`. Six workstreams: extraction & vision · RAG &
evaluation · frontend & observability · platform & orchestration · document categorization ·
Excel/PPT extraction. The upload/test UI is currently unassigned.
