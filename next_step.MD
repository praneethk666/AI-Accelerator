# Task — Karthii — Ingestion-as-a-Tool
**Date:** 2026-07-01 · **Target:** Fri 03-Jul · **Priority:** High

## Message
We're growing the accelerator from a Q&A system into a set of tools that agents can
use. Your task is the foundation for that. Right now our document ingestion works, but
it's tied into the API. I want it exposed as one clean tool: an agent hands it a file,
it runs the full pipeline, and returns the document id, status, and details.

Build a single `ingest_document(file)` entry point. Reuse what we already have —
`scripts/run_ingest.py` is the closest example — don't rebuild extraction. Make it so
running the same file twice updates instead of duplicating, keep the database status,
and register it so an agent can call it.

This is the most important task this week and I'm counting on you for it. Keep it clean
and simple. Don't touch the extraction tools or the schemas. If any part of the design
is unclear, talk to me before building.

## Definition of Done
- `ingest_document(file_path, document_id=None)` runs the existing ingestion graph and
  returns `{document_id, status, metrics, errors}`.
- Idempotent — same file/id updates, doesn't duplicate.
- DB-backed status (reuse existing ingestion status / PostgresStore).
- Registered so an agent can call it (Tool with a name + clear signature).
- Smoke-tested on a PDF, an XLSX, and a PPTX.

## Reference
`scripts/run_ingest.py`, `backend/api/main.py` (how /upload ingests), `backend/pipeline/graph.py`, `default_registry.py`.
**Don't touch:** extraction tools, `core/schemas.py`, graph internals.

## Next (after this lands)
Agent-executor core — the loop that decides which tools to call. 