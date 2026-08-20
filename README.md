# AI-Accelerator

A config-driven Document Intelligence + RAG platform, extending toward an agentic
query layer. Ingest documents (PDF, Excel, PPT, Word, images) → extract → enrich
→ chunk → embed → index → ask questions and get cited answers. An agent can also
pick between ingesting a file, searching the corpus, or querying a database,
instead of always doing the same fixed thing.

**Everything is a tool.** Both the ingestion pipeline and the query pipeline are
assembled at runtime from small, single-purpose tools via one config-driven
LangGraph engine — see [Architecture](#architecture) below. Self-hosted data
plane (Postgres + Qdrant); LLM/vision are external APIs, swappable via config.

Setup/run steps: [RUNNING.md](RUNNING.md). Coding standards: [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_smoke.py      # should print: smoke test passed
```

Full local setup (Docker data plane, API, frontend, keys) is in [RUNNING.md](RUNNING.md).

## Layout

```
backend/core        shared contracts: schemas, Tool protocol, config loader, LLM/vision clients (do not fork)
backend/pipeline     graph engine (LangGraph) + ingest/query entry points + tool registry
backend/categorize   document type/industry/route classification (uses vision)
backend/extraction   pdf / excel / ppt / word / image / cad extractors  (see backend/extraction/README.md)
backend/vision       figure/diagram captioning (vision API)
backend/chunking     NormalizedBlock[] -> Chunk[]
backend/enrichment   chunk tagging (industry/doc_type/topic/keywords)
backend/embeddings   dense + sparse embedding
backend/storage      Postgres (relational) + Qdrant (vector) + object store
backend/retrieval    hybrid retrieval + reranking + grounded answering
backend/agent        agent-executor: picks + calls agent tools (see Agent section)
backend/agent_tools.py   registry of agent-callable tools (ingest/search/sql)
backend/connectors   agent-callable connectors to external systems (SQL today)
backend/api          FastAPI HTTP layer behind the React UI
frontend             upload + chat UI
config               pipeline profiles (global.yaml is the real one; pipeline.example.yaml shows the shape)
scripts              operational CLIs (ingest, agent chat, DB reset) + scripts/dev (one-off R&D tools)
tests                pytest suite
```

---

## Architecture

Two things are true about this codebase and everything else follows from them:

1. **A tool does one job.** It reads what it needs from a shared state dict and
   writes its result back. Tools never call each other directly — see
   `backend/core/tool.py` for the `Tool` protocol (`name` + `run(state, config)`)
   and `PipelineState` (the shared dict's typed shape).
2. **The pipeline is assembled from config, not code.** `config/global.yaml`'s
   `steps:` list decides which tools run and in what order; a tool not listed
   simply doesn't run. Turning a feature on/off, or building a different pipeline
   for a different document route, is a config edit — see `backend/pipeline/graph.py`.

### The config system

`backend/core/config.py::load_config(path)` reads a YAML file, resolves `${VAR}`
placeholders from the environment (missing vars are left as the literal
`${VAR}` string), and returns a plain dict — no schema, no validation layer.
`PipelineConfig.from_dict(...)` wraps that dict with three things a tool actually
needs:

- `.route` — the active route (e.g. `text_default`, `diagram_heavy`, `cad_route`).
- `.steps` — the ordered list of tool names to run.
- `.section("vision")` — a tool's own settings block, `{}` if absent (so a tool
  never needs to guard for a missing key).

A tool reads its own section straight from the full config dict passed into
`run(state, config)` — e.g. `config["vision"]["dpi"]`. There's no central schema
tying config keys to tool code, by design: adding a new setting for your tool
never requires touching `backend/core`.

**Registration** (`backend/core/registry.py` + `backend/pipeline/default_registry.py`):
every real tool is registered under its `.name` in `_TOOL_SPECS`
(`default_registry.py`). Import is defensive — a tool whose optional dependency
is missing is logged and skipped, not fatal.

**The graph** (`backend/pipeline/graph.py::build_pipeline`) turns `config.steps`
into a LangGraph `StateGraph`: one node per step, wired in order, with two kinds
of branching handled automatically:
- **route gates** (`route_gates: {step: [routes that keep it]}`) — a step only
  runs on the routes listed; e.g. `vision_enrichment` only runs on
  `diagram_heavy`/`image_route`, never `text_default`, because it's expensive.
- **the `extract` placeholder** — every route lists a generic `extract` step;
  the graph expands it into whichever concrete extractor actually matches the
  file (`pdf_extractors` by `pdf_kind`, `route_extractors` by route override, or
  `extractors` by `file_type`) and runs exactly one.

A tool failing never kills the run — the graph node wrapper catches the
exception, appends it to `state["errors"]`, and the pipeline continues.

### Two profiles, one engine

`config/global.yaml` has two step lists that both run through the exact same
graph engine:
- `ingestion.steps` — `categorize → extract → vision_enrichment → chunk →
  enrich_chunks → embed → index` (route-dependent; not every step runs on every
  route).
- `query.steps` — `query_planner → retrieval → answerer`.

`backend/pipeline/ingest.py::ingest_document()` and
`backend/pipeline/query.py::run_query()` are the two entry points that build a
`PipelineConfig` for their profile and call `run_pipeline()`. Both are also
exposed as **agent tools** (see below), so an agent calls the exact same code
path a human upload/chat action does — nothing is reimplemented for the agent.

### Adding a tool

1. Write a class with `.name` and `.run(state, config)` (pipeline tool) or
   `.name` / `.description` / `.input_schema` / `.run(**kwargs)` (agent tool —
   see `backend/agent_tools.py`'s `AgentTool` protocol).
2. Register it (`default_registry.py` for a pipeline tool, `agent_tools.py` for
   an agent tool).
3. For a pipeline tool: add its name to `config/global.yaml`'s relevant
   `steps:` list (and a `route_gates` entry if it shouldn't run everywhere).

No other file changes — that's the whole integration contract.

---

## Ingestion flow

`upload → categorize → extract → vision_enrichment → chunk → enrich_chunks →
embed → index`

1. **categorize** (`backend/categorize/`) — reads the first page(s), calls
   vision once to classify `{document_type, industry, route, confidence}` (falls
   back to a keyword heuristic if vision is unavailable). For PDFs, also runs
   `detector.py` to classify digital/scanned/mixed (`state["pdf_kind"]`), which
   picks the PDF extractor.
2. **extract** (placeholder, see above) — one of:
   - `DoclingPDFTool` (`extraction/docling_pdf/`) — the only PDF extractor; one
     hybrid tool handles digital, scanned, and mixed PDFs (per-page routing
     between native text, Docling's layout+table model, and a VLM rescue for
     garbled/complex pages — see `extraction/page_router.py` and
     `extraction/vision_ocr.py`).
   - `ExcelExtractorTool` / `PPTExtractorTool` / `WordExtractorTool` — native
     parsing (openpyxl/pandas, python-pptx, python-docx) with per-block language
     detection; embedded images become `pending_vision` blocks for the next step.
   - `ImageExtractorTool` — a standalone image file becomes one `pending_vision` block.
   - `CADExtractionTool` (route override, `cad_route`/`circuit_route`) — CAD/circuit
     sheets are mostly image, not text; renders the page, tiles large sheets so
     reference designators survive downsampling, and asks a VLM for structured
     regions (title block, tables, notes).
3. **vision_enrichment** (route-gated, see [Vision](#vision) below) — captions
   significant figures/diagrams and resolves any `pending_vision` blocks from
   step 2 into real `image_caption` blocks.
4. **chunk** — `NormalizedBlock[] → Chunk[]`: sliding token-window split for
   text; tables and image captions are atomic (never split, so a citation always
   points at a complete table/figure).
5. **enrich_chunks** — LLM-tags each chunk (`industry`, `doc_type`, `section`,
   `summary`, `keywords`), batched to keep call count down.
6. **embed** — dense (nomic-embed-text-v1.5, 768-dim) + sparse (BM25) vectors per chunk.
7. **index** — writes to Postgres (relational: documents, chunks, tags, status)
   and Qdrant (vectors, named dense + sparse).

`ingest_document()` (`backend/pipeline/ingest.py`) wraps all of this as one
idempotent call: same file content (or an explicit `document_id`) updates in
place instead of duplicating.

### Models & limits, by step

Every model is config-driven and swappable — this is the shipped default, not
a hard dependency.

| Step | Model (default config) | Where it runs | Limit that matters |
|---|---|---|---|
| categorize (vision) | provider-dependent, `vision:` block | External API | 1 call/doc — cheap regardless of doc size |
| extract (docling) | Docling layout + TableFormer (local) | CPU, in-process | No external limit; CPU-bound, scales with page count |
| extract (VLM rescue) | `vision_ocr:` block | External API | Only pages flagged complex (or all, `mode: always`) |
| vision_enrichment | `vision:` block | External API | 1 call per significant figure — scales with figure count, not page count |
| enrich_chunks | `llm:` block, batched | External API | `enrichment.batch_size` chunks/call (default 5) — **sequential**, not concurrent (see Scale below) |
| embed (dense) | nomic-embed-text-v1.5 (local) | CPU/GPU, in-process | No external limit; one batch call for all chunks |
| embed (sparse) | Qdrant/bm25 (local) | CPU, in-process | No external limit |
| retrieval (rerank) | BAAI/bge-reranker-large (local) | CPU/GPU, in-process | No external limit, but ~1.3GB — see the warm-up note below |
| answerer | `llm.answer_model` | External API | `query.max_context_tokens` (default 3000) caps what's fed in |
| agent executor | `query.agent.model` (Groq default) | External API | Needs **native tool-calling** support — not every model/provider has this |

**Free-tier caps to know about** (from `config/global.yaml` comments): Groq
~100k tokens/day on the free tier; Google AI Studio's free tier is rate-limited
per-minute (not a hard daily wall, but bursts of vision calls will 429 — the
vision client retries with backoff). None of these are enforced by the
accelerator itself; they're the provider's limits, and swapping to a paid key
(`OPENAI_API_KEY`) removes them.

**Local models download on first use** (HuggingFace Hub) and are cached
(`~/.cache/huggingface`) after that. `backend/core/models.py::warm_up()` loads
all three (dense, sparse, reranker) at API startup with a bounded timeout
(120s/60s/60s) specifically so a slow/stuck first-time download is visible in
the startup log — see [Reliability](#reliability-blocking-calls-in-an-async-server) below for why that matters.

### Scale: what happens with a very large document (hundreds to 1000+ pages)?

Short answer: **it isn't built for that yet, and here's exactly why**, based on
reading the actual code paths (not a guess):

- **The whole document lives in memory for the whole run.** `state["blocks"]`
  and `state["chunks"]` are plain Python lists that grow for the entire graph
  execution (`backend/pipeline/graph.py`) — nothing is streamed or windowed to
  disk/DB mid-run. A 1000-page manual with hundreds of figures means hundreds
  of `NormalizedBlock`s and (at `chunking.size: 400` tokens) potentially
  thousands of `Chunk`s, all resident simultaneously.
- **Nothing is persisted until the very end.** `IndexTool` (the last step)
  is the *only* thing that writes to Postgres/Qdrant. If the process crashes,
  times out, or OOMs at page 999 of 1000, **zero** chunks are saved — not a
  partial result. (Status/progress updates DO stream incrementally via
  `on_step`/`PostgresStore.update_progress` — so the UI shows live progress —
  but that's status metadata, not the searchable content.)
- **`enrich_chunks` calls the LLM sequentially, batched 5 chunks/call, with a
  deliberate pacing delay** (`enrichment.min_interval_s: 2` — proactive Groq
  rate-limit avoidance). 2000 chunks ÷ 5 = 400 calls × (call latency + 2s pacing)
  — this alone can be tens of minutes for a genuinely huge document.
- **`vision_enrichment` runs concurrently within a document**
  (`ThreadPoolExecutor`, `max_concurrency: 4`) but the whole step still has to
  finish before `chunk` starts — a document with hundreds of figures means
  hundreds of vision API calls, batched 4 at a time.
- **`IndexTool` writes one chunk at a time** in a Python loop (`backend/storage/index_tool.py`)
  — not a batched/bulk insert. Thousands of chunks means thousands of
  individual Postgres + Qdrant round-trips at the end of the run.
- **`/upload` doesn't block the HTTP response** — ingestion runs as a FastAPI
  `BackgroundTasks` job, so the *request* returns instantly regardless of
  document size. But the background job itself has no time limit, and (per the
  points above) a very large doc could run for a very long time with no partial
  credit if it fails.

**What a real fix looks like** (not built — this is the known gap, tracked as
"SCALE" in the team's planning, not invented for this doc): page-range
windowing (process N pages, index them, release memory, continue — so a crash
loses at most one window, not the whole document), checkpoint/resume on
`document_id`, and batched (not per-row) DB writes in `IndexTool`. None of this
is architecturally hard given the existing per-page signals (`page_router.py`
already measures pages independently) — it just isn't wired up yet. **Today's
practical ceiling** is "however many pages complete within your patience and
the process's memory" — comfortable for tens of pages, workable for low
hundreds, risky past that without testing your specific document's figure/table
density (the real cost driver is vision + enrichment call count, not raw page
count).

### Tables, images, and other visuals — how they're stored and queried

A table or figure is **never split** during chunking (`ATOMIC_TYPES` in
`chunk_tool.py`) — each becomes exactly one `Chunk`, so a citation always points
at a *complete* table or figure, never a fragment.

- **Tables**: Docling's TableFormer recovers real row/column structure (not
  just text-in-a-box); the chunk carries both `text` (a markdown rendering, for
  embedding/keyword search) and `table_data` (`{headers, rows}`, structured —
  for exact-value display). A user asking "what's the torque spec for the M6
  bolt" matches on the chunk's text/embedding like any other content; the
  answer's citation can render the *exact* table, not a paraphrase, because
  `table_data` travels with the citation all the way to the frontend.
  Complex tables that Docling's model doesn't handle well auto-escalate to a
  VLM transcription instead (see `docling_extract.py`'s table auto-escalation).
- **Figures/diagrams/photos**: never searched by pixels — a vision model writes
  a `image_caption` block (searchable text description, informed by the
  surrounding page text for context — see `_prompt_with_context` in
  `vision_enrichment.py`), and the block carries `image_path` (where the
  rendered crop lives — see Storage below) for display. So "what does the
  exploded diagram on page 12 show" works because the caption's *text* is what
  matched the query — the image itself is never embedded or vector-searched,
  only described and shown.
- **Both are answerable in a normal question** — there's no separate "table
  mode" or "image mode" to invoke; `search_documents` retrieves across all
  chunk types uniformly, and the answerer cites whichever chunk(s) — text,
  table, or figure — actually answered the question.

## Query flow

`query_planner → retrieval → answerer`

- **query_planner** — rewrites a follow-up into a standalone query using
  conversation history, optionally decomposes into sub-questions.
- **retrieval** — hybrid dense+sparse search (RRF fusion) + cross-encoder rerank
  by default (`query.retrieval.method: hybrid_rerank` in config; naive/hybrid/hyde/enriched
  also available).
- **answerer** — grounds the answer in retrieved chunks only, cites
  `[filename, page/sheet/slide]`, and says "not found" rather than guessing when
  nothing relevant comes back.

`search_documents()` (`backend/retrieval/search_documents.py`) wraps this the
same way `ingest_document()` wraps ingestion — one call, agent-callable.

---

## Agent

`backend/agent/executor.py` is a small LangGraph loop (two nodes: `agent` ↔
`tools`) that lets a model **pick** which tool to call instead of a human always
driving the same fixed action. It uses the provider's native tool-calling
(`llm.bind_tools()`) — deliberately not LangChain's `AgentExecutor`/
`create_tool_calling_agent`, so the write-approval logic below stays in our own
code instead of a framework abstraction.

**Tools available today** (`backend/agent_tools.py::build_agent_registry()`):
| Tool | What it does | Backed by |
|---|---|---|
| `ingest_document` | Ingest a file so it becomes searchable | the ingestion pipeline above |
| `search_documents` | Answer a question from the ingested corpus, with citations | the query pipeline above |
| `sql_read` | Run a read-only SQL query, get rows back | `backend/connectors/sql_read.py` (blocks anything that isn't SELECT/WITH/SHOW/DESCRIBE/EXPLAIN) |

**Write approval:** any tool listed in `query.agent.write_tools` (config —
today just `ingest_document`) is **not executed** the first time the model asks
for it. The executor stops and returns `status: "needs_approval"` with what it
wants to run; nothing happens until the caller re-invokes with
`approved_writes=True` for the same message. Reads execute immediately.

**Model:** `query.agent.provider`/`model` in config — today `openai` /
`gpt-4o-mini`. Any tool-calling model works; the `llm.api_key` mechanism is
unaffected by swapping it.

Two caveats before pointing this at a different model. Ids get retired without
notice, so check the provider's live list rather than trusting one named here.
And `query.agent.intent` caps its call at 12 tokens — a *reasoning* model spends
that budget on reasoning tokens and returns empty content, so the classifier
silently falls back on every turn. Verify a candidate returns a bare label at
`max_tokens=12` before switching.

**Try it:**
```bash
python scripts/agent_chat.py "what does the warranty policy say about returns?"
python scripts/agent_chat.py "please ingest ./some_file.pdf"   # will ask you to approve first
python scripts/agent_chat.py                                    # interactive, keeps memory
```
or the frontend's `/chat` page (agent-only — see below), or directly:
`POST /agent/chat {"message": "...", "session_id": "...", "approved_writes": false}`.

**Conversation history and its limit:** every turn resends the whole
conversation to the LLM (no server-side chat state on the provider's side), so
history has to live somewhere between turns. Two layers:
- **In-memory per-session cache** (`_agent_sessions` in `backend/api/main.py`) —
  fast path within a running process; lost on restart.
- **Postgres** (`conversations` table, `backend/storage/conversation_store.py`)
  — every completed turn is persisted (with tool-call metadata), so the chat
  UI's sidebar can list and reopen past conversations even after a restart.

Both are capped at `query.agent.max_history_messages` (default 20) — a plain
sliding window, **not** summarization: a long conversation loses its earliest
turns rather than compacting them. This is deliberate: unlike an agentic coding
session (Claude Code, hours long, huge tool outputs, needs real token-aware
compaction), this is a document-Q&A assistant — sessions are short, tool
results are small JSON blobs, not file dumps. A fixed window is enough *for
now*; if usage ever shows long sessions where losing early context hurts answer
quality, that's the signal to build real summarization, not before.

**Frontend:** `/chat` (`frontend/src/components/ChatPage.jsx`) is a ChatGPT/
Claude-style interface — left sidebar (new chat + past conversations, backed by
`GET/DELETE /agent/sessions*`), centered message column with markdown
rendering, tool-call badges, and an inline approve/decline card for writes. A
paperclip button stages a file (`POST /files/stage` — saves it to disk **without**
ingesting; ingestion only happens if the agent calls `ingest_document`, which
still needs your approval) so "attach a file and ask to ingest it" is one
conversational flow, not a separate upload page. There is no direct
(non-agentic) RAG endpoint — every document question goes through the agent;
`search_documents` is a tool the agent calls, not its own HTTP route.

---

## Vision

There is **one shared vision client**, `backend/core/vision_client.py::describe_image()`
— every part of the system that needs a model to look at an image calls this
one function, with retry/backoff for rate limits built in. There are **three
call sites**, each with its own config block so they can point at different
models/providers if you want:

| Call site | Config block | Calls per doc | What it produces |
|---|---|---|---|
| `categorize` (`backend/categorize/classifier.py`) | `vision:` | 1 (stitched first page(s)) | `{document_type, industry, confidence, reasoning}` |
| `vision_enrichment` (`backend/vision/vision_enrichment.py`) | `vision:` | 1 per significant figure/diagram/vector-graphics page | `image_caption` blocks (searchable text describing each figure) |
| `vision_ocr` (`backend/extraction/vision_ocr.py`) | `vision_ocr:` | 1 per page flagged complex (or every page, `mode: always`) | replaces garbled/table text blocks with a clean VLM transcription |

`categorize` and `vision_enrichment` share the `vision:` block by convention
(same model captions figures and classifies documents); `vision_ocr` is a
**separate** block — it's full-page transcription, a different job with
different quality/cost tradeoffs, so it can run a different model/DPI/mode
independently. `vision_ocr.mode` controls how aggressively it runs: `off`
(native text/OCR only), `complex` (only pages flagged garbled/table/figure —
cheap, default), `always` (every page through the VLM — highest quality, most calls).

### Providers — do you need a self-hosted GPU vision model?

**No, not for this accelerator.** `describe_image()` supports three provider
kinds (`config["vision"]["provider"]`):
- `google` — Gemini/Gemma via Google AI Studio. **Free tier**, no infrastructure.
  Good default for development and light production use.
- `openai` + `base_url` — any OpenAI-compatible multimodal endpoint: OpenAI
  GPT-4o-vision, NVIDIA Build API / NIM, OpenRouter, or a self-hosted vLLM box.
- `ollama` — a local model via Ollama, no key, no external calls at all.

A self-hosted GPU running a large open-weight VLM (e.g. Qwen3-VL) makes sense
when you have a **specific reason** the free/hosted tiers don't cover:
sustained high volume that would blow through free-tier rate limits, data that
legally/contractually can't leave your network, or a latency/consistency need a
shared free API can't guarantee. None of those are inherent to this
accelerator — they're deployment-specific decisions. If a given deployment has
sensitive documents or high daily volume, point `vision.base_url` /
`vision_ocr.base_url` at a self-hosted OpenAI-compatible endpoint (the config
has a commented-out recipe for exactly this — search for "self-hosted" in
`config/global.yaml`); otherwise Gemini's free tier or a Build-API-style hosted
endpoint is simpler to run and cheaper to maintain than owning a GPU box.

`vision_ocr` in particular is the expensive one to run on `mode: always` (every
page, not just flagged ones) — that's where volume/cost considerations matter
most, and where a dedicated GPU pays off first if you get there.

---

## Data plane

- **Postgres** — documents, chunks, tags, ingestion status, conversation history.
- **Qdrant** — dense (nomic, 768-dim) + sparse (BM25) vectors, named per leg,
  filterable by chunk tags for scoped search.
- **Local disk** (`uploads/`) — original files, rendered figure crops, full-page
  images for citation grounding. An `ObjectStore` abstraction exists
  (`backend/storage/object_store.py`) if you want to swap this for S3/MinIO.

Bring the data plane up with Docker — see [Docker services](#docker-services).
Reset everything for a clean test run: `./scripts/reset_state.sh` (Postgres rows +
Qdrant collection + local `uploads/`) or `docker compose down -v` (also wipes the
Docker volumes themselves).

## Reliability: blocking calls in an async server

Found live while testing, not theoretical: **this API has exactly one worker
running an async event loop** (`uvicorn backend.api.main:app`, no `--workers`
flag). FastAPI route handlers are `async def`, but almost everything they call
underneath — every `Tool.run()`, `describe_image()`, the local model calls — is
**plain synchronous Python**, called directly, not via `run_in_executor` or a
thread pool. A synchronous call inside an `async def` handler blocks the entire
event loop — not just that request, **every** request, including `/health`.

This bit twice during this session's testing:
1. `get_reranker()` lazy-loaded `BAAI/bge-reranker-large` (~1.3GB) on the first
   real retrieval call. The download stalled in the sandbox; the request never
   returned, and `/health` stopped responding too — proof the whole process was
   frozen, not just that one slow request.
2. `describe_image()`'s retry loop calls `time.sleep(delay)` between attempts
   (`backend/core/vision_client.py`) — a provider hiccup (a 500, a rate limit)
   during an agent-triggered ingest blocked the server for the full backoff
   window, again server-wide.

**Fix applied for #1**: `warm_up()` now loads the reranker (and sparse model) at
startup, not on first request, with a bounded per-model timeout — see Models &
limits above. This moves the risk to a visible startup-log line instead of a
silent hang on someone's first query.

**Not fixed, worth knowing (#2 and the general pattern)**: any tool that does
blocking I/O with retries/sleeps — which is most of them, by design, since
`Tool.run()` is a plain sync method — can still stall the whole server while
it's running. The industry-standard fixes, in order of effort:
1. **Run each tool call in a thread pool** (`asyncio.get_event_loop().run_in_executor(None, tool.run, state, config)`
   in the graph node wrapper, or in the `/agent/chat`/`/upload` handlers) — keeps
   the event loop free even while a tool blocks. Cheapest fix, no architecture change.
2. **Run uvicorn with multiple workers** (`--workers N`) — one stuck request no
   longer starves every other request, though each worker still blocks on its
   own. Complementary to #1, not a replacement (module-level singletons like the
   model caches would need to be per-worker or moved to a shared store).
3. **True async I/O** (httpx async client instead of `requests`/SDK sync calls,
   `asyncio.sleep` instead of `time.sleep`) — the correct long-term fix, but
   touches every provider call site; not a small patch.
For a single-user dev/demo setup this has never mattered; it matters the moment
more than one person can hit the API at the same time.

## Docker services

Split by concern so you only run what you need:

| File | Services | When |
|---|---|---|
| `docker-compose.yml` | Postgres, Qdrant | Always — required to run the pipeline. |
| `docker-compose.devtools.yml` | Adminer (`:8080`, browse Postgres) | Local development. |
| `docker-compose.observability.yml` | Langfuse (`:3001`, LLM call tracing), Grafana (`:3000`, metrics) | When you want tracing/dashboards. |

```bash
docker compose up -d                                                    # core only
docker compose -f docker-compose.yml -f docker-compose.devtools.yml up -d       # + Adminer
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d  # + Langfuse/Grafana
```

## Testing

```bash
pytest                       # full suite; DB-gated tests auto-skip if Postgres/Qdrant aren't up
python tests/test_smoke.py   # fast no-deps sanity check
```
See `CONTRIBUTING.md` for coding standards and PR expectations.
