# AI Accelerator — Technical Plan

> Architecture reference for the Document Intelligence + Enterprise RAG Accelerator.
> Read this to understand how the system is designed and why each decision was made.

---

## Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Tech Stack](#2-tech-stack)
3. [How the System Works](#3-how-the-system-works)
4. [Routes and Config](#4-routes-and-config)
5. [Connector Tools and MCP](#5-connector-tools-and-mcp)
6. [Shared Data Shapes](#6-shared-data-shapes)
7. [Tool Reference](#7-tool-reference)
8. [Observability](#8-observability)
9. [Local Dev Setup](#9-local-dev-setup)
10. [Design Decisions & Rationale](#10-design-decisions--rationale)
11. [What Comes Next — Phase 2](#11-what-comes-next--phase-2)

---

## 1. What We Are Building

A **Document Intelligence + Enterprise RAG Accelerator** — a reusable engine that ingests enterprise documents and answers questions about them, with citations.

**Supported document types:** PDF, Excel, PowerPoint, images. CAD drawings and circuit diagrams always arrive as PDFs — the vision model is the primary extraction method for them, not OCR.

**This is not a one-off chatbot.** The same engine serves different clients by changing a config file. A legal firm, an automotive company, and a pharma company all run the same code with different configurations.

**Example:** Toyota uploads assembly drawings and technical manuals. An engineer asks, *"What is the connector type for the power module on sheet 3?"* The system finds the relevant diagram, returns the answer with a citation — and shows the engineer the actual diagram image so they can see it directly.

**What the system returns for each answer:**
- The answer text, grounded in document content
- Citations: filename + page/slide/sheet number + a short text snippet
- For visual content: the actual image so the user can see what was found
- For table content: the structured table data so it can be rendered properly

### How people actually use it

One deployment serves one client, and all their uploaded documents form **one shared collection**. There are no separate logins or per-user document sets in Phase 1 — everyone working in a deployment sees the same pool of documents.

The flow from a user's point of view:

1. **Upload documents** — drag in PDFs, Excel, PPT, images. Each one processes in the background; the user sees its status (processing → ready, or failed).
2. **See the collection** — a list of all documents that have been ingested, with the ability to delete one.
3. **Ask questions** — type a question. By default it searches the **whole collection**. Optionally the user can narrow it to specific documents (tick a few files) before asking.
4. **Get a cited answer** — the answer comes back grounded in the documents, with citations, inline images, and tables.

**Phase 1 is single-shot Q&A.** Each question is answered independently. Multi-turn chat (follow-ups that understand previous turns) is Phase 2 — see §11.

This is delivered through a **minimal web UI** (upload + status, document list, ask-a-question with citations) backed by the FastAPI endpoints. Accounts, multiple projects, and admin screens are explicitly out of scope for Phase 1.

### Core design principles

**Everything is a Tool.** Each step in the pipeline is a self-contained function with a defined input and output. Tools do not call each other — they read from and write to a shared state object. Any tool can be added, removed, or replaced without touching the others.

**Config drives the pipeline.** Which tools run, in what order, and with what settings is defined in a YAML config file. Adding a new capability means adding a tool and one line in the config — no changes to existing code.

**Ingestion is deterministic. Queries are agentic.** When a document is ingested, every step is predictable and fixed. When a user asks a question, an AI agent decides how to answer — whether to call a connector for live data, or answer from retrieved chunks directly.

**Data stays on your infrastructure.** Files, extraction, embeddings, vectors, and databases are all self-hosted. Only the vision model and LLM are external API calls. What gets sent externally is kept minimal, downscaled, and logged.

---

## 2. Tech Stack

> Every choice below has a reason. See [Section 10 — Design Decisions & Rationale](#10-design-decisions--rationale) for *why* each one was picked and what the alternative was.

| Component | Technology |
|---|---|
| Pipeline orchestration | LangGraph |
| Backend API | FastAPI |
| PDF processing | PyMuPDF (fitz) |
| OCR (scanned pages) | PaddleOCR |
| Image / object detection | YOLO (ultralytics) |
| CAD and circuit diagram PDFs | Vision model (Gemma) is the primary extraction — OCR alone cannot read technical drawings |
| Excel extraction | openpyxl + pandas |
| PowerPoint extraction | python-pptx |
| Image processing | Pillow |
| Vision model | Gemma 4 (multimodal, API) — use the largest multimodal variant for best vision quality |
| Text LLM | Config-driven — any compatible provider |
| LLM client | `backend/core/llm_client.py` — one shared wrapper, no provider SDK calls in tools |
| Vision client | `backend/core/vision_client.py` — one shared `describe_image()` both vision callers use |
| Dense embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (1024-dim), runs locally — top MTEB benchmark performance for English retrieval |
| Sparse embeddings (BM25) | fastembed `Qdrant/bm25`, runs locally — needed for hybrid search |
| Reranker | `BAAI/bge-reranker-large` via sentence-transformers CrossEncoder, runs locally — same family as the embedder, better coherence |
| Chunking | `chonkie` SemanticChunker — token-aware; uses the same bge-large embedder loaded via `models.py` singleton (no second model in memory) |
| Vector database | Qdrant **≥ 1.10** — uses the native Query API with `Prefetch` + `Fusion: RRF` for dense+sparse hybrid search in a single round-trip |
| Relational database | PostgreSQL |
| File storage | Local filesystem (`uploads/` directory for documents, `uploads/images/` for extracted visuals) |
| Connector tools protocol | MCP (Model Context Protocol) |
| LLM call tracing | Langfuse |
| Config format | YAML |
| Code style | black + ruff |
| Local stack | Docker Compose |

**Note on metrics:** Grafana / Prometheus deferred to Phase 3. Phase 1 ships with Langfuse for LLM call tracing only — that's enough to debug quality issues; system-level metrics aren't needed until there's real traffic.

**Deployment assumption (Phase 1).** The Phase 1 service is designed to run on a **private network — behind a corporate VPN or reverse proxy** that handles authentication, TLS, and rate-limiting. The FastAPI service itself has no auth and no rate-limit middleware in Phase 1; CORS is permissive to localhost only. Exposing the service directly to the public internet is out of scope until Phase 2 adds an auth layer.

---

## 3. How the System Works

The system has two separate pipelines. They share the same tools but serve different purposes.

---

### 3.1 Ingestion Pipeline — Processing a document

Runs once when a document is uploaded. Every step is fixed and predictable.

```
Document uploaded
      │
      ▼
 CATEGORIZE
 Step 1 — Gemma LABELS the document (it does not pick the route):
   - document_type: invoice | cad_drawing | report | circuit_diagram | ...
   - industry:      automotive | finance | legal | ...
 Step 2 — the config maps that label to a route (a plain lookup):
   route = type_to_route[document_type]      ← see Section 4.4
      │
      ▼
 ROUTE SELECTED
 The type_to_route table lives in config/global.yaml. At runtime it is just a
 dictionary lookup — deterministic, no AI deciding the route.
      │
      ├── diagram_heavy    ──→  extract → vision_enrichment → chunk → enrich → embed → index
      │
      ├── table_heavy      ──→  extract → chunk → enrich → embed → index
      │
      ├── text_default     ──→  extract → chunk → enrich → embed → index
      │
      ├── presentation     ──→  extract → vision_enrichment → chunk → enrich → embed → index
      │
      └── cad_route        ──→  extract → vision_enrichment → chunk → enrich → embed → index
                                (vision is the primary extraction — OCR output is minimal for these)
      │
      ▼
 FINALIZE — pipeline writes document_type, industry, route, and final status
            back to the `documents` row in PostgreSQL.
```

**Two separate decisions — keep them distinct:**
- **The route** (from `document_type`) decides *content handling*: vision on or off, chunk size, the vision prompt.
- **The file type** decides *which extractor runs*. The `extract` step in every route is generic — the graph resolves it based on `state["file_type"]`:

  | file_type | What `extract` runs |
  |---|---|
  | pdf | `page_profile_tool` → `pdf_extraction_tool` |
  | excel | `excel_extraction_tool` |
  | ppt | `ppt_extraction_tool` |
  | image | `image_extraction_tool` (wraps the uploaded image as one image block for vision enrichment) |

  This is why an Excel invoice and a PDF invoice can share the `table_heavy` route (same chunk size, vision off) while still using different extractors. Adding a new file type later means adding one extractor and one line in the resolver — no route changes.

**Standalone image files** (`.jpg`, `.png`) are always routed to `image_route` (vision on) — see Section 4. The vision model is the only way to extract meaning from a bare image, so categorization sends every image file there regardless of what the image appears to contain.

**What each step does:**

| Step | What it does |
|---|---|
| **Categorize** | Identifies document type and industry. Chooses the processing route. |
| **Extract** | Resolved by file type (see table above). PDFs get page profiling + text/OCR; Excel and PPT get their own extractors. Produces NormalizedBlocks. Image regions become image blocks with empty `text`, filled later by vision enrichment. |
| **Vision Enrichment** | Renders each significant image region. Sends it to Gemma. Saves the rendered image to `uploads/images/` so it can be shown to users later. Stores Gemma's description as a searchable text block with the image path attached. |
| **Chunk** | Splits content into small passages for retrieval. Tables are never split — they stay as one chunk with their full structured data attached. |
| **Enrich Chunks** | Tags each chunk with topic, section, and keywords using the LLM. |
| **Embed** | Converts each chunk's text into a vector using a local model. |
| **Index** | Stores each chunk in PostgreSQL (text + tags + structured data) and Qdrant (vector + tags). |
| **Finalize** | `run_pipeline` writes `document_type`, `industry`, `route`, and final `status` (ready/failed) back to the `documents` row in PostgreSQL. A **startup sweep** runs on FastAPI boot: any row left in `status='processing'` from a prior crash is marked `failed` with an `ErrorEntry` of `{tool: "startup_sweep", level: "error", message: "orphaned during prior run"}` so the UI never shows a permanently-spinning row. |

**Key rules:**
- Every chunk is stored with metadata tags — document type, industry, topic, section, keywords.
- Image chunks carry the path to the saved image file.
- Table chunks carry the full structured `table_data` (headers + rows).
- There is **one database schema for all clients and document types**. Category is a tag on every chunk, not a separate database.
- The original filename is set on every block's `source_ref.filename` so citations always show the real document name.

**What each extractor produces (all output `NormalizedBlock` — see Section 6.4):**

| Source | Block types produced | `source_ref` fields used | Notes |
|---|---|---|---|
| PDF (digital page) | `text`, `heading` | `page` | Direct text via fitz |
| PDF (scanned page) | `text` | `page` | OCR via PaddleOCR; `confidence` set |
| PDF (image region) | `image` → becomes `image_caption` | `page`, `bbox` | Rendered from PDF region by vision |
| Excel | `table` | `sheet`, `cell_range` | `table_data` filled; markdown in `text` |
| Excel (chart) | `image` → `image_caption` | `sheet` | Chart image bytes saved by extractor |
| PPT (text slide) | `text` | `slide` | Titles, text boxes, notes |
| PPT (table) | `table` | `slide` | `table_data` filled |
| PPT (image) | `image` → `image_caption` | `slide` | Image bytes saved by extractor |
| PPT (chart) | `image` → `image_caption` | `slide` | Chart rendered to image bytes by extractor |
| Image file | `image` → `image_caption` | `filename` | Whole file is the image |
| CAD / circuit (PDF) | `image` → `image_caption`, minimal `text` | `page`, `bbox` | Vision is primary; OCR yields little |

The arrow (`image` → `image_caption`) means: the extractor creates an `image` block with empty `text`; `vision_enrichment_tool` then fills the description and appends an `image_caption` block.

---

### 3.2 Query Pipeline — Answering a question

Runs when a user asks a question. Has one AI agent that decides how to answer.

**One thing arrives with every question:**
- **Scope** — which documents to search. By default this is the **whole collection**. If the user narrowed to specific files, the request carries those `document_ids` and search is limited to them. (An empty/absent list means "search everything.")

Phase 1 is single-shot — there is no conversation history. Multi-turn / follow-up support is Phase 2.

```
User asks a question  (query + optional document_ids)
      │
      ▼
 QUERY PLANNER
 Decompose: if the question needs multiple lookups, break it into 2–3 sub-questions.
 Otherwise sub_questions = [query].
      │
      ▼
 RETRIEVE  (per sub-question)
 Hybrid search in Qdrant: dense vector search + BM25 keyword search, merged with RRF
 Filtered by document scope, industry, document type
      │
      ▼
 RERANK
 CrossEncoder rescores the candidates — keeps the most relevant top-K
      │
      ▼
 ┌──────────────────────────────────────────────────────┐
 │  AGENT NODE  — owns the ACTION decision               │
 │                                                       │
 │  The agent decides: can I answer with the chunks I    │
 │  have, or do I need to pull live data from an         │
 │  external system first?                               │
 │                                                       │
 │  Tools available to the agent (via create_react_agent):│
 │       db_query_tool      database lookup              │
 │       (additional connectors land in Phase 2)         │
 │                                                       │
 │  answer_tool is NOT an agent tool — it is the next    │
 │  graph node. The agent decides "done" by emitting     │
 │  no further tool call; the graph then routes to       │
 │  answer_tool unconditionally.                         │
 │                                                       │
 │  Connector tool outputs are mirrored from the         │
 │  agent's message stream into state["connector_results"]│
 │  by a thin tool wrapper, so answer_tool can cite them.│
 │                                                       │
 │  Max iterations is set in config; on exhaustion the   │
 │  graph force-routes to answer_tool with current state.│
 └──────────────────────────────────────────────────────┘
      │
      ▼
 ANSWER  (answer_tool — graph node, always runs after the agent)
 LLM generates a grounded answer with citations.
 The response includes:
   - answer text
   - for every cited chunk: filename + page/slide/sheet + text snippet
   - for image chunks: the image file path so the frontend can display it
   - for table chunks: the full table_data (headers + rows) so the frontend can render it
   - for connector calls: a connector citation (tool name + snippet of returned data)
```

**About the agent.** Retrieval quality is whatever the hybrid-search + reranker produces — Phase 1 does not run a corrective-RAG retry loop. The agent assumes retrieval is as good as it will get and decides whether to call a connector (for live data not in any document) before composing the answer. The agent is the only place connectors are ever called.

---

## 4. Routes and Config

A **route** is a named list of processing steps. It defines which tools run and with what settings for a given document type. Routes are entirely defined in config — nothing is hardcoded.

**Why this matters for clients:** A client changes their YAML file to change how their documents are processed. No code changes ever needed.

---

### 4.1 Default routes

The `extract` step is generic — the graph resolves it to the right extractor based on `file_type` (see Section 3.1). The route only controls vision settings and chunking.

```yaml
routes:

  # All chunk sizes and overlaps are in TOKENS (not characters).
  # bge-large tokenizer is used — roughly 1 token ≈ 3–4 characters for English.
  # The chunker uses semantic boundaries first, then falls back to token limits.

  diagram_heavy:
    # Technical documents with significant diagrams, charts, or visual content
    steps: [extract, vision_enrichment, chunk, enrich_chunks, embed, index]
    vision:
      enabled: true
      dpi: 150
      min_image_px: 100
      max_concurrency: 4
      timeout_s: 45
    chunking:
      strategy: semantic          # find natural semantic breaks first
      size: 256                   # tokens — keeps technical context tight
      overlap: 32                 # tokens
      fallback: recursive         # if a block is too large, split recursively

  table_heavy:
    # Financial reports, invoices, spreadsheets, data-dense documents
    # Works for both PDF and Excel — file_type picks the extractor
    steps: [extract, chunk, enrich_chunks, embed, index]
    vision:
      enabled: false
    chunking:
      strategy: recursive         # tables stay whole; text split recursively
      size: 480                   # tokens — largest used; capped under bge-large's 512-token input limit
      overlap: 64

  # NOTE on chunk sizes: bge-large-en-v1.5 and bge-reranker-large both truncate input at 512 tokens.
  # All route chunk sizes are kept ≤ 480 to leave headroom for any prefix/suffix the chunker adds
  # and to avoid silent truncation that would degrade embedding and rerank quality.

  text_default:
    # Reports, contracts, standard text documents
    steps: [extract, chunk, enrich_chunks, embed, index]
    vision:
      enabled: false
    chunking:
      strategy: semantic
      size: 384                   # tokens — enough for a paragraph of reasoning
      overlap: 48

  presentation_route:
    # PowerPoint and slide decks
    steps: [extract, vision_enrichment, chunk, enrich_chunks, embed, index]
    vision:
      enabled: true
      dpi: 120
      min_image_px: 200
      max_concurrency: 4
      timeout_s: 30
    chunking:
      strategy: recursive         # slides are already short units
      size: 192                   # tokens — one slide's worth of content
      overlap: 16

  cad_route:
    # CAD drawings and circuit diagrams — always arrive as PDFs
    # Vision model is the PRIMARY extraction method.
    steps: [extract, vision_enrichment, chunk, enrich_chunks, embed, index]
    vision:
      enabled: true
      dpi: 200
      min_image_px: 30
      max_concurrency: 4
      timeout_s: 60
      prompt: >
        Describe this technical drawing in detail. Extract all of the following that are visible:
        component labels, part numbers, dimension values, tolerances, connection types,
        net names, reference designators, layer names, and any annotations or notes.
        Be precise — exact values and labels matter.
    chunking:
      strategy: recursive
      size: 192                   # tokens — vision captions are concise
      overlap: 24

  image_route:
    # Standalone image files (.jpg, .png). Vision is the only extraction method.
    steps: [extract, vision_enrichment, chunk, enrich_chunks, embed, index]
    vision:
      enabled: true
      min_image_px: 0             # always enrich the whole image
      max_concurrency: 4
      timeout_s: 45
    chunking:
      strategy: recursive
      size: 384
      overlap: 48
```

**Note on CAD and circuit diagram extraction:** CAD drawings and circuit diagrams are complex and their content varies significantly depending on the diagram type, client, and tool that produced them. The approach to interpreting what Gemma returns for these documents — how to structure it, what to tag, what to prioritise — is driven by domain expertise. Whatever approach is used, the output must always be valid `NormalizedBlock` records. The shape is fixed; the logic that fills it is flexible.

---

### 4.2 Global config — models and providers

`config/global.yaml` holds settings that are not route-specific: the LLM provider, the vision model, and the local model names. `get_llm(config)` reads the `llm` block; tools read the rest as needed.

```yaml
# config/global.yaml

default_industry: general              # used when categorize cannot determine one

llm:
  provider: groq                       # groq | ollama | openai-compatible
  model: llama-3.3-70b-versatile       # the text model for planning, enrichment, answering
  temperature: 0.0

vision:
  provider: gemma                      # the multimodal model
  model: gemma-4                       # confirm exact variant for the deployment
  # HOW you reach Gemma (Google AI Studio, a self-hosted server, or HuggingFace) is the
  # implementer's choice — whatever works for you. Keep it behind this config + .env so the
  # access method can change without touching tool code.
  daily_call_limit: 5000               # safety budget — enforced via a Postgres counter row
                                       # `vision_daily_calls(day DATE PRIMARY KEY, count INT)` updated
                                       # with an atomic `UPDATE ... RETURNING` inside vision_client so
                                       # concurrent workers can't race past the cap. JSON-file counters
                                       # are NOT acceptable.

categorization:
  unknown_label_fallback: unknown      # if Gemma returns a document_type or industry NOT in the
                                       # allowed enum (§4.3), categorize_tool coerces it to "unknown"
                                       # (for document_type) or default_industry, and logs a warning.

embeddings:
  dense_model: BAAI/bge-large-en-v1.5  # 1024-dim
  sparse_model: Qdrant/bm25
  reranker_model: BAAI/bge-reranker-large

storage:
  qdrant_collection: chunks

upload:
  max_file_size_mb: 200
  allowed_extensions: [pdf, xlsx, xls, pptx, ppt, jpg, jpeg, png]

api:
  cors_origins: ["http://localhost:5173", "http://localhost:3000"]
```

---

### 4.3 Document types and industries

These lists are passed in the prompt to Gemma so it only returns values we recognize. Defined in `config/global.yaml`.

```yaml
document_types:
  - technical_diagram
  - datasheet
  - manual
  - invoice
  - financial_report
  - spreadsheet
  - presentation
  - report
  - contract
  - cad_drawing
  - circuit_diagram
  - unknown

industries:
  - automotive
  - pharma
  - finance
  - legal
  - manufacturing
  - general
```

---

### 4.4 type_to_route

Maps Gemma's `document_type` label to a route name. Defined in config.

```yaml
type_to_route:
  technical_diagram:  diagram_heavy
  datasheet:          diagram_heavy
  manual:             diagram_heavy
  invoice:            table_heavy
  financial_report:   table_heavy
  spreadsheet:        table_heavy
  presentation:       presentation_route
  report:             text_default
  contract:           text_default
  cad_drawing:        cad_route
  circuit_diagram:    cad_route
  unknown:            text_default
```

**Override for image files:** if `file_type == "image"`, `categorize_tool` sets `route = image_route` directly and skips the `type_to_route` lookup. A bare image always needs the vision model, whatever its apparent content.

**Per-client profile overrides** are Phase 2. Phase 1 ships one `global.yaml`.

---

### 4.5 Query settings

```yaml
query:
  max_context_tokens: 3000      # max tokens of chunk text in the answer prompt
  agent:
    max_iterations: 5           # on exhaustion, graph force-routes to answer_tool
  retrieval:
    mode: hybrid                # dense + BM25
    top_n: 20
    rerank: true
    rerank_top_k: 5
```

---

## 5. Connector Tools and MCP

Connector tools let the agent reach into external systems at query time — databases, inventory, or communication channels.

**Built as MCP servers** (Model Context Protocol) so any compatible AI client can call them without code changes.

```
Agent node in LangGraph
        │
        ▼  (via langchain-mcp-adapters)
┌───────────────────────────────────────────────┐
│  db_query MCP server   → client's database    │
│  (Phase 2: inventory, email, sms, teams)      │
└───────────────────────────────────────────────┘
```

**Phase 1:** One MCP server stub (`db_query`) — interface is real, response is mocked. Proves the agent-calls-connector pattern end-to-end.
**Phase 2:** Add the other four connectors (inventory, email, SMS, Teams) and wire each to the actual client system.

---

## 6. Shared Data Shapes

These are the agreed structures that connect all tools together — the exact field names and types that flow between them. Every tool reads from and writes to these shapes. **Do not change one without telling the whole team** — everyone builds to them, so a change in one place breaks others.

They all live in `backend/core/`.

---

### 6.1 PipelineState

```python
class PipelineState(TypedDict, total=False):
    # Set at upload time
    document_id:    str
    file_path:      str        # on-disk path; never trust for citations
    filename:       str        # original uploaded filename — extractors copy this onto every block
    file_type:      str        # pdf | excel | ppt | image

    # Set by categorize_tool; persisted to documents row by run_pipeline finalization
    route:          str
    document_type:  str
    industry:       str
    confidence:     float

    # Set by page_profile_tool (PDF only)
    page_profiles:  list       # PageProfile[]

    # Populated by extraction and enrichment tools
    blocks:         list       # NormalizedBlock[]

    # Populated by chunk, enrich, and embed tools
    chunks:         list       # Chunk[]

    # Query pipeline
    query:              str
    document_scope:     list   # optional document_ids to search within; EMPTY = whole collection
    sub_questions:      list
    retrieved_chunks:   list
    connector_results:  list   # [{tool, input, output}] — populated by agent when it calls a connector
    answer:             str
    citations:          list

    errors:         list       # list[ErrorEntry] — always append, never raise
```

**ErrorEntry shape (fixed in `schemas.py`):**

```python
class ErrorEntry(TypedDict):
    tool: str                  # e.g. "vision_enrichment"
    level: str                 # "warning" | "error"
    message: str
    block_id: str | None       # optional, when the error is about one block
```

**What each field means, who sets it, and why it's there:**

| Field | What it is (plain) | Set by | Why it exists |
|---|---|---|---|
| `document_id` | A unique ID for this uploaded document | upload | Links everything — chunks, images, status — back to one document |
| `file_path` | Where the uploaded file sits on disk | upload | Extractors open the file from here |
| `filename` | The original uploaded filename | upload | Copied onto every block's `source_ref.filename` so citations show real names |
| `file_type` | `pdf` / `excel` / `ppt` / `image` | upload | The graph uses it to pick the right extractor |
| `route` | Which processing recipe to run | categorize | Decides which tools run and their settings (vision on/off, chunk size) |
| `document_type` | What kind of doc Gemma thinks it is | categorize | Used to look up the route; also stored as a tag for filtering |
| `industry` | The document's industry | categorize | Stored as a tag so retrieval can filter by it |
| `confidence` | How sure categorize is (0–1) | categorize | Low confidence (< 0.6) is surfaced by `GET /status/{document_id}` as `needs_review: true` so a human can re-classify before relying on the document |
| `page_profiles` | Per-page x-ray of a PDF | page_profile | Tells extraction and vision how to treat each page |
| `blocks` | Raw extracted pieces (text/table/image) | extractors + vision | The common format everything downstream reads |
| `chunks` | Retrieval-sized pieces with tags + vectors | chunk/enrich/embed | What actually gets stored and searched |
| `query` | The user's current question | `/query` | What we're answering |
| `document_scope` | Which documents to search; **empty = whole collection** | `/query` | Lets a user narrow the search to chosen files; optional |
| `sub_questions` | The question broken into parts | query_planner | A complex question retrieves each part separately |
| `retrieved_chunks` | The chunks found for the question | retrieval | The evidence the answer is built from |
| `connector_results` | Live-data results from agent connector calls | agent | So `answer_tool` can include live data and cite it |
| `answer` | The final grounded answer | answer_tool | What we return to the user |
| `citations` | The sources behind the answer (file, page, image, table, or connector) | answer_tool | So the user can verify it and see the visuals |
| `errors` | Problems collected along the way | any tool | Nothing crashes a document — issues are surfaced here, not raised |

**`total=False`** means every field is optional — a tool only fills the fields it owns, and reads the ones it needs. The ingestion fields are filled during upload-processing; the query fields during a question.

---

### 6.2 Tool interface

```python
class Tool(Protocol):
    name: str
    def run(self, state: PipelineState, config: dict) -> PipelineState: ...
```

---

### 6.3 PageProfile

A per-page "x-ray" of a PDF, produced before extraction so each page can be handled correctly.

```json
{
  "page_number": 3,
  "kind": "digital | scanned | mixed",
  "text_len": 1280,
  "has_vector_graphics": true,
  "table_hint": false,
  "images": [
    { "bbox": [0, 0, 220, 180], "width": 220, "height": 180, "significant": true }
  ]
}
```

| Field | What it means | Why it matters |
|---|---|---|
| `page_number` | Which page this describes | So blocks can cite the right page |
| `kind` | `digital` (real text), `scanned` (image of text), `mixed` | Decides extraction: pull text directly vs. run OCR |
| `text_len` | How much extractable text is on the page | The main signal for `kind` — little text usually means scanned |
| `has_vector_graphics` | Whether the page has drawn shapes/lines | A hint that there are diagrams worth vision |
| `table_hint` | Whether the layout looks like a table | Helps extraction decide to keep tabular structure |
| `images[].bbox` | The rectangle of an image on the page | Vision renders exactly this region |
| `images[].width/height` | The image's pixel size | Used to decide if it's big enough to matter |
| `images[].significant` | Whether this image is worth sending to vision | Skips tiny logos/icons; only meaningful visuals get enriched |

---

### 6.4 NormalizedBlock

Every extractor produces this shape. All downstream tools work identically regardless of the source document type.

```json
{
  "block_id":    "uuid",
  "document_id": "uuid",
  "type":        "text | table | heading | image | image_caption",
  "text":        "always a string — tables get markdown text, images get Gemma's description",
  "table_data":  {
    "headers": ["col1", "col2"],
    "rows":    [["val1", "val2"]]
  },
  "source_ref":  {
    "filename":   "report.pdf",
    "page":       3,
    "sheet":      null,
    "slide":      null,
    "cell_range": null,
    "bbox":       [0, 0, 100, 100]
  },
  "confidence":  0.95,
  "language":    "en",
  "metadata": {
    "raw_image_path":    "uploads/images/<doc_id>/<block_id>_raw.jpg",
    "image_path":        "uploads/images/<doc_id>/<block_id>.jpg",
    "enrichment_failed": false
  }
}
```

| Field | What it means | Why it's there |
|---|---|---|
| `block_id` | Unique ID for this one piece of content (UUID) | So a chunk and its image can be traced back to the exact block |
| `document_id` | Which document this came from | Links the block to its source |
| `type` | `text` / `table` / `heading` / `image` / `image_caption` | Tells chunking how to handle it (e.g. never split a table) |
| `text` | The readable text of this block | What gets embedded and searched; always a string |
| `table_data` | A table's headers + rows, kept structured | So the UI can show a real table, not flattened text; null if not a table |
| `source_ref` | Where in the document this came from | Powers citations — `filename` + `page`/`sheet`/`slide`/`cell_range`/`bbox` |
| `confidence` | How sure the extractor/OCR is | Low-confidence OCR text can be flagged or de-prioritised |
| `language` | Detected language of the text | Set by extractors via langdetect; non-`en` triggers a warning (bge-large is English-only) |
| `metadata.raw_image_path` | Saved bytes of a PPT/Excel/standalone image | Vision reads it directly (these can't be rendered from a PDF) |
| `metadata.image_path` | The final saved image shown to the user | Returned in citations so the answer can display the picture |
| `metadata.enrichment_failed` | Whether vision failed on this image | The pipeline continues; this marks the gap instead of crashing |

**Rules:**
- `text` is always a string, never null.
- `source_ref.filename` is **always** set — extractors copy it from `state["filename"]` (the original upload name) on every block.
- `block_id` is generated by the extractor: `str(uuid4())`.
- Image blocks come in two render styles, and `vision_enrichment_tool` handles both:
  - **PDF image regions** set `source_ref.bbox` (and no `raw_image_path`) — vision renders the region from the PDF with fitz.
  - **PPT / Excel / standalone-image** blocks cannot be rendered with fitz. The extractor saves the embedded image bytes to `metadata.raw_image_path` — vision reads that file directly.
- Image blocks: `text = ""` until `vision_enrichment_tool` fills it. `metadata.image_path` (the final saved/displayed image) is set at that point.
- Table blocks: `text` = markdown version of the table. `table_data` holds the full structured data. `source_ref.cell_range` (e.g. `"Sheet1!A1:D20"`) is set for Excel tables.
- `table_data` is null for non-table blocks.
- `metadata.image_path` / `raw_image_path` are null for non-image blocks.

---

### 6.5 Chunk

What gets stored and indexed. Carries everything needed for retrieval and display.

```json
{
  "chunk_id":    "uuid",
  "document_id": "uuid",
  "text":        "the text used for search and embedding",
  "tags": {
    "industry":  "automotive",
    "doc_type":  "circuit_diagram",
    "topic":     "power supply section",
    "section":   "3.2 Power Rail",
    "keywords":  ["op-amp", "5V rail", "bypass capacitor"]
  },
  "source_ref": {
    "filename": "wiring.pdf",
    "page":     12
  },
  "table_data": {
    "headers": ["Part", "Value", "Tolerance"],
    "rows":    [["R1", "10kΩ", "1%"]]
  },
  "image_path": "uploads/images/<doc_id>/<block_id>.jpg",
  "token_count":   312,
  "vector":        [0.12, -0.34],
  "sparse_vector": { "indices": [12, 88, 301], "values": [0.4, 0.9, 0.2] }
}
```

| Field | What it means | Why it's there |
|---|---|---|
| `chunk_id` | Unique ID for this chunk (UUID) | The shared key linking the PostgreSQL row and the Qdrant point |
| `document_id` | Which document it belongs to | Lets search filter to a document and lets delete remove all its chunks |
| `text` | The passage of text | What gets embedded and what the answer is grounded in |
| `tags` | Labels: industry, doc_type, topic, section, keywords | Used to filter search ("only automotive") and sharpen matching |
| `source_ref` | Filename + page/slide/sheet | Powers the citation shown with the answer |
| `table_data` | Structured table, if this chunk is a table | So the UI renders a real table in the answer; null otherwise |
| `image_path` | The image, if this chunk is an image caption | So the UI shows the picture in the answer; null otherwise |
| `token_count` | How many tokens the text is | Lets the answer step pack the prompt to a known budget |
| `vector` | Dense embedding (meaning) | Semantic search — finds chunks that *mean* the same thing |
| `sparse_vector` | BM25 sparse embedding (keywords) | Keyword search — catches exact terms like part numbers |

- `table_data` is null for non-table chunks.
- `image_path` is null for non-image chunks.
- `vector` is the dense embedding (`bge-large-en-v1.5`, **1024-dim**). `sparse_vector` is the BM25 sparse embedding (fastembed). Both are produced by `embed_tool` and both are needed for hybrid search.
- `token_count` is set during chunking. For table chunks (which bypass chonkie) the chunk tool counts tokens via the bge-large tokenizer directly.
- **Storage split:** `text`, `tags`, `source_ref`, `image_path`, the dense `vector`, and the `sparse_vector` are stored in Qdrant (the two vectors as named vectors — see index_tool). `text`, `tags`, `source_ref`, `image_path`, and `table_data` are stored in PostgreSQL. `table_data` is **not** in Qdrant — it can be large and is only needed after a chunk is retrieved, so the retrieval tool fetches it from PostgreSQL by `chunk_id` for any table chunks in the results.
- When a chunk is retrieved and returned to the user, `image_path` and `table_data` are included in the citation so the frontend can display them.

---

### 6.6 Citation (API response shape)

What the API returns for each cited source. Two flavors — document or connector.

**Document citation:**

```json
{
  "source_type": "document",
  "chunk_id":    "uuid",
  "filename":    "assembly_drawing.pdf",
  "page":        12,
  "snippet":     "short text excerpt from the chunk",
  "image_path":  "/images/<doc_id>/<block_id>.jpg",
  "table_data":  null
}
```

**Connector citation:**

```json
{
  "source_type": "connector",
  "tool":        "db_query",
  "snippet":     "JSON or text excerpt of the connector's response"
}
```

`image_path` is non-null only for image caption chunks. `table_data` is non-null only for table chunks. The frontend checks `source_type` first, then which optional fields are present, and renders accordingly.

**Note on `image_path` format.** Citations carry the URL path served by `GET /images/{path:path}` (e.g. `/images/<doc_id>/<block_id>.jpg`), **not** the on-disk path. The frontend can use it directly: `<img src={citation.image_path}>`. The on-disk path lives only inside `metadata.image_path` on the source block.

---

## 7. Tool Reference

---

### Ingestion tools

| Tool | Reads from state | Writes to state | Libraries |
|---|---|---|---|
| `categorize_tool` | `file_path`, `file_type` | `route`, `document_type`, `industry`, `confidence` | fitz, Pillow, Gemma API |
| `page_profile_tool` | `file_path` | `page_profiles` | fitz, YOLO **(optional — falls back to fitz `page.get_image_info()` when YOLO is unavailable)** |
| `pdf_extraction_tool` | `file_path`, `page_profiles`, `filename` | appends to `blocks` | fitz, PaddleOCR |
| `embed_tool` | `chunks` | `chunks[*].vector` (dense) + `chunks[*].sparse_vector` (BM25) | sentence-transformers, fastembed |
| `excel_extraction_tool` | `file_path`, `filename` | appends to `blocks` (tables with `table_data`; charts saved to `raw_image_path`) | openpyxl, pandas, Pillow |
| `ppt_extraction_tool` | `file_path`, `filename` | appends to `blocks` (images + charts saved to `raw_image_path`) | python-pptx, Pillow |
| `index_tool` | `chunks` | writes to PostgreSQL + Qdrant (dense + sparse named vectors); **on first run, creates Qdrant payload indexes on `document_id`, `industry`, and `doc_type` so filtered hybrid search stays fast as the collection grows** | SQLAlchemy, qdrant-client |
| `image_extraction_tool` | `file_path`, `filename` | appends one `image` block with `raw_image_path` set | Pillow |
| `vision_enrichment_tool` | `blocks`, `page_profiles` | fills `text` and `metadata.image_path` on image blocks, appends caption blocks | fitz, Pillow, Gemma API, asyncio |
| `chunk_tool` | `blocks` | `chunks` (with `table_data` and `image_path` carried through, `token_count` set) | chonkie |
| `enrich_chunks_tool` | `chunks` | `chunks[*].tags` | llm_client — **batches N chunks per LLM call (configurable, default 8) instead of one-per-call; cuts enrichment latency and cost ~Nx on large documents** |

### Query tools

| Tool | Reads from state | Writes to state | Libraries |
|---|---|---|---|
| `query_planner_tool` | `query` | `sub_questions` (decompose only — no contextualize in Phase 1) | llm_client |
| `retrieval_tool` | `sub_questions`, `document_scope` | `retrieved_chunks` | qdrant-client, sentence-transformers |
| `rerank_tool` | `retrieved_chunks`, `query` | reordered `retrieved_chunks` | sentence-transformers CrossEncoder |
| `answer_tool` | `retrieved_chunks`, `connector_results`, `query` | `answer`, `citations` (with `image_path` + `table_data` + connector entries) | llm_client |

The agent node lives in the query graph itself (built with `create_react_agent`) — it is not a separate tool file. `answer_tool` is **not** registered as an agent tool; it is the next graph node and always runs after the agent exits. The agent's only tools in Phase 1 are the MCP connectors (just `db_query_tool`). A thin tool wrapper around each MCP connector also mirrors the connector's output into `state["connector_results"]` so `answer_tool` can read and cite them. On `max_iterations` exhaustion the graph force-routes to `answer_tool` with whatever state contains.

### Connector tools (MCP servers)

| Tool | Connects to | Phase 1 |
|---|---|---|
| `db_query_tool` | Client's database | Stub (the only one in Phase 1 — proves the pattern) |
| `inventory_tool` | Client's ERP / inventory API | **Phase 2** |
| `email_tool` | SMTP / SendGrid | **Phase 2** |
| `sms_tool` | SMS gateway | **Phase 2** |
| `teams_notify_tool` | Teams webhook | **Phase 2** |

### FastAPI endpoints

| Endpoint | What it does |
|---|---|
| `POST /upload` | Validates file size + extension, computes **SHA256 of file bytes** and checks the `documents.content_hash` index — if a `ready` document with the same hash already exists, returns its `document_id` instead of re-ingesting. Otherwise saves to `uploads/`, triggers ingestion graph, returns `document_id` |
| `GET /status/{document_id}` | Returns processing status and any errors for one document; surfaces `needs_review: true` when `confidence < 0.6` |
| `GET /documents` | Lists every document in the collection (filename, type, status, document_type, industry) |
| `DELETE /documents/{document_id}` | **Tombstone-and-sweep:** first marks the `documents` row `status='deleting'` (so it disappears from `GET /documents`), then deletes Qdrant points by `document_id` filter, then deletes Postgres `chunks` rows, then files from `uploads/`, then the `documents` row. A startup sweep also re-runs the deletion for any row left in `deleting` state — guarantees Postgres and Qdrant cannot diverge if the process crashes mid-delete. |
| `POST /query` | Accepts `{query, document_ids?}` — runs the query graph, returns answer + citations. `document_ids` optional; absent = whole collection |
| `GET /images/{path:path}` | Serves a stored image file from `uploads/images/`, with path-traversal protection |

`document_ids` on `/query` is optional: omit it (or send an empty list) to search the whole collection; send specific IDs only when the user narrowed the search. **Reprocess endpoint is Phase 2** — for now, delete and re-upload.

### Infrastructure

| Component | What it does |
|---|---|
| LangGraph ingestion graph | Assembles tool nodes from config, conditional routing after categorize, resolves the generic `extract` step to the right extractor by `file_type`, finalizes document row at end |
| LangGraph query graph | planner → retrieval → rerank → agent → answer_tool; agent calls connectors only; `answer_tool` is the terminal node and always runs after the agent; force-routes to `answer_tool` on iteration exhaustion |
| `llm_client.py` | Shared LLM wrapper with Langfuse tracing built in |
| `vision_client.py` | Shared `describe_image()` with Langfuse tracing + daily call-limit guard |
| `models.py` | Module-level singleton loaders for the local models (dense embedder, sparse, reranker) — one instance shared across tools |
| `tests/fixtures.py` | Shared test fixture (sample state, blocks, chunks, query response) — everyone tests against the same shapes |
| Docker Compose stack | Qdrant, PostgreSQL, Langfuse |
| `scripts/init_db.sql` | PostgreSQL schema: `documents`, `chunks` tables (no `conversations` in Phase 1) |

---

## 8. Observability

### Langfuse — LLM call tracing

Records every LLM call: prompt, response, token count, latency. Used to debug unexpected outputs and evaluate quality.

**What gets traced:**
- Gemma call in `categorize_tool`
- Gemma calls in `vision_enrichment_tool` — image sent, description returned
- LLM call in `enrich_chunks_tool` — chunk text, tags returned
- LLM call in `query_planner_tool` — question, sub-questions returned
- Agent node — every Think → Act → Observe loop iteration
- LLM call in `answer_tool` — full prompt, answer returned

**How:** Langfuse is wired as a callback in `llm_client.py` and `vision_client.py`. All tools that call `get_llm()` or `describe_image()` are traced automatically with no extra work elsewhere.

### System metrics

Deferred to Phase 3. No Grafana / Prometheus in Phase 1.

---

## 9. Local Dev Setup

### First-time setup

```bash
git clone https://github.com/praneethk666/AI-Accelerator.git
cd AI-Accelerator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
docker compose up -d
python tests/test_smoke.py
# Should print: smoke test passed
```

### Environment variables (`.env` — never commit this)

```
GEMMA_API_KEY=
GEMMA_API_ENDPOINT=

GROQ_API_KEY=               # or your chosen LLM provider

LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://localhost:3001

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=accelerator
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Folder structure

```
backend/
  core/           Shared shapes + helpers — schemas.py, tool.py, config.py, llm_client.py, vision_client.py, models.py
  pipeline/       LangGraph graphs — ingestion + query
  categorize/     categorize_tool
  connectors/     db_query MCP stub
  extraction/
    pdf/          page_profile_tool, pdf_extraction_tool
    excel/        excel_extraction_tool
    ppt/          ppt_extraction_tool
    image/        image_extraction_tool
  vision/         vision_enrichment_tool
  chunking/       chunk_tool
  enrichment/     enrich_chunks_tool
  embeddings/     embed_tool
  storage/        index_tool + delete_document
  retrieval/      retrieval_tool, rerank_tool
                  query_planner_tool, answer_tool
  api/            FastAPI endpoints
frontend/         Minimal web UI — upload + status, document list, ask
config/
  global.yaml
tests/
  fixtures.py
uploads/          Runtime — gitignored
  images/         Extracted visual regions — served by GET /images/{path}
docker-compose.yml
scripts/
  init_db.sql
```

### Code standards

- Format with `black`, lint with `ruff` before every push
- Type hints on every public function
- No magic numbers or model names in code — all settings come from config
- No secrets in code — keys come from `.env`
- Fail gracefully — append a structured `ErrorEntry` to `state["errors"]`, never raise and crash a document
- One tool per file, named by what it does
- Branch `feat/<yourname>-<task>` → Pull Request into `main`

### Model warm-up

FastAPI registers an `@app.on_event("startup")` hook that touches the `models.py` singleton loaders for **bge-large**, **bge-reranker-large**, and the **fastembed BM25** model. This forces the first model load to happen at process boot instead of on the first user query — without it, the very first `/query` pays a 5–15 s cold-start hit while sentence-transformers downloads/loads weights.

### Hardware requirements

- **CPU-only is workable for Phase 1** but the reranker is the bottleneck: bge-reranker-large on `top_n=20` candidates takes roughly **2–5 s per query** on a modern x86 CPU.
- **A single consumer GPU (≥ 8 GB VRAM)** drops reranker latency to well under 500 ms and is the recommended config for any demo to leadership or customer-facing pilot.
- bge-large embedding at ingest is comparatively cheap; it runs fine on CPU even for hundreds of documents.

### Retrieval evaluation (Phase 1 deliverable)

A **20-question gold set** (`tests/eval/gold_set.yaml`) is checked in alongside a fixed corpus of representative documents. `pytest tests/eval/test_retrieval.py` measures recall@5 and answer-citation accuracy. Before any retrieval-touching change is merged, the eval must run and the numbers must not regress. This is the only objective signal that "we improved retrieval" actually means something.

---

## 10. Design Decisions & Rationale

Every non-obvious choice in this plan, why it was made, and the alternative we did not pick. If you disagree with one, this is the place to push back — none of these are sacred, but each has a reason.

---

**Two databases — PostgreSQL *and* Qdrant.**
*Why:* They do different jobs. Qdrant does fast similarity search over vectors (the core of retrieval); Postgres is the durable source of truth, handles relational queries ("all chunks for this document", "sum the token counts"), holds the `documents` table and upload status, and stores large `table_data` that we keep out of the vector index. If Qdrant is ever rebuilt or we change embedding models, we re-embed from Postgres.
*Alternative:* Qdrant-only (put everything in its payload). Simpler, works for a demo, but loses the relational source-of-truth, status tracking, and a clean home for big table data. We chose the more production-sound split.

---

**Dense embeddings: `bge-large-en-v1.5` (1024-dim), not `all-MiniLM-L6-v2` (384-dim).**
*Why:* MiniLM is built for speed on small hardware; bge-large leads the MTEB retrieval benchmark for English. For enterprise documents a wrong retrieval becomes a wrong answer, so retrieval quality matters more than embedding speed.
*Alternative:* MiniLM (faster, smaller, less RAM). We traded speed for accuracy because accuracy is the product.

---

**Hybrid search (dense + BM25 sparse), not dense-only.**
*Why:* Dense vectors capture meaning but miss exact-match terms — part numbers, error codes, reference designators (`PM-2245`, `R1`, `5V_RAIL`). BM25 keyword search nails those. Fusing both with RRF gets the best of each.
*Alternative:* Dense-only (simpler, one vector). Would miss exact-identifier queries — unacceptable for technical docs.

---

**Semantic + token-aware chunking (chonkie), not fixed-size character splitting.**
*Why:* (1) Splitting by character count cuts mid-sentence and breaks meaning; semantic chunking splits where the topic actually shifts. (2) LLMs count tokens, not characters; measuring in tokens lets us pack the answer prompt to a known budget.
*Alternative:* `RecursiveCharacterTextSplitter` (simple, popular). Cheaper but blunter; we kept it as the `recursive` fallback for short/structured content.

---

**Vision-first for CAD / circuit diagrams, not OCR.**
*Why:* These arrive as PDFs but OCR returns almost nothing useful from a technical drawing — the meaning is in the layout, symbols, and connections. A vision model can describe what the drawing actually shows.
*Alternative:* OCR + text extraction (our default PDF path). Produces noise, not content.

---

**Routes chosen by a config table (`type_to_route`), not hardcoded logic.**
*Why:* This is what makes one engine serve many clients. Gemma only *labels* the document; a YAML lookup maps the label to a route.
*Alternative:* `if document_type == "invoice": ...` in code. Every new client or doc type would mean a code change.

---

**Everything is a tool that reads/writes shared state; tools never call each other.**
*Why:* Decoupling. A tool only depends on the *shape* of the data in `state`, not on another tool's internals. You can add, remove, reorder, or replace any tool by editing config — without touching the others.
*Alternative:* Tools calling tools directly. Faster to write at first, but every change ripples and the pipeline can't be reconfigured.

---

**Ingestion is deterministic; only the query side is agentic.**
*Why:* Ingestion should be predictable and debuggable. Query time genuinely needs flexibility (call a connector, or answer directly), so that is where the one agent node lives.

---

**Connectors built as MCP servers, not inline functions.**
*Why:* The agent calls them through a standard protocol, so the same connector is reusable by any MCP-compatible client later. Phase 1 ships one (`db_query`) to prove the pattern; the rest land in Phase 2.

---

**Local embeddings + reranker; only the vision model and text LLM are external APIs.**
*Why:* The data-boundary rule. Files, text, vectors, and databases stay on our infrastructure; only minimal, downscaled payloads go to external model APIs.

---

**One shared document collection per deployment, no accounts.**
*Why:* One deployment serves one client, so a single shared pool of documents matches reality and removes a whole layer of complexity. Questions search the whole collection by default, with optional narrowing.
*Alternative:* Per-user accounts / projects from day one. Real product need eventually, but it would slow Phase 1 down for no demo value. Deferred to Phase 2.

---

**Single-shot Q&A in Phase 1, not multi-turn chat.**
*Why:* Multi-turn (the "and part Y?" experience) needs a conversation store, a contextualize step in the query planner, and session plumbing through the API. None of that is hard, but it's not on the critical path to demonstrating the core value — *cited answers from your documents with the visuals shown*. Phase 1 ships that minimum and adds conversation in Phase 2.
*Alternative:* Multi-turn from day one. Pushed back to keep Sprint 1 achievable within the available team capacity.

---

**No corrective-RAG retry (CRAG judge) in Phase 1.**
*Why:* Hybrid search + a high-quality reranker is already a strong baseline. Adding a quality-judge + retry loop doubles the LLM cost per query and adds a graph cycle for marginal gain on the first pass. If real users hit poor-retrieval cases, add it in Phase 2 with proper telemetry.

---

**No adaptive router in Phase 1.**
*Why:* The "if everything fits, skip retrieval" optimization only matters at very small corpus sizes (a handful of docs). Always-retrieve is simpler, removes a graph branch, and removes the token-counting query against Postgres on every question. Add it back in Phase 2 if profiling shows it matters.

---

**HyDE off (Phase 2 toggle).**
*Why:* HyDE (encoding a hypothetical answer instead of the question) helps when query wording is far from document wording — but it adds one LLM call per sub-question, every query. Phase 1 skips it; Phase 2 can A/B it per client.

---

**Minimal web UI in Phase 1, not API-only and not a full app.**
*Why:* The product is "ask your documents" — that has to be *seen* to be believed, especially the images and tables in answers. A thin UI (upload + status, document list, ask with citations) demonstrates the whole system end to end.

---

## 11. What Comes Next — Phase 2

These capabilities are not in Sprint 1. They are listed here so the Phase 1 design leaves room for them.

---

### Deferred from Phase 1

- **Multi-turn chat (contextualize + conversation store).** Adds a `conversations` table, a contextualize step in the planner, and `session_id` plumbing through `/query`. Lets follow-ups like "and part Y?" work in context.
- **Streaming `answer_tool`.** Stream tokens of the final answer back through `/query` (SSE or WebSocket) so the UI shows progress instead of a 3–8 s blank wait. Phase 1 returns the answer as a single JSON response.
- **Real task queue.** Replace FastAPI `BackgroundTasks` with **arq / rq / Celery** so ingestion survives API restarts, gets retries, and scales to a worker pool. Phase 1 in-process background tasks are fine for the demo but lose work on a crash.
- **Adaptive router.** Skip retrieval when the whole in-scope corpus fits the context budget.
- **CRAG judge + retry.** Re-retrieve once when the first pass is judged insufficient. Wider `top_n` and dropped optional filters on retry.
- **HyDE in retrieval.** Per-client toggle for hypothetical-answer embedding.
- **Expanded retrieval eval harness.** Grow the Phase 1 gold set from 20 to 100+ questions, add nightly run with trend tracking.
- **Reprocess endpoint.** `POST /documents/{id}/reprocess` — re-run ingestion for a failed or stale document, with a status-guard to refuse if already processing.
- **The other 4 MCP connectors.** `inventory`, `email`, `sms`, `teams_notify` — interface and wiring.
- **Per-client YAML profiles.** `config/profiles/<client>.yaml` overrides on top of `global.yaml`.
- **Grafana + Prometheus.** System metrics with starter dashboards.
- **CI.** GitHub Actions running `black --check && ruff check && pytest`.
- **`DELETE /sessions/{id}`** — wipe a conversation (depends on multi-turn landing first).
- **`GET /documents/{id}/chunks`** — debug endpoint to inspect what got indexed.

### Cross-Document Reasoning

**The problem it solves.**

Today the system retrieves chunks and answers from them. But real questions often span multiple documents. A Toyota engineer asks: *"What are the wiring specs for the power module in the assembly drawing?"*

A human would:
1. Find the assembly drawing section showing the power module
2. Read the part number from the diagram (e.g. PM-2245)
3. Open the related spec sheet or BOM and look up PM-2245
4. Combine what they found from both documents into one answer

The current system does step 1 well. It cannot do steps 2 and 3 — it cannot follow a reference from one document into another.

**What this requires.**

- **Entity extraction during enrichment.** The `enrich_chunks_tool` extracts named entities from each chunk — part numbers, component IDs, drawing references. Stored as an `entities` tag.
- **Cross-reference index.** A PostgreSQL table linking entity values to every chunk that mentions them.
- **Multi-hop agent retrieval.** The agent makes a first retrieval → extracts entities from results → makes a second retrieval using those entities → synthesises an answer from both.

**Why Phase 1 supports this.** The `tags` JSONB field already exists — adding `entities` costs nothing. The agent loop already supports multiple tool calls. The retrieval tool already supports tag filtering.

### Auth / multi-user / workspaces

`conversations.user_id` migration, per-user document scoping, projects/workspaces, admin screens. None of this is in Phase 1.

---

*Last updated: 2026-06-04 (Plan v1.4)*
