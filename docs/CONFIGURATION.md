# Configuration Guide — setting up extraction for your documents

This explains how the pipeline decides what to do with a file, and which knobs to
turn for a given document set. Secrets live in `.env`; everything else lives in a
YAML config (`config/global.yaml` by default).

---

## 1. How the extraction path is chosen (you don't hard-code it)

When a file is uploaded, the `categorize` step inspects it and sets three things on
the pipeline state:

- `file_type` — `pdf` | `excel` | `ppt` | `image` (from the extension)
- `pdf_kind` — for PDFs only: `digital` | `scanned` | `mixed` (from `detector.py`)
- `route` — from `document_type` via `type_to_route` (e.g. `report → text_default`)

The graph then picks **exactly one extractor**, in this priority order:

| Priority | Config map | Keyed by | Example |
|---|---|---|---|
| 1 | `route_extractors` | `route` | `cad_route → cad_extract` (route overrides everything) |
| 2 | `pdf_extractors` | `pdf_kind` | `digital → pdf_digital`, `scanned → scanned_pdf`, `mixed → mixed_pdf` |
| 3 | `extractors` | `file_type` | `excel → excel_extraction`, `ppt → ppt_extraction` |

After extraction, **`route_gates`** decide which optional steps run for that route
(today only `vision_enrichment` is gated). Everything else in `ingestion.steps`
runs for every route.

> So "configuring extraction" = editing these maps + the per-tool settings blocks
> (`ocr`, `vision`, `chunking`). You rarely touch code.

---

## 2. Per-customer / per-deployment config

Don't fork the code — fork the YAML.

1. Copy the base: `cp config/global.yaml config/argo.yaml`
2. Edit `config/argo.yaml` for that customer's documents.
3. Run the API against it:
   ```bash
   CONFIG_PATH=config/argo.yaml uvicorn backend.api.main:app --reload --port 8000
   ```

`CONFIG_PATH` (read in `backend/api/main.py`) is the only thing that changes. Each
customer/deployment gets its own profile; the code is identical.

---

## 3. The knobs that matter (by document type)

### OCR — for scanned pages (`ocr.engine`)
```yaml
ocr:
  engine: paddle            # paddle | surya  (runs in an isolated subprocess)
  surya_timeout_s: 90       # per-page Surya cap; on stall/error the page uses paddle
  subprocess_timeout_s: 1800  # whole-document OCR cap before the child is killed
```
- **paddle** (default): fast CNN OCR + DocLayout-YOLO/contour region detection.
  Reliable on CPU — the right default for dev/Mac and large scanned sets.
- **surya**: one pass gives text + layout, 90+ languages, best accuracy on complex
  layouts. Needs the `llama-server` binary (`brew install llama.cpp`). Slower on CPU
  and can stall over long docs — **switch to `surya` on a GPU box**, where
  `llama-server` runs on-GPU.

OCR runs in an **isolated subprocess** (`scanned_pdf/ocr_worker.py`): the native OCR
stack (Paddle / Torch-Surya / llama-server) can't crash the backend, and a child
crash/timeout just fails that document gracefully. Engine choice is per deployment —
set it in the active config or a per-customer profile.

### Vision — captions for images/diagrams (`vision`)
```yaml
vision:
  provider: google          # google | ollama | openai (any multimodal endpoint)
  model: gemma-3-27b-it
  api_key: ${GOOGLE_API_KEY} # set the real key in .env
  enabled: true
  dpi: 150                  # higher = sharper crops, slower
  max_concurrency: 4        # parallel calls per document
```
Vision is **required** for two things: good categorize confidence, and captioning
diagrams in digital/scanned PDFs. Without a working key, confidence falls back to a
keyword heuristic and images are not captioned.

### Vision on/off per route (`ingestion.route_gates`)
```yaml
ingestion:
  route_gates:
    vision_enrichment: [text_default, diagram_heavy, image_route]
```
`text_default` is included on purpose — most real documents (manuals, reports) land
there and have diagrams. CAD/circuit routes are excluded because their extractor
(`cad_extract`) does its own vision pass.

### Chunking (`chunking`)
```yaml
chunking:
  strategy: semantic   # semantic | recursive
  size: 400            # tokens per chunk
  overlap: 50          # tokens shared between consecutive chunks
```
Tables and image captions are **atomic** (one chunk each, never split) so citations
stay intact. Only body text is windowed. Larger `size` = fewer, broader chunks;
smaller = more precise but more vectors.

### Routing (`type_to_route`, `extractors`, `pdf_extractors`, `route_extractors`)
Change these only to add a document type or a new extractor. The values must match a
tool's `name` (see `backend/extraction/*/tool.py`).

---

## 4. What each step produces (the data flow)

```
categorize  → file_type, pdf_kind, route, industry, confidence
extract     → blocks[]  (text / heading / table / image_caption), page_profiles[]
vision      → fills image_caption blocks with descriptions + image_path
chunk       → chunks[]  (atomic tables/images; windowed text; carries source_ref)
enrich      → chunk.tags (industry, doc_type, keywords)
embed       → chunk.vector (dense 768) + chunk.sparse_vector (BM25)
index       → Postgres: full text + table_data + source_ref + image_path
              Qdrant:   vectors + {chunk_id, document_id, tags}  (text hydrated from PG by chunk_id)
```

Tables: the markdown form of the table is what gets **embedded** (so it's
searchable); the structured `table_data` is stored in Postgres for exact display.

---

## 5. Worked profile — ARGO equipment manuals

Mostly digital PDFs with diagrams + parts tables, plus some scanned ones.

```yaml
# config/argo.yaml (only the lines that differ from global.yaml)
ocr:
  engine: paddle      # big scanned manuals on CPU -> fast; switch to surya on a GPU box
vision:
  enabled: true       # diagrams/exploded views get captioned
  dpi: 150
  max_concurrency: 4
chunking:
  size: 400
  overlap: 50
default_industry: automotive   # ARGO = vehicles; biases industry tagging
```
Everything else (routing, extractors) is already correct: digital ARGO PDFs →
`text_default` → `pdf_digital` + vision; scanned ones → `scanned_pdf` + OCR.

---

## 6. .env checklist (per machine)

```ini
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/accelerator
QDRANT_URL=http://localhost:6333
GROQ_API_KEY=...        # LLM (answers)
GOOGLE_API_KEY=AIza...  # vision (categorize + image captions) — get from https://aistudio.google.com/apikey
```
Each person needs their own keys (free tiers are fine for testing). After editing
`.env`, restart the backend so it's picked up.

---

## 7. Verify your config quickly

```bash
# config loads + the route gate is what you expect
python -c "from backend.core.config import load_config; c=load_config('config/argo.yaml'); print(c['ocr'], c['ingestion']['route_gates'])"
# health + registered tools
curl -s localhost:8000/health
```
Then upload one document of each type from the UI and confirm it reaches `ready`.
