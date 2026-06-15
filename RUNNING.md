# Running the Accelerator locally (Phase 1)

End-to-end: upload a document in the UI → it's categorized, extracted, chunked,
embedded and indexed → ask questions and get cited answers.

## 0. Prerequisites
- Docker running, Python venv at `.venv` (deps installed), Node 18+ for the UI.

## 1. Data plane (Postgres + Qdrant)
```bash
docker compose up -d postgres qdrant
docker compose ps                     # both should be Up / healthy
```
- Postgres `:5432` (db `accelerator`, user/pw `postgres/postgres`) — tables auto-created.
- Qdrant `:6333`.
- Reset everything (wipes data, re-runs init_db.sql): `docker compose down -v`
- Optional tracing/metrics: `docker compose up -d langfuse grafana` (3001 / 3000).

## 2. Keys (`.env` at repo root — gitignored)
DB URLs are already filled in. Add the keys for whichever provider you use:
```ini
GROQ_API_KEY=...        # console.groq.com (default LLM)
GOOGLE_API_KEY=...      # aistudio.google.com (default vision: Gemma)
```
Switching provider = edit the `llm:` / `vision:` block in `config/global.yaml`.
Recipes (Groq / Gemini-Vertex / NVIDIA NIM / OpenRouter / local vLLM) are documented
inline there. `openai` + `base_url` reaches any OpenAI-compatible API.

## 3. Backend API
```bash
.venv/bin/uvicorn backend.api.main:app --reload --port 8000
```
- Health: http://localhost:8000/health  → `{"status":"ok","tools":[...15...]}`
- API docs (try endpoints directly): http://localhost:8000/docs

## 4. Frontend
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

## 5. Use it (all from the UI)
1. **Ingestion page** → drag in a PDF / Excel / PPT. The card shows route,
   document type, industry, confidence, and status (processing → ready).
2. Click a ready document → **Chat** → ask a question. Answer comes back with
   source citations (filename + page/sheet/slide + snippet); image citations render
   from `/images/...`.

## Endpoints (frontend ↔ API)
| UI action | Method | Endpoint |
|---|---|---|
| Upload | POST | `/upload` |
| List documents | GET | `/files` |
| One document | GET | `/files/{id}` |
| Delete | DELETE | `/files/{id}` |
| Ask | POST | `/chat` |
| Health | GET | `/health` |
| Images | GET | `/images/<doc>/<file>` (static) |

## Troubleshooting
- **`database unavailable`** → `docker compose ps`; bring Postgres up.
- **Empty/`An error occurred` answer** → no LLM key, or no documents ingested yet.
- **Vision skipped** → set `GOOGLE_API_KEY` (or switch `vision.provider`); ingestion
  still works without it (visuals just aren't captioned).
- **Schema changed** (e.g. after a pull) → `docker compose down -v && docker compose up -d postgres qdrant`.
