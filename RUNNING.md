# Running the Accelerator locally

End-to-end: ingest a document (via the agent chat or the ingestion page) →
it's categorized, extracted, chunked, embedded, indexed → ask questions and
get cited answers.

**Two frontend pages, easy to mix up:**
- **http://localhost:5173/chat** — the agent chat (ChatGPT/Claude-style: sidebar +
  history, attach a file to ingest it, ask questions). **This is the one to use
  for testing the agent.**
- **http://localhost:5173/** — the older, separate Ingestion page (drag-drop
  upload only, no chat, no agent). Landing on `/` and expecting the chat UI is
  a common mix-up — there's no link between the two pages yet.

## 0. Prerequisites
- Docker running, Python venv at `.venv` (deps installed), Node 18+ for the UI.

## Quick start (copy-paste, 3 terminals)
```bash
# 1. Data plane
cd /Users/contenterra/Projects/python_projects/AI-Accelerator
docker compose up -d

# 2. Backend (own terminal — leave running)
source .venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000
# first startup can take 30-60s while it warms the embedding/reranker models

# 3. Frontend (own terminal — leave running)
cd frontend && npm run dev
```
Then open http://localhost:5173/chat.

## 1. Data plane (Postgres + Qdrant)
```bash
docker compose up -d postgres qdrant
docker compose ps                     # both should be Up / healthy
```
- Postgres `:5432` (db `accelerator`, user/pw `postgres/postgres`) — tables auto-created.
- Qdrant `:6333`.
- Reset everything (wipes data, re-runs init_db.sql): `docker compose down -v`
- Optional add-ons live in their own compose files — combine with `-f`:
  - DB browser (Adminer, `:8080`): `docker compose -f docker-compose.yml -f docker-compose.devtools.yml up -d`
  - Tracing/metrics (Langfuse `:3001`, Grafana `:3000`): `docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d`

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

## 5. Use it

**Agent chat (http://localhost:5173/chat) — the main way to test this:**
1. **New chat** (top of sidebar) starts a fresh conversation.
2. Click the paperclip, pick a file. It stages on the server (not ingested yet).
3. Type something (or leave blank — it defaults to "please ingest this") and send.
   The agent proposes calling `ingest_document` and shows an **Approve/Decline**
   card — nothing runs until you approve.
4. Approve → all pipeline steps run inline (a few seconds to ~a minute
   depending on the file) → it reports back with the document_id and status.
5. Ask a question in the same or a new chat → the agent calls `search_documents`
   and answers with citations (filename + page, shown as chips under the answer).
6. Past conversations persist in the sidebar (Postgres-backed) — click one to
   reopen it, even after restarting the backend.

**Ingestion page (http://localhost:5173/) — the older, simpler alternative:**
drag in a PDF / Excel / PPT, no agent/approval step, ingestion just runs. Useful
for bulk-loading files without going through chat. The card shows route,
document type, industry, confidence, and status (processing → ready).

## Endpoints (frontend ↔ API)
| UI action | Method | Endpoint |
|---|---|---|
| Upload (direct, auto-ingest) | POST | `/upload` |
| Stage a file for the agent (no auto-ingest) | POST | `/files/stage` |
| List documents | GET | `/files` |
| One document | GET | `/files/{id}` |
| Delete | DELETE | `/files/{id}` |
| Ask (agent picks a tool) | POST | `/agent/chat` |
| List agent chat sessions | GET | `/agent/sessions` |
| One session's history | GET | `/agent/sessions/{id}` |
| Delete a session | DELETE | `/agent/sessions/{id}` |
| Health | GET | `/health` |
| Images | GET | `/images/<doc>/<file>` (static) |

CLI alternative to the chat UI: `python scripts/agent_chat.py "<message>"`
(one-shot) or with no args (interactive, keeps memory). See the root README's
"Agent" section for the full flow and how the write-approval gate works.

## Troubleshooting
- **`database unavailable`** → `docker compose ps`; bring Postgres up.
- **Empty/`An error occurred` answer** → no LLM key, or no documents ingested yet.
- **Vision skipped** → set `GOOGLE_API_KEY` (or switch `vision.provider`); ingestion
  still works without it (visuals just aren't captioned).
- **Schema changed** (e.g. after a pull) → `docker compose down -v && docker compose up -d postgres qdrant`.
