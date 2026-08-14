# FastAPI Backend API Subsystem

The **API Module** (`backend/api/`) exposes the high-performance REST HTTP layer connecting the React frontend, background ingestion workers, and conversational agent workflows.

---

## 1. Key Capabilities & Features

- **Non-Blocking Asynchronous Processing**:
  - Offloads CPU- and I/O-bound document ingestion pipelines to FastAPI `BackgroundTasks`, returning immediate document IDs and metadata to the client.
- **Agentic Conversational Gateway**:
  - Serves `/agent/chat` routing all document intelligence and retrieval actions through the LangGraph agent executor with human-in-the-loop write approvals.
- **Noise-Filtered Access Logging**:
  - Custom Uvicorn `EndpointFilter` suppresses noisy polling requests (`/progress`, `/health`, `/pages/.../image`, `/pdf-info`) to keep server terminal outputs clean and readable.
- **Dynamic Configuration & Cost Metrics**:
  - Endpoints (`/config`, `/metrics/costs`) allowing live round-trip YAML configuration updates and multi-model token cost queries with real-time USD/INR conversions.
- **Static Artifact Streaming**:
  - Mounts `uploads/images/` and `uploads/pages/` via `StaticFiles` for sub-millisecond diagram and page preview rendering.

---

## 2. API Endpoints Reference

### 1. Ingestion & Document Management
| Method | Endpoint | Description | Request / Response Payload |
|---|---|---|---|
| `POST` | `/upload` | Uploads and kicks off asynchronous pipeline ingestion. | Multipart file $\rightarrow$ `{ document_id, filename, file_type, status }` |
| `POST` | `/files/stage` | Stages file to `uploads/staged/` without ingestion (for chat attachments). | Multipart file $\rightarrow$ `{ file_path, filename }` |
| `GET` | `/files` | Lists all ingested documents with page counts and metadata. | `[]` $\rightarrow$ `list[DocumentSummary]` |
| `GET` | `/files/{id}` | Fetches detailed document status, logs, and error traces. | URL parameter $\rightarrow$ `DocumentDetail` |
| `DELETE`| `/files/{id}` | Deletes a document, cascading chunk rows in Postgres and vector points in Qdrant. | URL parameter $\rightarrow$ `{ status: "deleted" }` |
| `GET` | `/files/{id}/original` | Streams raw file bytes (PDF/Image) for direct browser preview. | URL parameter $\rightarrow$ Raw File Bytes |
| `GET` | `/files/{id}/docx-html` | Converts Word `.docx` XML structures to clean HTML for direct in-panel rendering. | URL parameter $\rightarrow$ `{ html: "..." }` |

### 2. Conversational Agent & Sessions
| Method | Endpoint | Description | Request / Response Payload |
|---|---|---|---|
| `POST` | `/agent/chat` | Main conversational endpoint invoking the LangGraph agent loop. | `{ message: str, session_id?: str, approved_writes?: bool }` $\rightarrow$ `{ answer, citations, status, trace, token_usage, costs }` |
| `GET` | `/agent/sessions` | Lists historical conversation sessions for the sidebar. | `[]` $\rightarrow$ `list[SessionSummary]` |
| `GET` | `/agent/sessions/{id}` | Fetches the complete message history for an active thread. | URL parameter $\rightarrow$ `{ session_id, messages: [...] }` |
| `DELETE`| `/agent/sessions/{id}` | Deletes a conversation session from the database. | URL parameter $\rightarrow$ `{ status: "deleted" }` |

### 3. System & Configuration
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness health check returning service status. |
| `GET` | `/config` | Returns active `global.yaml` settings dictionary. |
| `PUT` | `/config` | Applies round-trip in-place YAML settings updates. |
| `GET` | `/metrics/costs` | Returns aggregated token usage and cost metrics (USD and INR). |

---

## 3. Server Startup & Lifecycle

When Uvicorn launches `backend.api.main:app`:
1. **Database Schema Sync**: Auto-runs `scripts/init_db.sql` via `psycopg` to ensure relational tables exist.
2. **Directory Initialization**: Creates and mounts `uploads/images/`, `uploads/pages/`, and `uploads/staged/`.
3. **Model Warm-Up**: Executes `backend.core.models.warm_up()` with bounded timeouts to pre-warm dense, sparse, and reranker singletons.

---

## 4. Running & Testing

```powershell
# Launch FastAPI development server
uvicorn backend.api.main:app --reload --port 8000

# Verify API and smoke tests
pytest tests/test_smoke.py
```
