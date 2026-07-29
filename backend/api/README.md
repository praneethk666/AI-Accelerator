# API Module

The API module exposes the HTTP interface for the RAG & Document Intelligence Accelerator, serving as the interface between the client frontend and the backend graph pipelines.

## Technology Stack

The API is built using the following technologies:
* **FastAPI**: Main web framework for routing, request validation, and OpenAPI schema generation.
* **Uvicorn**: ASGI web server running a single-worker async event loop.
* **Pydantic**: Data validation and parsing using schemas (`BaseModel`).
* **psycopg**: PostgreSQL adapter for schema synchronization at startup.
* **FastAPI BackgroundTasks**: Used to run CPU-bound ingestion pipelines outside the main request thread.

## Server Startup & Initialization

When Uvicorn runs `backend.api.main:app`, the `@app.on_event("startup")` hook executes:
1. **Database Schema Sync**: Connects to Postgres using `psycopg` to verify connection state and auto-initialize tables (`documents`, `chunks`, `conversations`) using `scripts/init_db.sql` if they do not exist.
2. **Directory Mounts**: Creates and mounts `uploads/images/` and `uploads/pages/` using `StaticFiles` to serve cropped visual diagrams and page screenshots to the frontend.
3. **Model Warm-Up**: Invokes `backend.core.models.warm_up()` to download/load local models (dense, sparse, and reranker) with timeouts, preventing lazy-loading request stalls.

## API Endpoints

### 1. Ingestion & Documents
* **`POST /upload`**
  * *Parameters*: `file: UploadFile` (Multipart form-data).
  * *Logic*: Sniffs file extension via `EXT_TO_FILE_TYPE` to assign a standard `file_type`. Assigns a unique UUID `document_id`. Invokes `ingest_document()` inside a `BackgroundTasks` runner to execute the graph.
  * *Response*: Returns `document_id` and metadata immediately.
* **`POST /files/stage`**
  * *Logic*: Saves an uploaded file to `uploads/staged/{uuid}` without running ingestion. Used when attaching images or PDFs inside chat threads.
* **`GET /files`**
  * *Logic*: Instantiates `PostgresStore` to query the `documents` table, returning list profiles.
* **`GET /files/{id}`**
  * *Logic*: Retrieves metadata, status, error list, and logging metrics for a specific document.
* **`DELETE /files/{id}`**
  * *Logic*: Connects to Postgres (`PostgresStore`) to delete document rows and cascading chunk rows. Connects to Qdrant (`QdrantStore`) to delete vector points matching the `document_id`.
* **`GET /files/{id}/original`**: Returns the raw file bytes using FastAPI's `FileResponse`.
* **`GET /files/{id}/docx-html`**: Converts Word docx XML structures to clean HTML panels for direct rendering.

### 2. Conversational Agent
* **`POST /agent/chat`**
  * *Payload*: `{ "message": str, "session_id": Optional[str], "approved_writes": bool }`.
  * *Logic*: Invokes `run_agent()` (`backend.agent.executor.py`). If the agent model requests a write tool (e.g., `ingest_document`) and `approved_writes` is `false`, the API halts execution and returns a `needs_approval` response.
* **`GET /agent/sessions`**: Fetches active and historical conversation threads.
* **`GET /agent/sessions/{id}`**: Retrieves the complete multi-turn message history.
* **`DELETE /agent/sessions/{id}`**: Deletes a conversation session.
