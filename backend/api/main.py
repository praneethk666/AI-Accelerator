"""FastAPI app — the real ingestion + query pipeline behind the React UI.

Endpoints (match frontend/src/api.jsx):
    POST   /upload         multipart file -> run ingestion, return doc metadata
    GET    /files          list ingested documents
    GET    /files/{id}     one document's metadata
    DELETE /files/{id}     delete a document (chunks cascade)
    POST   /chat           {question, file_id?} -> {answer, sources}
    GET    /chat-history   recent turns for the web session
    GET    /health         liveness

This wires the WHOLE pipeline (categorize -> extract -> ... -> index for ingest;
plan -> retrieve -> answer for chat), not just categorization. Running it needs
the stack up (Postgres, Qdrant) and models available — see docker-compose + .env.

Run:  uvicorn backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid

from dotenv import load_dotenv

# Load .env BEFORE load_config so ${GROQ_API_KEY}/${POSTGRES_URL}/... resolve.
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend.core.config import PipelineConfig, load_config  # noqa: E402
from backend.core.models import warm_up  # noqa: E402
from backend.pipeline.default_registry import build_default_registry  # noqa: E402
from backend.pipeline.graph import run_pipeline  # noqa: E402
from backend.pipeline.query import run_query  # noqa: E402
from backend.storage.postgres_store import PostgresStore  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/global.yaml")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
WEB_SESSION = "web"  # the single-session the browser UI uses for chat history

EXT_TO_FILE_TYPE = {
    ".pdf": "pdf",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".pptx": "ppt", ".ppt": "ppt",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tif": "image", ".tiff": "image",
}

app = FastAPI(title="Document Intelligence + RAG Accelerator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve enriched images so citation image_path (/images/<doc>/<file>) resolves.
# vision writes them under uploads/images/<doc_id>/<block_id>.png.
_IMAGES_DIR = os.path.join(UPLOAD_DIR, "images")
os.makedirs(_IMAGES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=_IMAGES_DIR), name="images")

# Loaded once at import; the registry caches model singletons across requests.
_config = load_config(CONFIG_PATH)
# Warm torch/nomic BEFORE anything can import paddle (paddleocr) — paddle-first
# corrupts torch's allocator. paddleocr is lazy-imported, so this ordering holds.
warm_up(_config)
_registry = build_default_registry()
_ingestion_cfg = PipelineConfig.from_dict({
    **_config,
    "steps": _config.get("ingestion", {}).get("steps", []),
    "route_gates": _config.get("ingestion", {}).get("route_gates", {}),
})


class ChatRequest(BaseModel):
    question: str
    file_id: str | None = None


def _file_type(filename: str) -> str:
    return EXT_TO_FILE_TYPE.get(os.path.splitext(filename)[1].lower(), "unknown")


def _pg() -> PostgresStore:
    try:
        return PostgresStore()
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {exc}") from exc


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"{document_id}_{file.filename}")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_type = _file_type(file.filename)
    pg = _pg()
    try:
        pg.insert_document(document_id, file.filename, file_type, dest)
    finally:
        pg.close()

    state = {
        "document_id": document_id,
        "file_path": dest,
        "file_type": file_type,
        "errors": [],
    }
    try:
        result = run_pipeline(_registry, state, _ingestion_cfg)
        status = "ready" if not result.get("errors") else "failed"
    except Exception as exc:
        logger.exception("ingestion failed for %s", file.filename)
        result, status = {"errors": [str(exc)]}, "failed"

    pg = _pg()
    try:
        pg.finalize_document(
            document_id,
            document_type=result.get("document_type"),
            industry=result.get("industry"),
            route=result.get("route"),
            confidence=result.get("confidence"),
            status=status,
            errors=result.get("errors"),
        )
        doc = pg.get_document(document_id)
    finally:
        pg.close()
    # surface per-step timings + any errors for observability
    doc["metrics"] = result.get("metrics", [])
    doc["errors"] = result.get("errors", [])
    return doc


@app.get("/files")
async def list_files():
    pg = _pg()
    try:
        return pg.list_documents()
    finally:
        pg.close()


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    pg = _pg()
    try:
        pg.delete_document(file_id)
    finally:
        pg.close()
    return {"deleted": file_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    scope = [req.file_id] if req.file_id else []
    final = run_query(
        req.question, _registry, _config,
        session_id=WEB_SESSION, document_scope=scope,
    )
    # frontend expects {answer, sources:[{filename, page, snippet}]}
    sources = [
        {"filename": c.get("filename"), "page": c.get("page"), "snippet": c.get("snippet")}
        for c in (final.get("citations") or [])
    ]
    return {
        "answer": final.get("answer", ""),
        "sources": sources,
        "metrics": final.get("metrics", []),   # per-step timings (observability)
    }


@app.get("/chat-history")
async def chat_history():
    try:
        from backend.storage.conversation_store import get_conversation_store

        return get_conversation_store().load_history(WEB_SESSION, n=50)
    except Exception as exc:
        logger.debug("chat history unavailable: %s", exc)
        return []


@app.get("/health")
async def health():
    return {"status": "ok", "tools": sorted(_registry.names())}


@app.get("/")
async def root():
    return {"service": "Document Intelligence + RAG Accelerator", "docs": "/docs"}
