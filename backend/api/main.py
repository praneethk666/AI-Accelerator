"""FastAPI app — the real ingestion + query pipeline behind the React UI.

Endpoints (match frontend/src/api.jsx):
    POST   /upload         multipart file -> run ingestion, return doc metadata
    POST   /files/stage    multipart file -> save only (no ingest) -> {file_path}, for chat attach
    GET    /files          list ingested documents
    GET    /files/{id}     one document's metadata
    DELETE /files/{id}     delete a document (chunks cascade)
    POST   /agent/chat     {message, session_id?, approved_writes?} -> agent picks a tool
                           (ingest_document / search_documents / list_documents / sql_read);
                           writes need approval
    GET    /agent/sessions           list past agent-chat conversations (sidebar)
    GET    /agent/sessions/{id}      one conversation's full turn history
    DELETE /agent/sessions/{id}      delete a conversation
    GET    /health         liveness

Everything document-facing goes through the agent now — there is no direct
(non-agentic) RAG endpoint. search_documents is just another tool the agent
calls; it is not exposed as its own HTTP route.

This wires the WHOLE pipeline (categorize -> extract -> ... -> index for ingest;
plan -> retrieve -> answer for chat), not just categorization. Running it needs
the stack up (Postgres, Qdrant) and models available — see docker-compose + .env.

Run:  uvicorn backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import uuid

import yaml
from dotenv import load_dotenv

# Load .env BEFORE load_config so ${GROQ_API_KEY}/${POSTGRES_URL}/... resolve.
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend.agent.executor import run_agent  # noqa: E402
from backend.agent_tools import build_agent_registry  # noqa: E402
from backend.core.config import load_config  # noqa: E402
from backend.core.models import warm_up  # noqa: E402
from backend.core.llm_client import clean_message_content  # noqa: E402
from backend.pipeline.default_registry import build_default_registry  # noqa: E402
from backend.pipeline.ingest import ingest_document  # noqa: E402
from backend.storage.postgres_store import PostgresStore  # noqa: E402
from backend.storage.qdrant_store import QdrantStore  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/global.yaml")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

EXT_TO_FILE_TYPE = {
    ".pdf": "pdf",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".pptx": "ppt", ".ppt": "ppt",
    ".docx": "docx", ".doc": "docx",
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

# Serve full-page images (visual grounding). Rendered at ingest for PDF pages that
# produced chunks; stored under uploads/pages/<doc_id>/p{N}.jpg, served at /pages/...
_PAGES_DIR = os.path.join(UPLOAD_DIR, "pages")
os.makedirs(_PAGES_DIR, exist_ok=True)
app.mount("/pages", StaticFiles(directory=_PAGES_DIR), name="pages")

# Loaded once at import; the registry caches model singletons across requests.
_config = load_config(CONFIG_PATH)
# Warm torch/nomic BEFORE anything can import paddle (paddleocr) — paddle-first
# corrupts torch's allocator. paddleocr is lazy-imported, so this ordering holds.
warm_up(_config)
# The OCR engine (surya|paddle, config `vision_ocr`/scanned-path settings) is NOT
# warmed here — only the warm_up() ordering above is what prevents the
# paddle/torch allocator crash. (An earlier version of this comment described
# running OCR in an isolated subprocess; that isolation does not exist in the
# current code — verify the crash risk is still mitigated before changing
# warm_up() ordering or the OCR call path.)
_registry = build_default_registry()
_agent_registry = build_agent_registry()
# In-memory per-session LangChain message history for the agent chat — good enough
# for a demo; resets on restart. Only advanced past a turn that fully completed
# (not one awaiting write approval), so a decline/retry replays cleanly.
_agent_sessions: dict[str, list] = {}

# search_documents/ingest_document results can be large (full citation snippets,
# table_data, image paths...), and every intermediate tool-call round-trip adds
# more messages on top. tools_node needs the FULL detail for the current turn
# (it's what the frontend's citation strip parses from tool_calls[].result), but
# once the turn is done we only cache the plain Q&A — not the tool calls/results
# or the system prompt — for replay as conversation_history on the NEXT question.
# The model only needs to remember what was asked and answered, not how it got
# there; re-sending full tool payloads on every later turn is exactly what was
# blowing through Groq free-tier TPM/TPD limits after just one or two follow-ups.
# Same simplification _history_to_messages already applies to history reloaded
# from Postgres — this just applies it to the in-memory cache too.
def _qa_only(messages: list) -> list:
    return [
        m for m in messages
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and clean_message_content(m.content).strip())
    ]



CONFIG_DIR = os.path.dirname(CONFIG_PATH) or "config"


def _reload_pipeline() -> None:
    """Re-read CONFIG_PATH and rebuild the pipeline objects in place after a config
    edit. Models are NOT re-warmed (routing/prompt/OCR/chunking edits don't change
    the embedding/vision models); to switch those, edit + restart the server.
    ingest_document derives the ingestion profile from _config on each call, so the
    reloaded _config is all it needs — nothing else to rebuild here."""
    global _config, _registry
    _config = load_config(CONFIG_PATH)
    # OCR engine/timeout are read from config by the isolated OCR subprocess at
    # ingest time, so there's nothing to set in-process here.
    _registry = build_default_registry()


class AgentChatRequest(BaseModel):
    message: str
    session_id: str = "web"
    approved_writes: bool = False


class ConfigSave(BaseModel):
    yaml: str


class ProfileSave(BaseModel):
    name: str
    yaml: str


class ProfileActivate(BaseModel):
    name: str


class SettingsSave(BaseModel):
    settings: dict
    save_as: str | None = None   # optional: write to a new profile instead of active


# Flat form-field -> nested config path. The Settings UI edits these; everything
# else stays as-is in the YAML.
_SETTINGS_MAP = {
    "default_industry": ["default_industry"],
    "pdf_extractor_digital": ["pdf_extractors", "digital"],
    "ocr_engine": ["ocr", "engine"],
    "llm_provider": ["llm", "provider"],
    "llm_model": ["llm", "model"],
    "vision_provider": ["vision", "provider"],
    "vision_model": ["vision", "model"],
    "vision_enabled": ["vision", "enabled"],
    "chunking_strategy": ["chunking", "strategy"],
    "chunking_size": ["chunking", "size"],
    "chunking_overlap": ["chunking", "overlap"],
    "enrichment_summarize": ["enrichment", "summarize"],
    "enrichment_keyword_count": ["enrichment", "keyword_count"],
    "enrichment_prompt": ["enrichment", "prompt"],
}


def _settings_view(cfg: dict) -> dict:
    """Curated, flat view of the editable settings for the form UI."""
    def dig(path, default=None):
        cur = cfg
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default
    view = {k: dig(p) for k, p in _SETTINGS_MAP.items()}
    # structured (dict/list) settings edited with dedicated UI widgets
    view["vision_prompts"] = dig(["vision", "prompt"]) or {}
    view["ingestion_steps"] = dig(["ingestion", "steps"]) or []
    view["route_gates"] = dig(["ingestion", "route_gates"]) or {}
    # dropdown / option lists
    view["_industries"] = cfg.get("industries", [])
    view["_available_tools"] = sorted(_registry.names())
    view["_digital_pdf_options"] = ["pymupdf_pdf", "docling_pdf"]
    view["_active"] = os.path.basename(CONFIG_PATH)
    return view


def _apply_settings(raw: dict, settings: dict) -> dict:
    for key, path in _SETTINGS_MAP.items():
        if key not in settings or settings[key] is None:
            continue
        cur = raw
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = settings[key]
    # structured settings (replace wholesale when provided)
    if isinstance(settings.get("vision_prompts"), dict):
        raw.setdefault("vision", {})["prompt"] = settings["vision_prompts"]
    if isinstance(settings.get("ingestion_steps"), list):
        raw.setdefault("ingestion", {})["steps"] = settings["ingestion_steps"]
    if isinstance(settings.get("route_gates"), dict):
        raw.setdefault("ingestion", {})["route_gates"] = settings["route_gates"]
    return raw


def _validate_yaml(text: str) -> dict:
    try:
        parsed = yaml.safe_load(text)
    except Exception as exc:
        raise HTTPException(400, f"invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(400, "config must be a YAML mapping")
    return parsed


def _file_type(filename: str) -> str:
    return EXT_TO_FILE_TYPE.get(os.path.splitext(filename)[1].lower(), "unknown")


def _pg() -> PostgresStore:
    try:
        return PostgresStore()
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {exc}") from exc


# Ingestion runs in a FastAPI background task so /upload returns immediately. Progress
# is written to Postgres after EVERY step — the documents row is the single source of
# truth — so /files/{id}/progress reads it straight from the DB. No in-memory state:
# survives restarts and works across multiple workers.
_INGEST_STEPS = list((_config.get("ingestion") or {}).get("steps") or [])


def _save_page_images(document_id: str, pdf_path: str, pages: set, dpi: int = 150) -> list:
    """Render the given (1-based) PDF pages to JPEGs under uploads/pages/<doc_id>/
    and return [(page, web_path, width, height)]. PDF-only; best-effort. fitz
    rendering is pure C (no torch/paddle), so it's safe in the backend process."""
    import fitz
    page_dir = os.path.join(_PAGES_DIR, document_id)
    os.makedirs(page_dir, exist_ok=True)
    out = []
    doc = fitz.open(pdf_path)
    try:
        for p in sorted(pages):
            if p < 1 or p > len(doc):
                continue
            pix = doc[p - 1].get_pixmap(dpi=dpi)
            fname = f"p{p}.jpg"
            with open(os.path.join(page_dir, fname), "wb") as f:
                f.write(pix.tobytes("jpeg", jpg_quality=80))
            out.append((p, f"/pages/{document_id}/{fname}", pix.width, pix.height))
    finally:
        doc.close()
    return out


def _run_ingestion(document_id: str, dest: str, file_type: str, filename: str) -> None:
    """Run the full pipeline for one upload. Delegates run + status + finalize to the
    shared ingest_document entry point; the API-only tail (live DB progress + PDF
    page images) is injected via the on_step / on_complete hooks."""
    total = len(_INGEST_STEPS) or None

    # One connection for the whole run (per-step progress UPDATEs + page images).
    pg = None
    try:
        pg = PostgresStore()
    except Exception:
        logger.exception("progress DB unavailable for %s; running without live progress",
                         document_id)

    def on_step(entry: dict, snapshot: dict) -> None:
        if pg is None:
            return
        metrics = snapshot.get("metrics", []) or []
        progress = min(len(metrics) / total, 0.99) if total else None
        try:
            pg.update_progress(
                document_id,
                metrics=metrics,
                current_step=entry.get("step"),
                progress=progress,
                total_steps=total,
                route=snapshot.get("route"),
                confidence=snapshot.get("confidence"),
                document_type=snapshot.get("document_type"),
                industry=snapshot.get("industry"),
            )
        except Exception:
            logger.warning("update_progress failed for %s (step=%s)",
                           document_id, entry.get("step"))

    def on_complete(result: dict) -> None:
        # API-only tail: full-page images for the PDF pages that produced chunks —
        # for visual grounding ("pull up the page"). Only content pages; skip on
        # failure or if the DB is down.
        if pg is None or file_type != "pdf" or result.get("status") == "failed":
            return
        try:
            chunks = result.get("chunks", []) or []
            pages = {
                int(c["source_ref"]["page"]) for c in chunks
                if isinstance(c.get("source_ref"), dict)
                and c["source_ref"].get("page") is not None
            }
            for p, web, w, h in _save_page_images(document_id, dest, pages):
                pg.insert_page_image(document_id, p, web, w, h)
            logger.info("saved %d page images for %s", len(pages), document_id)
        except Exception:
            logger.exception("page-image save failed for %s", document_id)

    try:
        ingest_document(dest, document_id, config=_config, registry=_registry,
                        on_step=on_step, on_complete=on_complete)
    finally:
        if pg is not None:
            pg.close()


@app.post("/upload")
def upload(background: BackgroundTasks, file: UploadFile = File(...)):
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

    # Kick off ingestion in the background; progress is persisted to Postgres and the
    # UI polls /files/{id}/progress (DB-backed) below.
    background.add_task(_run_ingestion, document_id, dest, file_type, file.filename)

    return {"id": document_id, "document_id": document_id,
            "filename": file.filename, "file_type": file_type,
            "status": "processing", "metrics": []}


@app.post("/files/stage")
def stage_file(file: UploadFile = File(...)):
    """Save an uploaded file to disk WITHOUT triggering ingestion — for the agent
    chat's file-attach flow. /upload always auto-ingests (deterministic path, no
    agent involved); a staged file instead becomes a file_path the agent can see
    and choose (with approval, since ingest_document is a write) to ingest via
    the normal agent-executor flow. Not registered as a `documents` row — it only
    becomes one if/when ingest_document actually runs on it."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stage_id = str(uuid.uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{stage_id}_{file.filename}")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_path": dest, "filename": file.filename}


@app.get("/files/{file_id}/progress")
def file_progress(file_id: str):
    """Live per-step status for an in-flight (or finished) ingestion, read straight
    from Postgres — the single source of truth (survives restarts + multi-worker)."""
    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


@app.get("/files")
def list_files():
    pg = _pg()
    try:
        return pg.list_documents()
    finally:
        pg.close()


@app.get("/files/{file_id}")
def get_file(file_id: str):
    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


@app.get("/files/{file_id}/pages")
def file_pages(file_id: str):
    """List rendered full-page images for a document (page -> /pages/... url).
    Only pages that produced chunks are present."""
    pg = _pg()
    try:
        return {"document_id": file_id, "pages": pg.list_page_images(file_id)}
    finally:
        pg.close()


@app.get("/files/{file_id}/pages/{page}")
def file_page(file_id: str, page: int):
    """Metadata for one page image (bytes served at the returned image_path).
    Lets the answerer/agent pull up a full page for visual grounding."""
    pg = _pg()
    try:
        rec = pg.get_page_image(file_id, page)
    finally:
        pg.close()
    if not rec:
        raise HTTPException(404, "page image not found")
    return rec


@app.get("/files/{file_id}/pdf")
def file_pdf(file_id: str):
    """Stream the original PDF file for this document so the frontend can
    embed it in an <iframe> with a #page=N fragment for visual grounding.
    Only serves PDF documents; returns 404 for other file types."""
    from fastapi.responses import FileResponse
    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()
    if not doc:
        raise HTTPException(404, "document not found")
    if doc.get("file_type") != "pdf":
        raise HTTPException(400, "document is not a PDF")
    file_path = doc.get("file_path") or ""
    if not file_path or not os.path.isfile(file_path):
        # Fall back: scan uploads dir for {file_id}_*.pdf
        import glob as _glob
        matches = _glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}_*.pdf"))
        if not matches:
            raise HTTPException(404, "PDF file not found on disk")
        file_path = matches[0]
    filename = doc.get("filename") or os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/files/{file_id}/pages/{page}/image")
def file_page_image_ondemand(file_id: str, page: int):
    """Render a specific page of a PDF to JPEG on the fly and return it."""
    import fitz
    from fastapi.responses import Response

    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()

    if not doc:
        raise HTTPException(404, "document not found")

    file_path = doc.get("file_path") or ""
    if not file_path or not os.path.isfile(file_path):
        import glob as _glob
        matches = _glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}_*.pdf"))
        if not matches:
            raise HTTPException(404, "PDF file not found on disk")
        file_path = matches[0]

    try:
        pdf = fitz.open(file_path)
        total_pages = len(pdf)
        if page < 1 or page > total_pages:
            raise HTTPException(400, f"invalid page number: {page}. PDF has {total_pages} pages.")

        # Render page to JPEG with 150 DPI for high quality
        pix = pdf[page - 1].get_pixmap(dpi=150)
        image_bytes = pix.tobytes("jpeg", jpg_quality=85)
        pdf.close()

        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as exc:
        logger.exception("Failed to render page %d on-demand for %s", page, file_id)
        raise HTTPException(500, f"failed to render page: {exc}")


@app.get("/files/{file_id}/pdf-info")
def file_pdf_info(file_id: str):
    """Get metadata about the PDF (like total number of pages) using PyMuPDF."""
    import fitz
    pg = _pg()
    try:
        doc = pg.get_document(file_id)
    finally:
        pg.close()
    if not doc:
        raise HTTPException(404, "document not found")

    file_path = doc.get("file_path") or ""
    if not file_path or not os.path.isfile(file_path):
        import glob as _glob
        matches = _glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}_*.pdf"))
        if not matches:
            raise HTTPException(404, "PDF file not found on disk")
        file_path = matches[0]

    try:
        pdf = fitz.open(file_path)
        total_pages = len(pdf)
        pdf.close()
        return {"total_pages": total_pages, "filename": doc.get("filename")}
    except Exception as exc:
        raise HTTPException(500, f"failed to read PDF: {exc}")


@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    dim = _config.get("embeddings", {}).get("dense_dim", 768)
    collection = _config.get("database", {}).get("qdrant_collection", "chunks")

    vectors = QdrantStore(dim, collection)
    try:
        vectors.delete_by_document(file_id)
    finally:
        vectors.close()

    pg = _pg()
    try:
        pg.delete_document(file_id)
    finally:
        pg.close()
    return {"deleted": file_id}


def _history_to_messages(history: list[dict]) -> list:
    """DB rows (role/content) -> LangChain messages, for seeding run_agent's
    conversation_history when a session isn't in the in-memory cache (server
    restart, or reopening an old chat). Drops tool-call structure — the model
    only needs the visible text to keep context, not the exact prior calls."""
    msgs = []
    for h in history:
        if h.get("role") == "user":
            msgs.append(HumanMessage(h.get("content") or ""))
        elif h.get("role") == "assistant" and h.get("content"):
            msgs.append(AIMessage(h["content"]))
    return msgs


@app.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    """Agentic chat: the model picks which tool to call (ingest/search/sql) instead
    of always going straight to retrieval. Writes (ingest_document) stop and report
    what they want to run — POST again with approved_writes=true to actually run it.
    Turns are persisted (Postgres) so the UI's session sidebar can list + reopen
    past conversations; the in-memory cache is just a fast path within one run.
    """
    from backend.storage.conversation_store import get_conversation_store

    max_history = (_config.get("query", {}).get("agent", {}) or {}).get("max_history_messages", 20)
    history = _agent_sessions.get(req.session_id)
    if history is None:
        try:
            history = _history_to_messages(
                get_conversation_store().load_history(req.session_id, n=max_history)
            )
        except Exception:
            logger.debug("agent chat history load failed", exc_info=True)
            history = []
    elif len(history) > max_history:
        # sliding window, not summarization — see query.agent.max_history_messages
        history = history[-max_history:]

    result = run_agent(
        req.message, config=_config, registry=_agent_registry,
        conversation_history=history, approved_writes=req.approved_writes,
        session_id=req.session_id,
    )
    tool_calls = [
        {"name": c["name"], "args": c["args"], "result": c.get("result")}
        for c in result.get("tool_calls", [])
    ]
    if result["status"] == "done":
        _agent_sessions[req.session_id] = _qa_only(result["messages"])
        try:
            store = get_conversation_store()
            store.save_turn(req.session_id, "user", req.message)
            store.save_turn(
                req.session_id, "assistant", result.get("answer") or "",
                metadata={
                    "tool_calls": tool_calls,
                    "token_usage": result.get("token_usage"),
                    "trace_id": result.get("trace_id"),
                } if (tool_calls or result.get("token_usage")) else None,
            )
        except Exception:
            logger.debug("agent chat history save failed", exc_info=True)
    return {
        "status": result["status"],
        "answer": result.get("answer"),
        "pending": result.get("pending"),
        "tool_calls": tool_calls,
        "token_usage": result.get("token_usage"),
        "trace_id": result.get("trace_id"),
    }


@app.get("/agent/sessions")
def list_agent_sessions():
    """Past conversations for the chat UI's sidebar, most recently active first."""
    from backend.storage.conversation_store import get_conversation_store

    return get_conversation_store().list_sessions()


@app.get("/agent/sessions/{session_id}")
def get_agent_session(session_id: str):
    """One conversation's full turn history, in the same shape /agent/chat
    returns per turn, so the frontend can render live and reloaded messages
    with the same component."""
    from backend.storage.conversation_store import get_conversation_store

    history = get_conversation_store().load_history(session_id, n=200)
    return [
        {"role": h["role"], "content": h["content"], **(h.get("metadata") or {})}
        for h in history
    ]


@app.patch("/agent/sessions/{session_id}")
def patch_agent_session(session_id: str, body: dict):
    """Update session metadata: { "title": "...", "pinned": true/false }."""
    from backend.storage.conversation_store import get_conversation_store

    get_conversation_store().update_session(
        session_id,
        title=body.get("title"),
        pinned=body.get("pinned"),
    )
    return {"updated": session_id}


@app.delete("/agent/sessions/{session_id}")
def delete_agent_session(session_id: str):
    from backend.storage.conversation_store import get_conversation_store

    get_conversation_store().delete_session(session_id)
    _agent_sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "tools": sorted(_registry.names())}


# ── Config / customer profiles (edited from the UI Settings page) ─────────────
# The raw YAML keeps ${ENV} placeholders for secrets, so it's safe to read/write
# over the API (real keys live only in .env and are resolved at load time).

@app.get("/config/raw")
def config_raw():
    """Active config file as editable YAML text (secrets stay as ${ENV} refs)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {"path": CONFIG_PATH, "active": os.path.basename(CONFIG_PATH), "yaml": f.read()}


@app.put("/config")
def config_save(body: ConfigSave):
    """Validate + write the active config, then hot-reload the pipeline."""
    _validate_yaml(body.yaml)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(body.yaml)
    _reload_pipeline()
    return {"ok": True, "path": CONFIG_PATH}


@app.get("/config/profiles")
def config_profiles():
    """List available config profiles (config/*.yaml) + which one is active."""
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    return {"active": os.path.basename(CONFIG_PATH), "profiles": files}


@app.post("/config/profiles")
def config_save_profile(body: ProfileSave):
    """Save the YAML as a new/!existing profile config/<name>.yaml (per customer)."""
    _validate_yaml(body.yaml)
    name = os.path.basename(body.name.strip())
    if not name:
        raise HTTPException(400, "profile name required")
    if not name.endswith(".yaml"):
        name += ".yaml"
    with open(os.path.join(CONFIG_DIR, name), "w", encoding="utf-8") as f:
        f.write(body.yaml)
    return {"ok": True, "name": name}


@app.get("/config/settings")
def config_settings():
    """Flat, form-friendly view of the editable settings (no YAML for the user)."""
    return _settings_view(_config)


@app.put("/config/settings")
def config_settings_save(body: SettingsSave):
    """Apply form settings to the active config (or save_as a new profile) + reload.
    Preserves all other config keys; only the mapped fields change."""
    global CONFIG_PATH
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f.read()) or {}
    raw = _apply_settings(raw, body.settings)
    text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)

    target = CONFIG_PATH
    if body.save_as and body.save_as.strip():
        name = os.path.basename(body.save_as.strip())
        if not name.endswith(".yaml"):
            name += ".yaml"
        target = os.path.join(CONFIG_DIR, name)
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    CONFIG_PATH = target
    _reload_pipeline()
    return {"ok": True, "active": os.path.basename(CONFIG_PATH)}


@app.post("/config/activate")
def config_activate(body: ProfileActivate):
    """Switch the active profile to config/<name>.yaml and hot-reload."""
    global CONFIG_PATH
    name = os.path.basename(body.name)
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"profile {name} not found")
    CONFIG_PATH = path
    _reload_pipeline()
    return {"ok": True, "active": name}


@app.get("/")
def root():
    return {"service": "Document Intelligence + RAG Accelerator", "docs": "/docs"}
