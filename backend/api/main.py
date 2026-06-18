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
# Select the OCR engine (surya|paddle) from config for scanned pages.
# OCR (Surya/Paddle/llama-server) now runs in an isolated subprocess per scanned
# doc (see scanned_pdf/ocr_worker + tool). The backend process must NOT load the
# OCR stack itself — that coexistence with the torch embedder + vision threads is
# exactly what crashed it (fork-from-thread abort, OpenMP segfault). So we do NOT
# warm Surya here. The child warms it on its own main thread. engine selection is
# read from config by the worker; nothing OCR-related is initialized in-process.
_registry = build_default_registry()


def _build_ingestion_cfg(cfg: dict) -> PipelineConfig:
    return PipelineConfig.from_dict({
        **cfg,
        "steps": cfg.get("ingestion", {}).get("steps", []),
        "route_gates": cfg.get("ingestion", {}).get("route_gates", {}),
    })


_ingestion_cfg = _build_ingestion_cfg(_config)

CONFIG_DIR = os.path.dirname(CONFIG_PATH) or "config"


def _reload_pipeline() -> None:
    """Re-read CONFIG_PATH and rebuild the pipeline objects in place after a config
    edit. Models are NOT re-warmed (routing/prompt/OCR/chunking edits don't change
    the embedding/vision models); to switch those, edit + restart the server."""
    global _config, _registry, _ingestion_cfg
    _config = load_config(CONFIG_PATH)
    # OCR engine/timeout are read from config by the isolated OCR subprocess at
    # ingest time, so there's nothing to set in-process here.
    _registry = build_default_registry()
    _ingestion_cfg = _build_ingestion_cfg(_config)


class ChatRequest(BaseModel):
    question: str
    file_id: str | None = None


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
    """Run the full pipeline for one upload, persisting per-step progress to the DB."""
    state = {"document_id": document_id, "file_path": dest,
             "file_type": file_type, "errors": []}
    total = len(_INGEST_STEPS) or None

    # One connection for the whole run (per-step UPDATEs + the final finalize).
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

    try:
        result = run_pipeline(_registry, state, _ingestion_cfg, on_step=on_step)
        # "failed" only if a STEP errored — a non-fatal "low confidence" warning
        # in errors must not mark an otherwise-successful ingest as failed.
        step_failed = any(m.get("status") == "error" for m in result.get("metrics", []))
        status = "failed" if step_failed else "ready"
    except Exception as exc:
        logger.exception("ingestion failed for %s", filename)
        result, status = {"errors": [str(exc)]}, "failed"

    chunks = result.get("chunks", []) or []
    indexed_tokens = sum(int(c.get("token_count") or 0) for c in chunks)
    try:
        if pg is None:
            pg = PostgresStore()
        pg.finalize_document(
            document_id,
            document_type=result.get("document_type"),
            industry=result.get("industry"),
            route=result.get("route"),
            confidence=result.get("confidence"),
            status=status,
            errors=result.get("errors"),
            metrics=result.get("metrics"),
            token_usage=result.get("token_usage"),     # LLM/vision tokens consumed
            indexed_tokens=indexed_tokens,              # tokens of text indexed
            chunk_count=len(chunks),
        )
        # Full-page images for the PDF pages that produced chunks — for visual
        # grounding ("pull up the page" when a chunk is ambiguous). Only pages with
        # content, so blank pages aren't rendered/stored.
        if file_type == "pdf" and status != "failed":
            try:
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
    except Exception:
        logger.exception("finalize_document failed for %s", document_id)
    finally:
        if pg is not None:
            pg.close()


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
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


@app.get("/files/{file_id}/progress")
async def file_progress(file_id: str):
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


@app.get("/files/{file_id}/pages")
async def file_pages(file_id: str):
    """List rendered full-page images for a document (page -> /pages/... url).
    Only pages that produced chunks are present."""
    pg = _pg()
    try:
        return {"document_id": file_id, "pages": pg.list_page_images(file_id)}
    finally:
        pg.close()


@app.get("/files/{file_id}/pages/{page}")
async def file_page(file_id: str, page: int):
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
        {"filename": c.get("filename"), "page": c.get("page"),
         "snippet": c.get("snippet"), "summary": c.get("summary"),
         "image_path": c.get("image_path")}
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


# ── Config / customer profiles (edited from the UI Settings page) ─────────────
# The raw YAML keeps ${ENV} placeholders for secrets, so it's safe to read/write
# over the API (real keys live only in .env and are resolved at load time).

@app.get("/config/raw")
async def config_raw():
    """Active config file as editable YAML text (secrets stay as ${ENV} refs)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {"path": CONFIG_PATH, "active": os.path.basename(CONFIG_PATH), "yaml": f.read()}


@app.put("/config")
async def config_save(body: ConfigSave):
    """Validate + write the active config, then hot-reload the pipeline."""
    _validate_yaml(body.yaml)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(body.yaml)
    _reload_pipeline()
    return {"ok": True, "path": CONFIG_PATH}


@app.get("/config/profiles")
async def config_profiles():
    """List available config profiles (config/*.yaml) + which one is active."""
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    return {"active": os.path.basename(CONFIG_PATH), "profiles": files}


@app.post("/config/profiles")
async def config_save_profile(body: ProfileSave):
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
async def config_settings():
    """Flat, form-friendly view of the editable settings (no YAML for the user)."""
    return _settings_view(_config)


@app.put("/config/settings")
async def config_settings_save(body: SettingsSave):
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
async def config_activate(body: ProfileActivate):
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
async def root():
    return {"service": "Document Intelligence + RAG Accelerator", "docs": "/docs"}
