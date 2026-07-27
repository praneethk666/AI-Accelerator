"""ingest_document — one clean entry point to run the full ingestion pipeline on a
file and return its outcome. Same recipe the API's /upload and scripts/run_ingest.py
use, but file-type-aware, idempotent, and agent-callable.

- hand it a path; it runs categorize -> extract -> ... -> index
- idempotent: same file (by content hash) or same document_id updates in place,
  never duplicates — prior chunks are cleared from Postgres + Qdrant before the run
- DB-backed status via PostgresStore (insert_document / finalize_document)
- returns {document_id, status, metrics, errors}

Note on the pre-clean: chunk_ids are regenerated per run, so the only way to make
re-ingestion idempotent without touching chunk_tool is to delete the document's old
chunks up front. Trade-off: during a re-ingest the doc is briefly chunk-less, and a
mid-run failure leaves it empty. Acceptable for an explicit re-ingest.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid

from backend.core.config import PipelineConfig, load_config
from backend.core.tracing import traced_request, record_handled_error
from backend.pipeline.default_registry import build_default_registry
from backend.pipeline.graph import run_pipeline
from backend.core.paths import display_filename
from backend.storage.postgres_store import PostgresStore
from backend.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/global.yaml")

# file extension -> pipeline file_type (mirrors the API's map; unknown -> "unknown")
EXT_TO_FILE_TYPE = {
    ".pdf": "pdf",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".pptx": "ppt", ".ppt": "ppt",
    ".docx": "docx", ".doc": "docx",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".tif": "image", ".tiff": "image", ".bmp": "image", ".gif": "image",
}

# Known-but-unextractable formats seen in real corpora (e.g. Toyota manuals). We
# can't read these, so we FAIL LOUD with a specific reason rather than silently
# finalizing an empty "ready" doc. Convert to PDF/image upstream to ingest them.
_UNSUPPORTED_HINTS = {
    ".xdw": "Fuji Xerox DocuWorks — convert to PDF first",
    ".xbd": "Fuji Xerox DocuWorks binder — convert to PDF first",
    ".ipw": "proprietary drawing format — export to PDF/image first",
    ".shzz": "proprietary format — export to PDF/image first",
    ".zmw": "proprietary format — export to PDF/image first",
    ".zmn": "proprietary format — export to PDF/image first",
}

# Stable namespace so the same file bytes always map to the same document_id.
_DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ai-accelerator/ingest")


def file_type_of(path: str) -> str:
    """Map a filename/extension to the pipeline file_type (or 'unknown')."""
    return EXT_TO_FILE_TYPE.get(os.path.splitext(path)[1].lower(), "unknown")


def _content_id(file_path: str) -> str:
    """Deterministic document_id from file content — same bytes => same id."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return str(uuid.uuid5(_DOC_NAMESPACE, h.hexdigest()))


def _ingestion_cfg(cfg: dict) -> PipelineConfig:
    # the ingestion profile is the subset of steps/gates under `ingestion:`
    return PipelineConfig.from_dict({
        **cfg,
        "steps": cfg.get("ingestion", {}).get("steps", []),
        "route_gates": cfg.get("ingestion", {}).get("route_gates", {}),
    })


def _preclean(document_id: str, cfg: dict) -> None:
    """Remove any prior chunks for this document from both stores (idempotency)."""
    pg = PostgresStore()
    try:
        pg.delete_chunks(document_id)
    finally:
        pg.close()
    dim = cfg.get("embeddings", {}).get("dense_dim", 768)
    collection = cfg.get("database", {}).get("qdrant_collection", "chunks")
    try:
        vectors = QdrantStore(dim, collection)
        try:
            vectors.delete_by_document(document_id)
        finally:
            vectors.close()
    except Exception:
        # Qdrant down/empty is non-fatal for the clean step; index will surface it
        logger.warning("qdrant pre-clean skipped for %s (store unavailable)", document_id)


def ingest_document(
    file_path: str,
    document_id: str | None = None,
    *,
    config: dict | None = None,
    registry=None,
    on_step=None,
    on_complete=None,
) -> dict:
    """Run the full ingestion pipeline on one file and return its outcome.

    Args:
        file_path: path to the document (pdf/xlsx/pptx/image).
        document_id: reuse this id (update in place). If None, it's derived from
            the file's content hash so the same file is idempotent.
        config: loaded config dict; defaults to load_config(CONFIG_PATH).
        registry: tool registry; defaults to build_default_registry(). Pass a
            prebuilt one (e.g. from the API) to avoid re-warming models.
        on_step: optional callback(entry, snapshot) for live per-step progress.
        on_complete: optional callback(result) run after finalize with the FULL
            pipeline result (includes chunks + "status") — lets a caller do its own
            tail work (e.g. the API renders page images) without bloating the
            agent-facing return.

    Returns:
        {"document_id", "status", "metrics", "errors", "trace_id"}. status is one of:
          ready       — extracted content and indexed (>=1 chunk)
          empty       — supported format but zero chunks extracted (surfaced, not hidden)
          unsupported — file type we can't extract (e.g. .xdw DocuWorks); nothing run
          failed      — a pipeline step raised
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    cfg = config if config is not None else load_config(CONFIG_PATH)
    reg = registry if registry is not None else build_default_registry()
    document_id = document_id or _content_id(file_path)
    file_type = file_type_of(file_path)
    filename = display_filename(file_path)

    # register the doc row (no-op if it exists) + clear prior chunks so re-ingest
    # replaces rather than duplicates.
    pg = PostgresStore()
    try:
        pg.insert_document(document_id, filename, file_type, file_path)
    finally:
        pg.close()
    _preclean(document_id, cfg)

    # Fail LOUD on formats we cannot extract: an "unknown" file_type enables no
    # extractor, so the pipeline would produce zero chunks yet still finalize
    # "ready" — hiding the file. Short-circuit to an explicit "unsupported" status.
    ext = os.path.splitext(file_path)[1].lower()
    unsupported_msg = None
    if file_type == "unknown":
        hint = _UNSUPPORTED_HINTS.get(ext)
        unsupported_msg = (
            f"Unsupported file type '{ext or '(none)'}'"
            + (f" ({hint})" if hint else "")
            + f". Supported: {', '.join(sorted(set(EXT_TO_FILE_TYPE)))}."
        )
        logger.warning("ingest %s (%s): %s", filename, document_id, unsupported_msg)

    # run the pipeline (graph owns routing/extraction; we just seed file_type)
    state = {"document_id": document_id, "file_path": file_path,
             "file_type": file_type, "errors": []}

    with traced_request(
        "ingest_document", input={"filename": filename, "file_type": file_type},
        metadata={"document_id": document_id, "file_path": file_path},
    ) as trace_info:
        if unsupported_msg:
            result, status = {"errors": [unsupported_msg]}, "unsupported"
        else:
            try:
                result = run_pipeline(reg, state, _ingestion_cfg(cfg), on_step=on_step)
                # "failed" only if a STEP errored — a non-fatal warning in errors must not
                # mark an otherwise-successful ingest as failed.
                step_failed = any(m.get("status") == "error" for m in result.get("metrics", []))
                status = "failed" if step_failed else "ready"
            except Exception as exc:
                from backend.core.tool import IngestionCancelledError
                if isinstance(exc, IngestionCancelledError):
                    logger.info("Ingestion cancelled for %s: document was deleted by user.", filename)
                else:
                    logger.exception("ingestion failed for %s", filename)
                result, status = {"errors": [str(exc)]}, "failed"
                record_handled_error(
                    "ingest_pipeline_failure", str(exc), **{"document_id": document_id}
                )
    trace_id = trace_info["trace_id"]

    metrics = result.get("metrics", []) or []
    errors = result.get("errors", []) or []
    chunks = result.get("chunks", []) or []

    # A SUPPORTED file that extracted nothing is not a success — surface it instead
    # of finalizing "ready" with zero content (corrupt/empty/image-only-without-OCR).
    if status == "ready" and not chunks:
        status = "empty"
        errors = list(errors) + [
            "No content extracted (zero chunks). The file may be corrupt, empty, "
            "or image-only without a working OCR/vision path."
        ]
        logger.warning("ingest %s (%s): zero chunks -> status 'empty'", filename, document_id)

    indexed_tokens = sum(int(c.get("token_count") or 0) for c in chunks)

    # persist terminal status + aggregates (DB-backed status the API reports)
    try:
        pg = PostgresStore()
        try:
            if not pg.document_exists(document_id):
                logger.warning("Document %s was deleted mid-ingestion; aborting database finalization.", document_id)
                return {"document_id": document_id, "status": "deleted",
                        "metrics": metrics, "errors": errors, "trace_id": trace_id}
                
            # Render and register page/slide images for visual grounding
            if status == "ready":
                try:
                    pages_dir = os.path.join("uploads", "pages")
                    page_dir = os.path.join(pages_dir, document_id)
                    os.makedirs(page_dir, exist_ok=True)
                    
                    ext = os.path.splitext(file_path)[1].lower()
                    file_type = result.get("document_type") or ""
                    
                    if ext in (".pptx", ".ppt") or file_type == "presentation":
                        from backend.core.office_renderer import render_pptx_slides
                        render_pptx_slides(file_path, page_dir)
                        
                        # Register PPT slide entries in DB
                        pages = {
                            int(c["source_ref"]["slide"]) for c in chunks
                            if isinstance(c.get("source_ref"), dict)
                            and c["source_ref"].get("slide") is not None
                        }
                        for p in pages:
                            fname = f"p{p}.jpg"
                            img_path = os.path.join(page_dir, fname)
                            w, h = 960, 720
                            if os.path.isfile(img_path):
                                try:
                                    from PIL import Image
                                    with Image.open(img_path) as img:
                                        w, h = img.size
                                except Exception:
                                    pass
                            pg.insert_page_image(document_id, p, f"/pages/{document_id}/{fname}", w, h)
                        logger.info("saved %d slide images for %s", len(pages), document_id)
                        
                    elif ext == ".pdf" or file_type == "pdf":
                        pages = {
                            int(c["source_ref"]["page"]) for c in chunks
                            if isinstance(c.get("source_ref"), dict)
                            and c["source_ref"].get("page") is not None
                        }
                        import fitz
                        doc = fitz.open(file_path)
                        try:
                            for p in sorted(pages):
                                if p < 1 or p > len(doc):
                                    continue
                                pix = doc[p - 1].get_pixmap(dpi=150)
                                fname = f"p{p}.jpg"
                                with open(os.path.join(page_dir, fname), "wb") as f:
                                    f.write(pix.tobytes("jpeg", jpg_quality=80))
                                pg.insert_page_image(document_id, p, f"/pages/{document_id}/{fname}", pix.width, pix.height)
                        finally:
                            doc.close()
                        logger.info("saved %d page images for %s", len(pages), document_id)
                except Exception:
                    logger.exception("Failed to render and save page/slide images for %s", document_id)
                
            pg.finalize_document(
                document_id,
                document_type=result.get("document_type"),
                industry=result.get("industry"),
                route=result.get("route"),
                confidence=result.get("confidence"),
                status=status,
                errors=errors,
                metrics=metrics,
                token_usage=result.get("token_usage"),
                indexed_tokens=indexed_tokens,
                chunk_count=len(chunks),
            )
            # Raw extracted blocks (pre-chunking) and the full prompt/response audit
            # trail for every LLM/vision call — kept even on a "failed"/"empty"
            # status, since that's exactly when the raw record is most useful for
            # debugging. Best-effort: never let this break a successful ingest.
            try:
                pg.write_blocks(document_id, result.get("blocks") or [])
            except Exception:
                logger.exception("write_blocks failed for %s", document_id)
            try:
                pg.write_llm_calls(document_id, result.get("_calls_log") or [])
            except Exception:
                logger.exception("write_llm_calls failed for %s", document_id)
        finally:
            pg.close()
    except Exception:
        logger.exception("finalize_document failed for %s", document_id)

    # caller tail work (page images, rich logging) gets the full result + status
    if on_complete is not None:
        try:
            on_complete({**result, "status": status})
        except Exception:
            logger.exception("on_complete hook failed for %s", document_id)

    return {"document_id": document_id, "status": status,
            "metrics": metrics, "errors": errors, "trace_id": trace_id}


class IngestDocumentTool:
    """Agent-callable tool: ingest one document through the full pipeline.

    Distinct from pipeline-step tools (which run inside the graph on shared state)
    — this orchestrates a whole graph run, so an agent calls it, not the graph.
    Exposes name + description + input_schema so the agent-executor can advertise
    and invoke it (standard tool-use shape).
    """

    name = "ingest_document"
    description = (
        "Ingest a document file (PDF, XLSX, PPTX, or image) through the full "
        "pipeline (categorize -> extract -> chunk -> embed -> index) so it becomes "
        "searchable. Idempotent: ingesting the same file again updates it in place "
        "rather than duplicating. Returns document_id, status, per-step metrics, "
        "and any errors."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the document to ingest.",
            },
            "document_id": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"}
                ],
                "description": "Optional id to update in place; omit to derive a "
                "stable id from the file's content.",
            },
        },
        "required": ["file_path"],
    }

    def run(self, file_path: str, document_id: str | None = None) -> dict:
        if document_id in ("null", "None"):
            document_id = None
        return ingest_document(file_path, document_id)

    __call__ = run
