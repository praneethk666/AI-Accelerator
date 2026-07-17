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
from backend.core.tracing import traced_request
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
    ".tif": "image", ".tiff": "image",
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
        {"document_id", "status", "metrics", "errors"} — status is "ready" unless
        a pipeline step errored, then "failed".
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

    # run the pipeline (graph owns routing/extraction; we just seed file_type)
    state = {"document_id": document_id, "file_path": file_path,
             "file_type": file_type, "errors": []}
    
    with traced_request(
        "ingest_document", input={"filename": filename, "file_type": file_type},
        metadata={"document_id": document_id, "file_path": file_path},
    ) as trace_info:
        try:
            result = run_pipeline(reg, state, _ingestion_cfg(cfg), on_step=on_step)
            # "failed" only if a STEP errored — a non-fatal warning in errors must not
            # mark an otherwise-successful ingest as failed.
            step_failed = any(m.get("status") == "error" for m in result.get("metrics", []))
            status = "failed" if step_failed else "ready"
        except Exception as exc:
            logger.exception("ingestion failed for %s", filename)
            result, status = {"errors": [str(exc)]}, "failed"
    trace_id = trace_info["trace_id"]

    metrics = result.get("metrics", []) or []
    errors = result.get("errors", []) or []
    chunks = result.get("chunks", []) or []
    indexed_tokens = sum(int(c.get("token_count") or 0) for c in chunks)

    # persist terminal status + aggregates (DB-backed status the API reports)
    try:
        pg = PostgresStore()
        try:
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
                "type": ["string", "null"],
                "description": "Optional id to update in place; omit to derive a "
                "stable id from the file's content.",
            },
        },
        "required": ["file_path"],
    }

    def run(self, file_path: str, document_id: str | None = None) -> dict:
        return ingest_document(file_path, document_id)

    __call__ = run
