"""Tests for ingest_document — the agent-callable ingestion entry point.

Two layers:
  * unit        — orchestration + idempotency logic with fakes (no infra, always run)
  * integration — real end-to-end on GENERATED fixtures (SKIP when the stack is down)

Integration fixtures are built in-test (openpyxl/python-pptx/fitz) so nothing depends
on the gitignored uploads/. They exercise the same path the API and run_ingest.py use.
"""
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # so dsn_from_env / QDRANT_URL resolve when a local .env is present

from backend.agent_tools import AgentTool, build_agent_registry  # noqa: E402
from backend.pipeline import ingest as ing  # noqa: E402
from backend.pipeline.ingest import file_type_of, ingest_document  # noqa: E402


# ── unit: pure helpers ────────────────────────────────────────────────────────
def test_file_type_of():
    assert file_type_of("a.pdf") == "pdf"
    assert file_type_of("A.XLSX") == "excel"      # case-insensitive
    assert file_type_of("deck.pptx") == "ppt"
    assert file_type_of("pic.png") == "image"
    assert file_type_of("weird.zzz") == "unknown"


def test_content_id_deterministic_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    id_a1, id_a2, id_b = ing._content_id(str(a)), ing._content_id(str(a)), ing._content_id(str(b))
    assert id_a1 == id_a2 and uuid.UUID(id_a1)   # stable + valid uuid => same file = same id
    assert id_a1 != id_b                          # different bytes => different id


def test_agent_tool_registered_and_shaped():
    tool = build_agent_registry()["ingest_document"]
    assert isinstance(tool, AgentTool)            # satisfies the agent-tool protocol
    assert tool.name == "ingest_document"
    assert tool.input_schema["required"] == ["file_path"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_document(str(tmp_path / "nope.pdf"))


# ── unit: orchestration with fakes (no infra) ─────────────────────────────────
class _FakePG:
    calls: list = []

    def __init__(self, *a, **k): pass
    def insert_document(self, *a, **k): _FakePG.calls.append("insert")
    def delete_chunks(self, *a, **k): _FakePG.calls.append("pg_delete")
    def document_exists(self, *a, **k): return True  # not deleted mid-ingest in these tests
    def finalize_document(self, *a, **k): _FakePG.calls.append(("finalize", k.get("status")))
    def close(self): pass


class _FakeQ:
    def __init__(self, *a, **k): pass
    def delete_by_document(self, *a, **k): _FakePG.calls.append("q_delete")
    def close(self): pass


def _patch(monkeypatch, result):
    """Stub storage + pipeline so a call exercises ingest_document's orchestration only."""
    _FakePG.calls = []

    def fake_run(reg, state, cfg, on_step=None):
        for m in result.get("metrics", []):
            if on_step:
                on_step(m, result)
        return result

    monkeypatch.setattr(ing, "PostgresStore", _FakePG)
    monkeypatch.setattr(ing, "QdrantStore", _FakeQ)
    monkeypatch.setattr(ing, "run_pipeline", fake_run)
    return _FakePG.calls


def test_orchestration_order_and_return(monkeypatch, tmp_path):
    result = {"metrics": [{"step": "chunk", "ms": 1.0, "status": "ok"}],
              "chunks": [{"token_count": 3, "source_ref": {"page": 1}}], "route": "text_default"}
    calls = _patch(monkeypatch, result)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    seen, done = [], {}
    out = ingest_document(str(f), config={"ingestion": {"steps": ["chunk"]}}, registry=object(),
                          on_step=lambda e, s: seen.append(e["step"]),
                          on_complete=lambda r: done.update(r))

    assert sorted(out) == ["document_id", "errors", "metrics", "status", "trace_id"]
    assert out["status"] == "ready"
    assert seen == ["chunk"]                                 # on_step fired per step
    assert done["status"] == "ready" and len(done["chunks"]) == 1  # on_complete got full result
    # idempotency pre-clean happens BEFORE the run, finalize after:
    assert calls[0] == "insert"
    assert calls.index("pg_delete") < calls.index(("finalize", "ready"))
    assert "q_delete" in calls and ("finalize", "ready") in calls


def test_status_failed_when_a_step_errors(monkeypatch, tmp_path):
    result = {"metrics": [{"step": "extract", "ms": 1.0, "status": "error", "error": "boom"}],
              "chunks": []}
    _patch(monkeypatch, result)
    f = tmp_path / "x.xlsx"
    f.write_bytes(b"fake")
    out = ingest_document(str(f), config={"ingestion": {"steps": ["extract"]}}, registry=object())
    assert out["status"] == "failed"


def test_unsupported_format_fails_loud(monkeypatch, tmp_path):
    # .xdw (DocuWorks) is unextractable: must FAIL LOUD, not finalize an empty "ready".
    calls = _patch(monkeypatch, {"metrics": [], "chunks": []})
    f = tmp_path / "drawing.xdw"
    f.write_bytes(b"DocuWorks binary")
    out = ingest_document(str(f), config={"ingestion": {"steps": ["chunk"]}}, registry=object())
    assert out["status"] == "unsupported"
    assert ("finalize", "unsupported") in calls
    assert any("DocuWorks" in e or "Unsupported" in e for e in out["errors"])


def test_zero_chunk_supported_is_empty(monkeypatch, tmp_path):
    # supported format (.pdf) but extraction yielded nothing => "empty", not "ready".
    calls = _patch(monkeypatch, {"metrics": [{"step": "extract", "ms": 1.0, "status": "ok"}],
                                 "chunks": [], "route": "text_default"})
    f = tmp_path / "blank.pdf"
    f.write_bytes(b"%PDF-1.4 empty")
    out = ingest_document(str(f), config={"ingestion": {"steps": ["extract"]}}, registry=object())
    assert out["status"] == "empty"
    assert ("finalize", "empty") in calls


def test_explicit_document_id_is_reused(monkeypatch, tmp_path):
    _patch(monkeypatch, {"metrics": [], "chunks": []})
    f = tmp_path / "x.pptx"
    f.write_bytes(b"fake")
    out = ingest_document(str(f), document_id="fixed-123",
                          config={"ingestion": {"steps": []}}, registry=object())
    assert out["document_id"] == "fixed-123"     # caller id wins over content hash


# ── integration: real end-to-end, gated on the running stack ──────────────────
def _stack_up() -> bool:
    try:
        import psycopg
        from qdrant_client import QdrantClient

        from backend.storage.postgres_store import dsn_from_env
        from backend.storage.qdrant_store import url_from_env
        psycopg.connect(dsn_from_env(), connect_timeout=2).close()
        QdrantClient(url=url_from_env()).get_collections()
        return True
    except Exception:
        return False


STACK = _stack_up()
needs_stack = pytest.mark.skipif(not STACK, reason="Postgres+Qdrant not running (docker compose up -d)")


def _chunk_count(doc_id: str) -> int:
    from backend.storage.postgres_store import PostgresStore
    pg = PostgresStore()
    try:
        return pg.conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id::text = %s", (doc_id,)
        ).fetchone()[0]
    finally:
        pg.close()


def _doc_status(doc_id: str) -> str | None:
    from backend.storage.postgres_store import PostgresStore
    pg = PostgresStore()
    try:
        doc = pg.get_document(doc_id)
        return doc.get("status") if doc else None
    finally:
        pg.close()


def _cleanup(doc_id: str, cfg: dict) -> None:
    from backend.storage.postgres_store import PostgresStore
    from backend.storage.qdrant_store import QdrantStore
    pg = PostgresStore()
    try:
        pg.delete_document(doc_id)
    finally:
        pg.close()
    try:
        q = QdrantStore(cfg.get("embeddings", {}).get("dense_dim", 768),
                        cfg.get("database", {}).get("qdrant_collection", "chunks"))
        try:
            q.delete_by_document(doc_id)
        finally:
            q.close()
    except Exception:
        pass


def _cfg() -> dict:
    from backend.core.config import load_config
    return load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))


def _roundtrip(path: str, cfg: dict):
    """Ingest -> assert ready + chunks persisted -> re-ingest is idempotent."""
    out = ingest_document(path, config=cfg)
    try:
        assert out["status"] == "ready", out["errors"]
        assert _doc_status(out["document_id"]) == "ready"          # DB-backed status
        n = _chunk_count(out["document_id"])
        assert n >= 1                                              # chunks persisted
        again = ingest_document(path, config=cfg)                 # re-ingest same file
        assert again["document_id"] == out["document_id"]         # stable content id
        assert _chunk_count(out["document_id"]) == n              # no duplication
    finally:
        _cleanup(out["document_id"], cfg)


@needs_stack
def test_ingest_xlsx_end_to_end(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "sales.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Region", "Q1", "Q2"])
    ws.append(["West", 100, 120])
    ws.append(["East", 90, 110])
    wb.save(p)
    _roundtrip(str(p), _cfg())


@needs_stack
def test_ingest_pptx_end_to_end(tmp_path):
    pptx = pytest.importorskip("pptx")
    p = tmp_path / "deck.pptx"
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Quarterly Review"
    box = slide.shapes.add_textbox(0, 0, prs.slide_width, prs.slide_height)
    box.text_frame.text = "Revenue grew across all regions this quarter. Details follow."
    prs.save(p)
    _roundtrip(str(p), _cfg())


@needs_stack
def test_ingest_pdf_end_to_end(tmp_path):
    pytest.importorskip("docling_core")      # docling_pdf extractor's dep
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "story.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a smoke-test PDF with enough text to chunk. " * 6)
    doc.save(str(p))
    doc.close()
    cfg = _cfg()
    cfg.setdefault("vision_ocr", {})["mode"] = "off"   # native extraction (no remote VLM in CI)
    _roundtrip(str(p), cfg)
