"""Tests for image-grounded answer generation (backend/retrieval/answerer.py) —
real finding, 28-Jul: a table's "Indication" 7-segment-display icon column was
misread by the table-OCR engine on 9/11 rows while every other column was
correct. Attaching the actual source page image to the answer LLM call gives it
a real chance to catch/correct what the extracted text alone can't reveal.

Mock-only (LLM, DB, filesystem) — no real infra, matches tests/test_answerer_expand.py.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from backend.retrieval.answerer import (
    AnswererTool,
    _build_user_content,
    _load_grounding_images,
    _select_grounding_targets,
)


def _chunk(chunk_id, text="body text", document_id="doc-1", page=54, summary="ok"):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "token_count": 200,
        "tags": {"summary": summary},
        "source_ref": {"page": page, "filename": "manual.pdf"},
    }


# ── _select_grounding_targets ────────────────────────────────────────────────

def test_select_grounding_targets_dedups_same_page():
    chunks = [_chunk("c1", page=54), _chunk("c2", page=54), _chunk("c3", page=61)]
    targets = _select_grounding_targets(chunks, top_k=5)
    assert [t["page"] for t in targets] == [54, 61]


def test_select_grounding_targets_respects_top_k():
    chunks = [_chunk("c1", page=1), _chunk("c2", page=2), _chunk("c3", page=3)]
    targets = _select_grounding_targets(chunks, top_k=1)
    assert len(targets) == 1
    assert targets[0]["page"] == 1  # best-first order preserved


def test_select_grounding_targets_skips_non_pdf_citations():
    excel_chunk = {"chunk_id": "c1", "document_id": "doc-1", "text": "x",
                    "source_ref": {"sheet": "Sheet1", "filename": "data.xlsx"}}
    targets = _select_grounding_targets([excel_chunk, _chunk("c2", page=5)], top_k=5)
    assert len(targets) == 1
    assert targets[0]["page"] == 5


# ── _load_grounding_images ───────────────────────────────────────────────────

def _fake_store(page_image_row):
    store = MagicMock()
    store.get_page_image.return_value = page_image_row
    return store


def test_load_grounding_images_reads_and_encodes_real_file(tmp_path):
    img_path = tmp_path / "p54.jpg"
    img_path.write_bytes(b"fake-jpeg-bytes")
    row = {"page": 54, "image_path": "/pages/doc-1/p54.jpg", "width": 100, "height": 200}
    targets = [{"document_id": "doc-1", "page": 54, "source_ref": {"page": 54, "filename": "manual.pdf"}}]

    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(row)), \
         patch("backend.pipeline.page_images.physical_path", return_value=str(img_path)):
        images = _load_grounding_images(targets)

    assert len(images) == 1
    assert images[0]["label"] == "manual.pdf, p.54"
    assert base64.b64decode(images[0]["b64"]) == b"fake-jpeg-bytes"


def test_load_grounding_images_skips_target_with_no_db_row():
    # Document ingested before this feature existed -- no document_pages row.
    targets = [{"document_id": "doc-1", "page": 54, "source_ref": {"page": 54, "filename": "m.pdf"}}]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(None)):
        images = _load_grounding_images(targets)
    assert images == []


def test_load_grounding_images_skips_target_with_unreadable_file():
    row = {"page": 54, "image_path": "/pages/doc-1/p54.jpg", "width": 1, "height": 1}
    targets = [{"document_id": "doc-1", "page": 54, "source_ref": {"page": 54, "filename": "m.pdf"}}]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(row)), \
         patch("backend.pipeline.page_images.physical_path",
               return_value="/nonexistent/path/p54.jpg"):
        images = _load_grounding_images(targets)
    assert images == []  # FileNotFoundError caught, skipped, no raise


def test_load_grounding_images_empty_targets_never_touches_db():
    with patch("backend.storage.postgres_store.PostgresStore") as mock_store:
        images = _load_grounding_images([])
    assert images == []
    mock_store.assert_not_called()


# ── _build_user_content ──────────────────────────────────────────────────────

def test_build_user_content_returns_same_string_when_no_images():
    msg = "Context:\n\n[1] some text\n\nQuestion: what is it?"
    result = _build_user_content(msg, [])
    assert result is msg  # identity, not just equality -- proves zero behavior change


def test_build_user_content_builds_multimodal_blocks_with_images():
    msg = "Context:\n\nQuestion: what is the indication code?"
    images = [{"label": "manual.pdf, p.54", "b64": "ZmFrZQ=="}]
    result = _build_user_content(msg, images)
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert "manual.pdf, p.54" in result[0]["text"]
    assert result[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}}


# ── AnswererTool.run() end-to-end (image_ground.enabled) ────────────────────

def _fake_llm(response_text="the answer"):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def _base_config(image_ground: dict | None = None):
    cfg = {"llm": {"answer_model": "gpt-4o-mini"},
           "query": {"answerer": {}}}
    if image_ground is not None:
        cfg["query"]["answerer"]["image_ground"] = image_ground
    return cfg


def _base_state(chunks):
    return {"query": "what is the indication code?", "retrieved_chunks": chunks,
            "session_id": "", "conversation_history": [], "errors": []}


def test_run_attaches_image_when_enabled_and_available():
    chunks = [_chunk("c1", text="Indication: [F25]", page=54)]
    row = {"page": 54, "image_path": "/pages/doc-1/p54.jpg", "width": 1, "height": 1}
    llm = _fake_llm()

    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")), \
         patch("backend.retrieval.answerer._expand_thin_chunks", side_effect=lambda c: c), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(row)), \
         patch("backend.pipeline.page_images.physical_path", return_value="/fake/p54.jpg"), \
         patch("builtins.open", MagicMock(return_value=MagicMock(
             __enter__=lambda s: MagicMock(read=lambda: b"jpeg-bytes"),
             __exit__=lambda *a: None))), \
         patch("backend.retrieval.answerer.usage.record_from_message"):
        state = AnswererTool().run(_base_state(chunks), _base_config({"enabled": True, "top_k": 1}))

    assert state["answer"] == "the answer"
    call_messages = llm.invoke.call_args[0][0]
    user_content = call_messages[1]["content"]
    assert isinstance(user_content, list)
    assert any(b.get("type") == "image_url" for b in user_content)


def test_run_fails_open_when_image_lookup_fails_entirely():
    chunks = [_chunk("c1", page=54)]
    llm = _fake_llm()

    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")), \
         patch("backend.retrieval.answerer._expand_thin_chunks", side_effect=lambda c: c), \
         patch("backend.storage.postgres_store.PostgresStore",
               side_effect=RuntimeError("db unreachable")), \
         patch("backend.retrieval.answerer.usage.record_from_message"):
        state = AnswererTool().run(_base_state(chunks), _base_config({"enabled": True, "top_k": 1}))

    assert state["answer"] == "the answer"
    assert state["errors"] == []
    call_messages = llm.invoke.call_args[0][0]
    assert isinstance(call_messages[1]["content"], str)  # plain text, no images available


def test_run_retries_text_only_when_provider_rejects_multimodal_content():
    chunks = [_chunk("c1", page=54)]
    row = {"page": 54, "image_path": "/pages/doc-1/p54.jpg", "width": 1, "height": 1}
    llm = MagicMock()
    llm.invoke.side_effect = [RuntimeError("model does not support images"),
                               MagicMock(content="text-only answer")]

    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")), \
         patch("backend.retrieval.answerer._expand_thin_chunks", side_effect=lambda c: c), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(row)), \
         patch("backend.pipeline.page_images.physical_path", return_value="/fake/p54.jpg"), \
         patch("builtins.open", MagicMock(return_value=MagicMock(
             __enter__=lambda s: MagicMock(read=lambda: b"jpeg-bytes"),
             __exit__=lambda *a: None))), \
         patch("backend.retrieval.answerer.usage.record_from_message"):
        state = AnswererTool().run(_base_state(chunks), _base_config({"enabled": True, "top_k": 1}))

    assert llm.invoke.call_count == 2
    assert state["answer"] == "text-only answer"
    second_call_content = llm.invoke.call_args_list[1][0][0][1]["content"]
    assert isinstance(second_call_content, str)  # the retry used plain text


def test_run_default_off_never_touches_db_or_filesystem():
    # image_ground key absent entirely -- simulates a config that predates this
    # feature. Must be byte-identical in behavior/cost to before it existed.
    chunks = [_chunk("c1", page=54)]
    llm = _fake_llm()

    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")), \
         patch("backend.retrieval.answerer._expand_thin_chunks", side_effect=lambda c: c), \
         patch("backend.storage.postgres_store.PostgresStore") as mock_store, \
         patch("backend.retrieval.answerer.usage.record_from_message"):
        state = AnswererTool().run(_base_state(chunks), _base_config(None))

    mock_store.assert_not_called()
    call_messages = llm.invoke.call_args[0][0]
    assert isinstance(call_messages[1]["content"], str)
    assert state["answer"] == "the answer"


if __name__ == "__main__":
    test_select_grounding_targets_dedups_same_page()
    test_select_grounding_targets_respects_top_k()
    test_select_grounding_targets_skips_non_pdf_citations()
    test_load_grounding_images_skips_target_with_no_db_row()
    test_load_grounding_images_skips_target_with_unreadable_file()
    test_load_grounding_images_empty_targets_never_touches_db()
    test_build_user_content_returns_same_string_when_no_images()
    test_build_user_content_builds_multimodal_blocks_with_images()
    test_run_attaches_image_when_enabled_and_available()
    test_run_fails_open_when_image_lookup_fails_entirely()
    test_run_retries_text_only_when_provider_rejects_multimodal_content()
    test_run_default_off_never_touches_db_or_filesystem()
    print("image-grounded answer tests passed")
