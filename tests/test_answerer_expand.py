from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from backend.retrieval.answerer import _is_thin, _expand_thin_chunks


def _chunk(chunk_id, text, token_count, summary=None, document_id="doc-1", page=61):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "token_count": token_count,
        "tags": {"summary": summary} if summary else {},
        "source_ref": {"page": page, "filename": "manual.pdf"},
    }


def test_chunk_without_summary_is_thin():
    assert _is_thin(_chunk("c1", "Alarm code\n11H", 5)) is True


def test_chunk_with_summary_and_enough_tokens_is_not_thin():
    assert _is_thin(_chunk("c1", "a real paragraph " * 10, 150, summary="A real summary.")) is False


def test_short_token_count_is_thin_even_with_a_summary():
    # defensive fallback: token_count below the floor is thin regardless
    assert _is_thin(_chunk("c1", "hi", 3, summary="short")) is True


def _fake_store(blocks):
    store = MagicMock()
    store.get_blocks.return_value = blocks
    return store


def test_expand_replaces_thin_chunk_text_with_full_page():
    thin = _chunk("c1", "Alarm code\n11H", 5)
    fine = _chunk("c2", "a real paragraph " * 10, 40, summary="ok", page=99)
    blocks = [
        {"block_id": "b1", "type": "heading", "text": "Alarm code\n11H",
         "source_ref": {"page": 61}},
        {"block_id": "b2", "type": "table", "text": "| Factor 1 | Servo unit failure |",
         "source_ref": {"page": 61}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(blocks)):
        out = _expand_thin_chunks([thin, fine])

    thin_out = next(c for c in out if c["chunk_id"] == "c1")
    fine_out = next(c for c in out if c["chunk_id"] == "c2")
    assert "Factor 1" in thin_out["text"]
    assert "Servo unit failure" in thin_out["text"]
    assert fine_out["text"] == fine["text"]  # untouched — wasn't thin


def test_no_thin_chunks_returns_original_list_untouched():
    fine = _chunk("c2", "a real paragraph " * 10, 40, summary="ok")
    out = _expand_thin_chunks([fine])
    assert out == [fine]


def test_expansion_failure_leaves_chunk_as_is():
    thin = _chunk("c1", "Alarm code\n11H", 5)
    store = MagicMock()
    store.get_blocks.side_effect = RuntimeError("db down")
    with patch("backend.storage.postgres_store.PostgresStore", return_value=store):
        out = _expand_thin_chunks([thin])
    assert out[0]["text"] == "Alarm code\n11H"  # unchanged, no crash


def test_document_id_as_uuid_object_still_expands():
    # psycopg returns a uuid.UUID (not a plain str) for this column elsewhere
    # (e.g. get_chunks_by_ids) — validated live, 21-Jul: an unmatched str/UUID
    # cache key silently skipped expansion even though the fetch succeeded.
    doc_id = uuid.uuid4()
    thin = _chunk("c1", "Alarm code\n11H", 5, document_id=doc_id)
    blocks = [{"block_id": "b1", "type": "table", "text": "Factor 1: Servo unit failure",
               "source_ref": {"page": 61}}]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(blocks)):
        out = _expand_thin_chunks([thin])
    assert "Factor 1" in out[0]["text"]


def test_multiple_thin_chunks_on_same_page_share_one_fetch():
    thin1 = _chunk("c1", "Alarm code\n11H", 5)
    thin2 = _chunk("c2", "State when alarm occurred.", 5)
    blocks = [{"block_id": "b1", "type": "text", "text": "full page content",
               "source_ref": {"page": 61}}]
    store = _fake_store(blocks)
    with patch("backend.storage.postgres_store.PostgresStore", return_value=store):
        out = _expand_thin_chunks([thin1, thin2])
    assert store.get_blocks.call_count == 1  # fetched once, reused for both
    assert all("full page content" in c["text"] for c in out)



def test_slide_sheet_expansion():
    # Test PPT slide thin chunk expansion
    slide_chunk = {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "text": "slide intro",
        "token_count": 5,
        "tags": {},
        "source_ref": {"slide": 4, "filename": "slides.pptx"},
    }
    slide_blocks = [
        {"block_id": "b1", "type": "text", "text": "slide intro", "source_ref": {"slide": 4}},
        {"block_id": "b2", "type": "text", "text": "slide detailed content", "source_ref": {"slide": 4}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(slide_blocks)):
        out = _expand_thin_chunks([slide_chunk])
    assert "slide detailed content" in out[0]["text"]

    # Test Excel sheet thin chunk expansion
    sheet_chunk = {
        "chunk_id": "c2",
        "document_id": "doc-1",
        "text": "sheet intro",
        "token_count": 5,
        "tags": {},
        "source_ref": {"sheet": "Overview", "filename": "data.xlsx"},
    }
    sheet_blocks = [
        {"block_id": "b1", "type": "text", "text": "sheet intro", "source_ref": {"sheet": "Overview"}},
        {"block_id": "b2", "type": "text", "text": "sheet data values", "source_ref": {"sheet": "Overview"}},
    ]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_store(sheet_blocks)):
        out = _expand_thin_chunks([sheet_chunk])
    assert "sheet data values" in out[0]["text"]


if __name__ == "__main__":
    test_chunk_without_summary_is_thin()
    test_chunk_with_summary_and_enough_tokens_is_not_thin()
    test_short_token_count_is_thin_even_with_a_summary()
    test_expand_replaces_thin_chunk_text_with_full_page()
    test_no_thin_chunks_returns_original_list_untouched()
    test_expansion_failure_leaves_chunk_as_is()
    test_document_id_as_uuid_object_still_expands()
    test_multiple_thin_chunks_on_same_page_share_one_fetch()
    test_slide_sheet_expansion()
    print("answerer expansion tests passed")
