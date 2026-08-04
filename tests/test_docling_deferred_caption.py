"""Tests for caption_deferred_figures() -- resolves figures _figure_block(defer=True)
deliberately left uncaptioned (metadata.caption_deferred=True). This only fires for
pages Docling extracted ZERO text from at crop-time (do_ocr:false + no text layer =
scanned), which previously meant the VLM figure gate ran completely blind (no page
context at all) on exactly the pages that need it most. docling_pdf/tool.py calls
this AFTER route_and_rescue() has produced the page's real OCR/VLM text.

Run: pytest tests/test_docling_deferred_caption.py
"""
import os
from unittest.mock import patch

import pytest

from backend.extraction.docling_pdf.docling_extract import caption_deferred_figures


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _deferred_block(block_id, page, image_path=None):
    return {
        "block_id": block_id,
        "type": "image_caption",
        "text": "[figure]",
        "source_ref": {"page": page, "bbox": [10, 10, 50, 50]},
        "metadata": {
            "pending_vision": True,
            "caption_deferred": True,
            "image_path": image_path or f"/images/doc1/{block_id}.png",
        },
    }


def _text_block(page, text):
    return {"type": "text", "text": text, "source_ref": {"page": page}, "metadata": {}}


def _write_fake_image(image_path: str, content: bytes = b"fakepngbytes"):
    fs_path = os.path.join("uploads", "images", *image_path.split("/")[2:])
    os.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with open(fs_path, "wb") as f:
        f.write(content)
    return fs_path


def test_noop_when_nothing_deferred():
    blocks = [_text_block(1, "hello"), {"type": "image_caption", "text": "already done",
              "source_ref": {"page": 1}, "metadata": {}}]
    out = caption_deferred_figures(blocks, {})
    assert out == blocks


def test_resolves_pending_figure_with_real_page_text():
    b = _deferred_block("fig1", page=2)
    _write_fake_image(b["metadata"]["image_path"])
    blocks = [_text_block(2, "Battery removal procedure"), b]

    captured = {}

    def fake_gate(png, page_context, config):
        captured["page_context"] = page_context
        return {"keep": True, "kind": "photo", "caption": "a battery pack"}

    with patch("backend.extraction.vision_ocr.classify_caption_crop", side_effect=fake_gate):
        out = caption_deferred_figures(blocks, {})

    resolved = next(x for x in out if x.get("block_id") == "fig1")
    assert resolved["text"] == "a battery pack"
    assert resolved["metadata"]["pending_vision"] is False
    assert resolved["metadata"]["caption_deferred"] is False
    assert resolved["metadata"]["figure_kind"] == "photo"
    assert "Battery removal procedure" in captured["page_context"]


def test_page_text_scoped_to_the_figures_own_page():
    b = _deferred_block("fig1", page=2)
    _write_fake_image(b["metadata"]["image_path"])
    blocks = [_text_block(1, "page one text -- should NOT leak in"),
              _text_block(2, "page two text"), b]

    captured = {}

    def fake_gate(png, page_context, config):
        captured["page_context"] = page_context
        return {"keep": True, "kind": "photo", "caption": "x"}

    with patch("backend.extraction.vision_ocr.classify_caption_crop", side_effect=fake_gate):
        caption_deferred_figures(blocks, {})

    assert "page two text" in captured["page_context"]
    assert "page one text" not in captured["page_context"]


def test_dropped_furniture_removed_and_image_deleted():
    b = _deferred_block("fig1", page=2)
    fs_path = _write_fake_image(b["metadata"]["image_path"])
    blocks = [b]

    with patch("backend.extraction.vision_ocr.classify_caption_crop",
              return_value={"keep": False, "kind": "logo", "caption": ""}):
        out = caption_deferred_figures(blocks, {})

    assert out == []
    assert not os.path.exists(fs_path)


def test_missing_image_file_leaves_block_pending_no_crash():
    b = _deferred_block("fig1", page=2, image_path="/images/doc1/does-not-exist.png")
    blocks = [b]

    out = caption_deferred_figures(blocks, {})

    assert out == [b]
    assert b["metadata"]["caption_deferred"] is True  # untouched, not falsely resolved


def test_gate_exception_leaves_block_pending_no_crash():
    b = _deferred_block("fig1", page=2)
    _write_fake_image(b["metadata"]["image_path"])
    blocks = [b]

    with patch("backend.extraction.vision_ocr.classify_caption_crop",
              side_effect=RuntimeError("boom")):
        out = caption_deferred_figures(blocks, {})

    assert out == [b]
    assert b["metadata"]["caption_deferred"] is True


def test_only_deferred_blocks_touched_others_untouched():
    already_done = {"type": "image_caption", "text": "done already",
                    "source_ref": {"page": 1}, "metadata": {}}
    blocks = [already_done]

    with patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        out = caption_deferred_figures(blocks, {})

    mock_gate.assert_not_called()
    assert out == [already_done]


def test_multiple_pending_figures_resolved_concurrently():
    b1 = _deferred_block("fig1", page=1)
    b2 = _deferred_block("fig2", page=1)
    _write_fake_image(b1["metadata"]["image_path"])
    _write_fake_image(b2["metadata"]["image_path"])
    blocks = [b1, b2]

    with patch("backend.extraction.vision_ocr.classify_caption_crop",
              return_value={"keep": True, "kind": "diagram", "caption": "ok"}):
        out = caption_deferred_figures(blocks, {"vision": {"max_concurrency": 4}})

    assert len(out) == 2
    assert all(x["text"] == "ok" for x in out)


def test_large_document_lazy_figures_are_never_auto_resolved():
    # Real design, 3-Aug: size-based lazy figure captioning defers PERMANENTLY as
    # a cost-control policy, not because OCR text isn't ready yet -- this pass
    # must leave them alone (an agent resolves them on demand via view_page_image).
    b = _deferred_block("fig1", page=1)
    b["metadata"]["defer_reason"] = "large_document_lazy"
    _write_fake_image(b["metadata"]["image_path"])
    blocks = [_text_block(1, "plenty of real page text here"), b]

    with patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        out = caption_deferred_figures(blocks, {})

    mock_gate.assert_not_called()
    resolved = next(x for x in out if x.get("block_id") == "fig1")
    assert resolved["metadata"]["caption_deferred"] is True   # still deferred


def test_scanned_no_text_reason_still_auto_resolves():
    # The OTHER reason (scanned page, no text at crop-time) must keep working
    # exactly as before -- only large_document_lazy is permanently skipped.
    b = _deferred_block("fig1", page=1)
    b["metadata"]["defer_reason"] = "scanned_no_text"
    _write_fake_image(b["metadata"]["image_path"])
    blocks = [_text_block(1, "real OCR text now available"), b]

    with patch("backend.extraction.vision_ocr.classify_caption_crop",
              return_value={"keep": True, "kind": "photo", "caption": "resolved"}):
        out = caption_deferred_figures(blocks, {})

    resolved = next(x for x in out if x.get("block_id") == "fig1")
    assert resolved["text"] == "resolved"
    assert resolved["metadata"]["caption_deferred"] is False
