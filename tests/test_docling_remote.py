"""Tests for backend/extraction/docling_remote.py's post-fetch table escalation —
real gap found live, 3-Aug: the server flags complex tables with
metadata.escalation_hint="vlm_or_local" but nothing ever consumed that hint, so
every complex table returned by the remote Docling server was silently left as
raw TableFormer/pymupdf output even with extraction.docling.table_engine: local
configured. Fixed by re-cropping flagged tables from the local pdf_path and
escalating through the same _local_table() the local extraction path uses.
"""
from unittest.mock import MagicMock, patch

import pytest

from backend.extraction.docling_remote import extract_docling_remote


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """_figure_block writes crops to a relative 'uploads/images/<doc_id>/' path."""
    monkeypatch.chdir(tmp_path)


def _server_response(blocks, n_pages=1):
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = {"blocks": blocks, "n_pages": n_pages, "elapsed_s": 1.2}
    return resp


def _table_block(escalation_hint=None, page=1, bbox=None, table_data=None):
    meta = {"table_source": "tableformer", "table_complex": bool(escalation_hint)}
    if escalation_hint:
        meta["escalation_hint"] = escalation_hint
    return {
        "block_id": "b1", "document_id": "d1", "type": "table",
        "text": "| a | b |", "table_data": table_data or {"headers": ["a"], "rows": [["x"]]},
        "source_ref": {"filename": "m.pdf", "page": page, "sheet": None, "slide": None, "bbox": bbox or [0, 0, 10, 10]},
        "metadata": meta,
    }


def _config(table_engine="local"):
    return {"extraction": {"docling": {
        "server_url": "http://gpu-box:8083", "table_engine": table_engine,
    }}}


def test_unresolved_server_url_fails_fast_with_clear_message():
    # real finding, 3-Aug: an unsubstituted ${DOCLING_SERVER_URL} used to surface
    # as a cryptic urllib3 "Invalid URL ... No scheme supplied" that reads like
    # the GPU server itself is down. Must fail immediately and obviously instead.
    cfg = _config()
    cfg["extraction"]["docling"]["server_url"] = "${DOCLING_SERVER_URL}"
    with patch("requests.post") as mock_post:
        try:
            extract_docling_remote("/fake/m.pdf", "d1", cfg)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "unresolved" in str(e)
            assert "${DOCLING_SERVER_URL}" in str(e)
    mock_post.assert_not_called()   # never even attempted the network call


def test_flagged_table_gets_escalated_via_local_table():
    blocks = [_table_block(escalation_hint="vlm_or_local")]
    improved_td = {"headers": ["a"], "rows": [["real value"]]}
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table",
               return_value=improved_td) as mock_local, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=True):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    mock_local.assert_called_once_with("/fake/m.pdf", 1, [0, 0, 10, 10], _config())
    assert result[0]["table_data"] == improved_td


def test_table_without_escalation_hint_is_left_untouched():
    original_td = {"headers": ["a"], "rows": [["x"]]}
    blocks = [_table_block(escalation_hint=None, table_data=original_td)]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table") as mock_local, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=True):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    mock_local.assert_not_called()
    assert result[0]["table_data"] == original_td


def test_escalation_skipped_entirely_when_local_table_engine_disabled():
    blocks = [_table_block(escalation_hint="vlm_or_local")]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table") as mock_local, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        extract_docling_remote("/fake/m.pdf", "d1", _config(table_engine="vlm"))

    mock_local.assert_not_called()


def test_escalation_failure_on_one_table_leaves_original_block_and_continues():
    original_td = {"headers": ["a"], "rows": [["x"]]}
    blocks = [_table_block(escalation_hint="vlm_or_local", table_data=original_td)]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table",
               side_effect=RuntimeError("gpu box down")), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=True):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    assert result[0]["table_data"] == original_td  # unchanged, no crash


# ---------------------------------------------------------------------------
# Figure captioning -- real bug found live, 3-Aug: the server only LOCATES
# figures (bbox, text literally "[figure]"); nothing downstream ever actually
# cropped or captioned them for remote mode. Every figure from a remote-
# extracted document stayed a useless placeholder forever. Fixed via
# _caption_remote_figures(), reusing the same _figure_block() gate local mode
# uses, re-cropped from the local pdf_path with the bbox the server found.
# ---------------------------------------------------------------------------

def _fig_block(page=1, bbox=None):
    return {
        "block_id": "fig1", "document_id": "d1", "type": "image_caption",
        "text": "[figure]", "table_data": None,
        "source_ref": {"filename": "m.pdf", "page": page, "sheet": None,
                       "slide": None, "bbox": bbox or [10, 10, 100, 100]},
        "metadata": {},
    }


def _text_block(page=1, text="some page text"):
    return {
        "block_id": "t1", "document_id": "d1", "type": "text", "text": text,
        "source_ref": {"filename": "m.pdf", "page": page, "sheet": None, "slide": None, "bbox": None},
        "metadata": {},
    }


def test_figure_with_page_text_is_captioned_immediately():
    blocks = [_text_block(page=1), _fig_block(page=1)]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        mock_gate.return_value = {"keep": True, "kind": "photo", "caption": "a photo"}
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    fig = next(b for b in result if b["type"] == "image_caption")
    assert fig["text"] == "a photo"
    mock_gate.assert_called_once()
    assert mock_gate.call_args.args[1] == "some page text"


def test_figure_on_textless_page_is_deferred_not_captioned_blind():
    blocks = [_fig_block(page=1)]   # no text block on this page at all
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    mock_gate.assert_not_called()
    fig = next(b for b in result if b["type"] == "image_caption")
    assert fig["metadata"]["caption_deferred"] is True
    assert fig["metadata"]["image_path"]   # still cropped+saved, just not captioned yet


def test_figure_gated_as_furniture_is_dropped():
    blocks = [_text_block(page=1), _fig_block(page=1)]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop",
               return_value={"keep": False, "kind": "logo", "caption": ""}), \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    assert not any(b["type"] == "image_caption" for b in result)
    assert any(b["type"] == "text" for b in result)   # unrelated blocks untouched


def test_no_figures_returned_is_a_noop():
    blocks = [_text_block(page=1)]
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    mock_gate.assert_not_called()
    assert result == blocks


def test_page_rescue_disabled_never_defers():
    # deferring only makes sense if a LATER pass (route_and_rescue) will actually
    # resolve it -- with page_rescue off, that pass never runs, so a deferred
    # figure would be stuck forever. Must caption immediately (best-effort) instead.
    blocks = [_fig_block(page=1)]   # no text on page -> would normally defer
    cfg = _config()
    cfg["extraction"]["docling"]["page_rescue"] = False
    with patch("requests.post", return_value=_server_response(blocks)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        mock_gate.return_value = {"keep": True, "kind": "photo", "caption": "x"}
        result = extract_docling_remote("/fake/m.pdf", "d1", cfg)

    mock_gate.assert_called_once()
    fig = next(b for b in result if b["type"] == "image_caption")
    assert fig["metadata"].get("caption_deferred") is not True


def test_size_based_lazy_defers_figures_on_large_document_even_with_page_text():
    # Real design, 3-Aug: above defer_figures_above_pages, figures are deferred
    # PERMANENTLY as a cost-control policy -- even on pages that DO have text
    # (unlike the scanned/no-text case, which only defers because a later pass
    # will resolve it).
    blocks = [_text_block(page=1), _fig_block(page=1)]
    cfg = _config()
    cfg["extraction"]["docling"]["figure_caption_mode"] = "size_based"
    cfg["extraction"]["docling"]["defer_figures_above_pages"] = 250
    with patch("requests.post", return_value=_server_response(blocks, n_pages=1147)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        result = extract_docling_remote("/fake/m.pdf", "d1", cfg)

    mock_gate.assert_not_called()
    fig = next(b for b in result if b["type"] == "image_caption")
    assert fig["metadata"]["caption_deferred"] is True
    assert fig["metadata"]["defer_reason"] == "large_document_lazy"
    assert "view_page_image" in fig["text"]


def test_size_based_lazy_does_not_apply_below_threshold():
    blocks = [_text_block(page=1), _fig_block(page=1)]
    cfg = _config()
    cfg["extraction"]["docling"]["figure_caption_mode"] = "size_based"
    cfg["extraction"]["docling"]["defer_figures_above_pages"] = 250
    with patch("requests.post", return_value=_server_response(blocks, n_pages=50)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        mock_gate.return_value = {"keep": True, "kind": "photo", "caption": "x"}
        result = extract_docling_remote("/fake/m.pdf", "d1", cfg)

    mock_gate.assert_called_once()
    fig = next(b for b in result if b["type"] == "image_caption")
    assert fig["metadata"].get("caption_deferred") is not True


def test_eager_mode_is_the_default_regardless_of_page_count():
    blocks = [_text_block(page=1), _fig_block(page=1)]
    with patch("requests.post", return_value=_server_response(blocks, n_pages=1147)), \
         patch("builtins.open", MagicMock()), \
         patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepng"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate, \
         patch("backend.extraction.docling_pdf.docling_extract._local_table_engine",
               return_value=False):
        mock_gate.return_value = {"keep": True, "kind": "photo", "caption": "x"}
        result = extract_docling_remote("/fake/m.pdf", "d1", _config())

    mock_gate.assert_called_once()   # figure_caption_mode not set -> "eager" default


if __name__ == "__main__":
    test_flagged_table_gets_escalated_via_local_table()
    test_table_without_escalation_hint_is_left_untouched()
    test_escalation_skipped_entirely_when_local_table_engine_disabled()
    test_escalation_failure_on_one_table_leaves_original_block_and_continues()
    print("docling_remote escalation tests passed")
