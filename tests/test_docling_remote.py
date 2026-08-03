"""Tests for backend/extraction/docling_remote.py's post-fetch table escalation —
real gap found live, 3-Aug: the server flags complex tables with
metadata.escalation_hint="vlm_or_local" but nothing ever consumed that hint, so
every complex table returned by the remote Docling server was silently left as
raw TableFormer/pymupdf output even with extraction.docling.table_engine: local
configured. Fixed by re-cropping flagged tables from the local pdf_path and
escalating through the same _local_table() the local extraction path uses.
"""
from unittest.mock import MagicMock, patch

from backend.extraction.docling_remote import extract_docling_remote


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


if __name__ == "__main__":
    test_flagged_table_gets_escalated_via_local_table()
    test_table_without_escalation_hint_is_left_untouched()
    test_escalation_skipped_entirely_when_local_table_engine_disabled()
    test_escalation_failure_on_one_table_leaves_original_block_and_continues()
    print("docling_remote escalation tests passed")
