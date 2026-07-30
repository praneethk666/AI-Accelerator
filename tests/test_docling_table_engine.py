"""Tests for docling_extract.py's local-engine (Unlimited-OCR) per-table-crop
escalation — the extraction.docling.table_engine config knob, separate from
vision_ocr.engine (whole-page rescue). See backend/extraction/unlimited_ocr.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from backend.extraction.docling_pdf.docling_extract import (
    _local_table_engine, _local_table, _local_table_ocr_engine, _table_has_span,
    _table_is_complex, _split_bbox_vertically, _merge_split_table_data,
    _sparse_column_indices, _cross_check_sparse_columns,
)


def _cell(row_span=1, col_span=1, column_header=False, text="", bbox=None):
    return SimpleNamespace(row_span=row_span, col_span=col_span,
                            column_header=column_header, text=text, bbox=bbox)


def _table(cells):
    return SimpleNamespace(data=SimpleNamespace(table_cells=cells))


def test_local_table_engine_defaults_to_false():
    assert _local_table_engine({}) is False
    assert _local_table_engine({"extraction": {"docling": {}}}) is False


def test_local_table_engine_false_when_set_to_vlm():
    cfg = {"extraction": {"docling": {"table_engine": "vlm"}}}
    assert _local_table_engine(cfg) is False


def test_local_table_engine_true_when_set_to_local():
    cfg = {"extraction": {"docling": {"table_engine": "local"}}}
    assert _local_table_engine(cfg) is True


# ── _local_table_ocr_engine: which self-hosted model handles table_engine: local ──

def test_local_table_ocr_engine_defaults_to_unlimited_ocr():
    assert _local_table_ocr_engine({}) == "unlimited_ocr"
    assert _local_table_ocr_engine({"extraction": {"docling": {}}}) == "unlimited_ocr"


def test_local_table_ocr_engine_reads_explicit_paddleocr_vl():
    cfg = {"extraction": {"docling": {"local_table_ocr_engine": "paddleocr_vl"}}}
    assert _local_table_ocr_engine(cfg) == "paddleocr_vl"


def test_local_table_routes_to_paddleocr_vl_when_configured():
    fake_crop = MagicMock()
    fake_crop.crop_region.return_value = b"cropped-png-bytes"
    table_data = {"headers": ["Code", "Name"], "rows": [["F7H", "Motor error"]]}
    cfg = {"extraction": {"docling": {"local_table_ocr_engine": "paddleocr_vl"}},
           "vision_ocr": {}}

    with patch("backend.vision.pdf_cropper.PDFCropper", return_value=fake_crop), \
         patch("backend.extraction.paddleocr_vl.transcribe_table_paddleocr_vl",
               return_value=table_data) as mock_paddle, \
         patch("backend.extraction.unlimited_ocr.transcribe_table_local") as mock_uo:
        result = _local_table("/fake/manual.pdf", 5, [10, 20, 30, 40], cfg)

    mock_paddle.assert_called_once_with(b"cropped-png-bytes", cfg)
    mock_uo.assert_not_called()  # the other engine's client must not be touched
    assert result == table_data


def test_local_table_crops_the_region_and_delegates_to_the_local_transcriber():
    fake_crop = MagicMock()
    fake_crop.crop_region.return_value = b"cropped-png-bytes"
    table_data = {"headers": ["Code", "Name"], "rows": [["F7H", "Motor error"]]}

    with patch("backend.vision.pdf_cropper.PDFCropper", return_value=fake_crop), \
         patch("backend.extraction.unlimited_ocr.transcribe_table_local",
               return_value=table_data) as mock_local:
        result = _local_table("/fake/manual.pdf", 5, [10, 20, 30, 40], {"vision_ocr": {}})

    fake_crop.crop_region.assert_called_once_with("/fake/manual.pdf", 5, [10, 20, 30, 40])
    mock_local.assert_called_once_with(b"cropped-png-bytes", {"vision_ocr": {}})
    assert result == table_data


# ── split-crop fallback: real finding, 27-Jul -- some tables OOM at every
# base_size because the actual bottleneck is decoder/output-length, not input
# resolution (confirmed live: identical "Tried to allocate" size regardless of
# base_size). Splitting into row-halves means half the output sequence length,
# which is the one thing that actually targets this bottleneck. ──────────────

def test_split_bbox_vertically_splits_at_midpoint_with_overlap():
    top, bottom = _split_bbox_vertically([0, 0, 100, 200], overlap_frac=0.15)
    # midpoint is y=100; 15% of height (200) = 30pt overlap on each side
    assert top == [0, 0, 100, 130]
    assert bottom == [0, 70, 100, 200]


def test_split_bbox_vertically_preserves_x_bounds():
    top, bottom = _split_bbox_vertically([10, 0, 90, 200])
    assert top[0] == 10 and top[2] == 90
    assert bottom[0] == 10 and bottom[2] == 90


def test_merge_split_table_data_concatenates_rows_in_order():
    top = {"headers": ["Code", "Name"], "rows": [["F7H", "Motor error"]]}
    bottom = {"headers": [], "rows": [["F8H", "Parameter error"]]}
    merged = _merge_split_table_data(top, bottom)
    assert merged["headers"] == ["Code", "Name"]
    assert merged["rows"] == [["F7H", "Motor error"], ["F8H", "Parameter error"]]


def test_merge_split_table_data_drops_an_exact_duplicate_seam_row():
    # The overlap band means the row right at the seam can get transcribed by
    # BOTH halves -- must not ship it twice.
    top = {"headers": ["Code"], "rows": [["F7H"], ["F8H"]]}
    bottom = {"headers": [], "rows": [["F8H"], ["F9H"]]}
    merged = _merge_split_table_data(top, bottom)
    assert merged["rows"] == [["F7H"], ["F8H"], ["F9H"]]


def test_merge_split_table_data_handles_one_half_missing():
    top = {"headers": ["Code"], "rows": [["F7H"]]}
    assert _merge_split_table_data(top, None) == top
    assert _merge_split_table_data(None, top) == top


def test_local_table_falls_back_to_split_crop_when_whole_table_exhausts_retries():
    fake_crop = MagicMock()
    fake_crop.crop_region.side_effect = [
        b"whole-crop-png", b"top-crop-png", b"bottom-crop-png",
    ]
    whole_table_oom = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=MagicMock(status_code=503))
    top_td = {"headers": ["Code", "Name"], "rows": [["F7H", "Motor error"]]}
    bottom_td = {"headers": [], "rows": [["F8H", "Parameter error"]]}

    with patch("backend.vision.pdf_cropper.PDFCropper", return_value=fake_crop), \
         patch("backend.extraction.unlimited_ocr.transcribe_table_local",
               side_effect=[whole_table_oom, top_td, bottom_td]) as mock_local:
        result = _local_table("/fake/manual.pdf", 59, [0, 0, 100, 200], {"vision_ocr": {}})

    assert mock_local.call_count == 3
    assert fake_crop.crop_region.call_count == 3
    # second/third crop_region calls used the split (top/bottom) bboxes, not the
    # original whole-table bbox again
    whole_call, top_call, bottom_call = fake_crop.crop_region.call_args_list
    assert whole_call.args[2] == [0, 0, 100, 200]
    assert top_call.args[2] != [0, 0, 100, 200]
    assert bottom_call.args[2] != [0, 0, 100, 200]
    assert result == {
        "headers": ["Code", "Name"],
        "rows": [["F7H", "Motor error"], ["F8H", "Parameter error"]],
    }


def test_local_table_reraises_a_non_oom_http_error_without_splitting():
    fake_crop = MagicMock()
    fake_crop.crop_region.return_value = b"whole-crop-png"
    non_oom_error = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock(status_code=500))

    with patch("backend.vision.pdf_cropper.PDFCropper", return_value=fake_crop), \
         patch("backend.extraction.unlimited_ocr.transcribe_table_local",
               side_effect=non_oom_error) as mock_local:
        raised = False
        try:
            _local_table("/fake/manual.pdf", 59, [0, 0, 100, 200], {"vision_ocr": {}})
        except httpx.HTTPStatusError:
            raised = True

    assert raised
    mock_local.assert_called_once()  # no split attempt for a non-OOM error


# ── _table_has_span: the real gap found 27-Jul on the servo manual's alarm table ──

def test_table_has_span_true_for_rowspan_cell():
    # Real shape: the servo manual's F7H-FFH alarm table has a rowspan=3 cell for
    # each alarm name/code, spanning the 3 physical rows of its description.
    t = _table([_cell(row_span=3, text="Motor model setting error"), _cell(text="F7H")])
    assert _table_has_span(t) is True


def test_table_has_span_true_for_colspan_cell():
    t = _table([_cell(col_span=2, text="Merged header")])
    assert _table_has_span(t) is True


def test_table_has_span_false_for_simple_ruled_table():
    t = _table([_cell(text="a"), _cell(text="b"), _cell(text="c")])
    assert _table_has_span(t) is False


def test_table_has_span_false_for_empty_table():
    assert _table_has_span(_table([])) is False


def test_table_is_complex_still_true_for_spanning_cells():
    # _table_is_complex must still report True for span (used for the vlm/local
    # escalation decision itself) -- only the PYMUPDF-FIRST preference changes.
    t = _table([_cell(row_span=3, text="x")])
    assert _table_is_complex(t) is True
    assert _table_has_span(t) is True


def test_table_is_complex_true_for_list_heavy_cells_without_span():
    # A DIFFERENT complexity reason (list-heavy data cells) with NO spanning cells
    # at all -- _table_has_span must be False here even though _table_is_complex
    # is True, since pymupdf's ruled-line reading isn't structurally wrong for
    # this case (unlike rowspan) and should still be tried first in "auto" mode.
    long_text = "1. First step\n2. Second step\n3. Third step " + ("x" * 150)
    t = _table([
        _cell(text=long_text),
        _cell(text="1. Item one\n2. Item two\n3. Item three"),
    ])
    assert _table_has_span(t) is False
    assert _table_is_complex(t) is True


# ── sparse-column cross-check: real bug found live, 28-Jul, page 54's alarm
# table -- PaddleOCR-VL dropped the "Indication" 7-segment-display icon column
# on 9/11 rows while every other column (including short ones like "Code")
# stayed fully populated. A naive "short values" heuristic would ALSO flag the
# correct Code column, which is why _sparse_column_indices checks emptiness,
# not value length. ────────────────────────────────────────────────────────────

def test_sparse_column_indices_flags_mostly_empty_column_with_populated_siblings():
    # Real shape: Code is always populated (correct); Indication is blank on
    # most rows (the actual bug) even though it's a real column with content.
    td = {
        "headers": ["Alarm name", "Code", "Indication"],
        "rows": [
            ["Sensor initial process error", "80H", ""],
            ["Sensor signal disconnection", "81H", ""],
            ["No response from sensor", "82H", "[F25]"],
            ["Sensor error 1", "83H", ""],
        ],
    }
    assert _sparse_column_indices(td) == [2]


def test_sparse_column_indices_does_not_flag_short_but_fully_populated_column():
    td = {
        "headers": ["Alarm name", "Code"],
        "rows": [["Sensor initial process error", "80H"],
                 ["Sensor signal disconnection", "81H"]],
    }
    assert _sparse_column_indices(td) == []


def test_sparse_column_indices_does_not_flag_a_table_thats_sparse_everywhere():
    # No column is well-populated -- nothing to compare against, don't flag.
    td = {
        "headers": ["A", "B"],
        "rows": [["", ""], ["", "x"], ["", ""], ["y", ""]],
    }
    assert _sparse_column_indices(td) == []


def test_cross_check_sparse_columns_fills_gaps_from_vlm_reading_matched_by_row_id():
    td = {
        "headers": ["Alarm name", "Code", "Indication"],
        "rows": [
            ["No response from sensor", "82H", ""],
            ["Sensor error 1", "83H", ""],
        ],
    }
    vlm_markdown = (
        "| Alarm name | Code | Indication |\n"
        "| --- | --- | --- |\n"
        "| No response from sensor | 82H | 8<=>2 |\n"
        "| Sensor error 1 | 83H | 8<=>3 |\n"
    )
    with patch("backend.extraction.docling_pdf.docling_extract._vlm_table_from_png",
               return_value=vlm_markdown) as mock_vlm:
        result = _cross_check_sparse_columns(td, b"png-bytes", {"vision_ocr": {}})

    mock_vlm.assert_called_once_with(b"png-bytes", {"vision_ocr": {}})
    assert result["rows"][0][2] == "8<=>2"
    assert result["rows"][1][2] == "8<=>3"


def test_cross_check_sparse_columns_never_overwrites_an_existing_value():
    td = {
        "headers": ["Alarm name", "Code", "Indication"],
        "rows": [["No response from sensor", "82H", "already-here"],
                 ["Sensor error 1", "83H", ""]],
    }
    vlm_markdown = (
        "| Alarm name | Code | Indication |\n"
        "| --- | --- | --- |\n"
        "| No response from sensor | 82H | different-value |\n"
        "| Sensor error 1 | 83H | 8<=>3 |\n"
    )
    with patch("backend.extraction.docling_pdf.docling_extract._vlm_table_from_png",
               return_value=vlm_markdown):
        result = _cross_check_sparse_columns(td, b"png-bytes", {"vision_ocr": {}})

    assert result["rows"][0][2] == "already-here"  # untouched
    assert result["rows"][1][2] == "8<=>3"          # gap filled


def test_cross_check_sparse_columns_skips_vlm_call_when_nothing_is_sparse():
    td = {
        "headers": ["Alarm name", "Code"],
        "rows": [["Sensor initial process error", "80H"]],
    }
    with patch("backend.extraction.docling_pdf.docling_extract._vlm_table_from_png") as mock_vlm:
        result = _cross_check_sparse_columns(td, b"png-bytes", {"vision_ocr": {}})

    mock_vlm.assert_not_called()
    assert result == td


def test_cross_check_sparse_columns_tolerates_vlm_failure():
    td = {
        "headers": ["Alarm name", "Code", "Indication"],
        "rows": [
            ["No response from sensor", "82H", ""],
            ["Sensor error 1", "83H", ""],
        ],
    }
    with patch("backend.extraction.docling_pdf.docling_extract._vlm_table_from_png",
               side_effect=RuntimeError("vision call failed")):
        result = _cross_check_sparse_columns(td, b"png-bytes", {"vision_ocr": {}})

    assert result == td  # unchanged, no exception propagated


if __name__ == "__main__":
    test_local_table_engine_defaults_to_false()
    test_local_table_engine_false_when_set_to_vlm()
    test_local_table_engine_true_when_set_to_local()
    test_local_table_crops_the_region_and_delegates_to_the_local_transcriber()
    test_table_has_span_true_for_rowspan_cell()
    test_table_has_span_true_for_colspan_cell()
    test_table_has_span_false_for_simple_ruled_table()
    test_table_has_span_false_for_empty_table()
    test_table_is_complex_still_true_for_spanning_cells()
    test_table_is_complex_true_for_list_heavy_cells_without_span()
    print("docling table-engine tests passed")
