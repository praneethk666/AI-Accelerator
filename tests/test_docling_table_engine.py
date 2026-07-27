"""Tests for docling_extract.py's local-engine (Unlimited-OCR) per-table-crop
escalation — the extraction.docling.table_engine config knob, separate from
vision_ocr.engine (whole-page rescue). See backend/extraction/unlimited_ocr.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.extraction.docling_pdf.docling_extract import (
    _local_table_engine, _local_table, _table_has_span, _table_is_complex,
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
