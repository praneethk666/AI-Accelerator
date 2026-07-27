"""Tests for docling_extract.py's local-engine (Unlimited-OCR) per-table-crop
escalation — the extraction.docling.table_engine config knob, separate from
vision_ocr.engine (whole-page rescue). See backend/extraction/unlimited_ocr.py.
"""
from unittest.mock import MagicMock, patch

from backend.extraction.docling_pdf.docling_extract import _local_table_engine, _local_table


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


if __name__ == "__main__":
    test_local_table_engine_defaults_to_false()
    test_local_table_engine_false_when_set_to_vlm()
    test_local_table_engine_true_when_set_to_local()
    test_local_table_crops_the_region_and_delegates_to_the_local_transcriber()
    print("docling table-engine tests passed")
