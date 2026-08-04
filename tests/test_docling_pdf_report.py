"""Tests for docling_pdf/tool.py's _new_report() -- real bug found live, 3-Aug,
during a real 100-page circuit-diagram ingestion: report["tables"] was
initialized WITHOUT a "vlm_escalated" key, but vision_ocr.py's rescue path does
`report["tables"]["vlm_escalated"] += tbl_count` (a bare +=, not .get()-guarded)
whenever a rescued/tiled page has tables. Every such page raised KeyError,
silently caught by rescue's broad except and logged as "... failed ('vlm_escalated')"
-- pages 11, 12, 13, 17+ all fell back to "keeping originals" instead of getting
the intended VLM rescue/tiling, a real, silent quality loss on a live run.

Run: pytest tests/test_docling_pdf_report.py
"""
from backend.extraction.docling_pdf.tool import _new_report


def test_report_tables_has_vlm_escalated_key():
    report = _new_report()
    report["tables"]["vlm_escalated"] += 1   # must not raise KeyError
    assert report["tables"]["vlm_escalated"] == 1


def test_report_tables_all_keys_present_for_static_increment_sites():
    # Every key any static (non-.get()) increment in vision_ocr.py/docling_extract.py
    # touches must exist at init -- a bare += on a missing key always raises,
    # regardless of value, unlike the .get(key, 0)-guarded dynamic sites elsewhere.
    report = _new_report()
    for key in ("total", "tableformer", "pymupdf", "vlm", "vlm_escalated"):
        assert key in report["tables"], f"tables.{key} missing from _new_report()"


def test_report_figures_and_pages_and_stitch_keys_present():
    report = _new_report()
    for key in ("total", "proposed", "docling", "yolo_added", "dropped_by_gate"):
        assert key in report["figures"]
    for key in ("total", "digital_kept", "vlm_rescued", "paddle_fallback", "rescued"):
        assert key in report["pages"]
    for key in ("merged", "arbitrations"):
        assert key in report["stitch"]
