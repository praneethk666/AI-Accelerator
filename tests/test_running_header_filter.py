"""Tests for docling_extract.py's _is_running_header_leak — real bug found live,
27-Jul, on the servo manual: a two-column running header (chapter number left,
chapter title right, joined by a huge run of literal spaces) wasn't labeled
page_header/page_footer by Docling, so it slipped through _DROP_LABELS and
surfaced as an ordinary text block on 23 of 105 real pages, producing a
near-empty noise chunk whenever it landed right before a table/figure.
"""
from backend.extraction.docling_pdf.docling_extract import _is_running_header_leak


def test_real_running_header_is_dropped():
    # Exact shape from the servo manual, page 31 (chapter-3 running header).
    text = "3" + " " * 152 + "Wiring"
    bbox = [56.7, 39.04, 552.58, 51.42]
    assert _is_running_header_leak(text, bbox) is True


def test_real_running_header_with_period_is_dropped():
    text = "2. " + " " * 140 + "Specifications"
    bbox = [42.5, 40.19, 799.9, 51.7]
    assert _is_running_header_leak(text, bbox) is True


def test_real_subsection_heading_is_not_dropped():
    # Genuine content heading, same chapter, no abnormal gap.
    text = "3.1 Wiring for power supplies"
    bbox = [46.79, 65.42, 260.5, 79.6]
    assert _is_running_header_leak(text, bbox) is False


def test_real_content_far_from_top_margin_is_not_dropped():
    text = "Motor model setting error"
    bbox = [50, 150, 300, 160]
    assert _is_running_header_leak(text, bbox) is False


def test_near_top_heading_without_huge_gap_is_not_dropped():
    text = "3.1.6 Wiring for motor power supply"
    bbox = [70.92, 77.0, 263.3, 87.6]
    assert _is_running_header_leak(text, bbox) is False


def test_missing_bbox_is_not_dropped():
    assert _is_running_header_leak("3" + " " * 20 + "Wiring", None) is False


def test_long_real_content_with_a_small_gap_is_not_dropped():
    # A genuine paragraph near the top margin with normal-width spacing (not a
    # 15+ space run) must not be caught just because it's short-ish and high up.
    text = "Safety precautions for this chapter are listed below."
    bbox = [50, 40, 500, 55]
    assert _is_running_header_leak(text, bbox) is False


if __name__ == "__main__":
    test_real_running_header_is_dropped()
    test_real_running_header_with_period_is_dropped()
    test_real_subsection_heading_is_not_dropped()
    test_real_content_far_from_top_margin_is_not_dropped()
    test_near_top_heading_without_huge_gap_is_not_dropped()
    test_missing_bbox_is_not_dropped()
    test_long_real_content_with_a_small_gap_is_not_dropped()
    print("running-header filter tests passed")
