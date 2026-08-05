"""Tests for PER-IMAGE size-based lazy figure captioning
(extraction.docling.eager_caption_min_area_frac) -- added 4-Aug, a separate axis
from the existing per-DOCUMENT one (defer_figures_above_pages). Real gap: we
already know which page/bbox every figure belongs to and can pull it on demand
(view_page_image), so eagerly VLM-captioning every small/minor figure in a
normal-sized document is real, avoidable ingestion cost.

Run: pytest tests/test_docling_image_size_defer.py
"""
from __future__ import annotations

import fitz
import pytest

from backend.extraction.docling_pdf.docling_extract import _bbox_area_frac


@pytest.fixture
def real_pdf(tmp_path):
    """A real, on-disk single-page PDF, 612x792pt (US Letter)."""
    path = str(tmp_path / "doc.pdf")
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()
    return path


def test_full_page_bbox_returns_area_frac_near_one(real_pdf):
    frac = _bbox_area_frac([0, 0, 612, 792], real_pdf, 1)
    assert frac == pytest.approx(1.0, abs=0.01)


def test_small_bbox_returns_small_area_frac(real_pdf):
    # A 30x30pt box on a 612x792pt page: (30*30)/(612*792) ~= 0.00186 (~0.2%)
    frac = _bbox_area_frac([0, 0, 30, 30], real_pdf, 1)
    assert frac < 0.01
    assert frac == pytest.approx((30 * 30) / (612 * 792), rel=0.01)


def test_half_page_bbox_returns_roughly_half(real_pdf):
    frac = _bbox_area_frac([0, 0, 612, 396], real_pdf, 1)
    assert frac == pytest.approx(0.5, abs=0.02)


def test_out_of_range_page_fails_open_to_one(real_pdf):
    # page_no doesn't exist on a 1-page doc -- must not raise, must fail OPEN
    # (return 1.0 = "caption eagerly") rather than silently defer real content.
    frac = _bbox_area_frac([0, 0, 30, 30], real_pdf, 99)
    assert frac == 1.0


def test_unreadable_pdf_path_fails_open_to_one():
    frac = _bbox_area_frac([0, 0, 30, 30], "/nonexistent/path/does_not_exist.pdf", 1)
    assert frac == 1.0


def test_malformed_bbox_fails_open_to_one(real_pdf):
    frac = _bbox_area_frac([0, 0], real_pdf, 1)  # too few coords
    assert frac == 1.0
