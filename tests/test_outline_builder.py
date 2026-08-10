"""Tests for backend.pipeline.outline_builder -- builds a navigable chapter/
section tree from either real PDF bookmarks (fitz.get_toc()) or a heading stack
derived during extraction, so the agent can locate a section by structure instead
of trusting semantic search among several similarly-worded procedures.

Run: pytest tests/test_outline_builder.py
"""
from unittest.mock import MagicMock, patch

import fitz
import pytest

from backend.pipeline.outline_builder import (
    _fill_page_ends,
    _nodes_from_flat_list,
    _outline_from_bookmarks,
    _outline_from_headings,
    build_outline,
    get_outline_children,
    write_document_outline,
)


# ── _fill_page_ends / _nodes_from_flat_list ───────────────────────────────────

def test_nodes_from_flat_list_builds_parent_child_edges():
    entries = [(1, "Chapter 1", 1), (2, "1.1 Section", 2), (2, "1.2 Section", 4), (1, "Chapter 2", 6)]
    nodes = _nodes_from_flat_list(entries, "heading_detect")
    by_title = {n["title"]: n for n in nodes}
    assert by_title["Chapter 1"]["parent_id"] is None
    assert by_title["1.1 Section"]["parent_id"] == by_title["Chapter 1"]["node_id"]
    assert by_title["1.2 Section"]["parent_id"] == by_title["Chapter 1"]["node_id"]
    assert by_title["Chapter 2"]["parent_id"] is None


def test_fill_page_ends_stops_at_next_sibling_or_higher():
    entries = [(1, "Chapter 1", 1), (2, "1.1", 2), (2, "1.2", 4), (1, "Chapter 2", 6)]
    nodes = _nodes_from_flat_list(entries, "heading_detect")
    _fill_page_ends(nodes, total_pages=10)
    by_title = {n["title"]: n for n in nodes}
    assert by_title["1.1"]["page_end"] == 3        # stops before 1.2 (same level)
    assert by_title["1.2"]["page_end"] == 5         # stops before Chapter 2 (higher level)
    assert by_title["Chapter 1"]["page_end"] == 5    # stops before Chapter 2 (same level)
    assert by_title["Chapter 2"]["page_end"] == 10   # last node -> total_pages


def test_fill_page_ends_last_node_never_below_its_own_start():
    nodes = _nodes_from_flat_list([(1, "Only", 5)], "heading_detect")
    _fill_page_ends(nodes, total_pages=3)  # pathological: total_pages < page_start
    assert nodes[0]["page_end"] == 5


# ── _outline_from_headings ────────────────────────────────────────────────────

def _heading(text, page, bbox=None):
    return {"type": "heading", "text": text, "source_ref": {"page": page, "bbox": bbox}}


def test_outline_from_headings_real_shape():
    # Mirrors the real changeover manual's structure: "1." / "1.1" / "1.2"
    blocks = [
        _heading("1. CHANGING THE SETUP OF WORKPIECE HOLDER AND PHASE INDEXING", 5),
        {"type": "text", "text": "intro", "source_ref": {"page": 5}},
        _heading("1.1 Replacing the Workpiece Holder", 5),
        {"type": "text", "text": "steps...", "source_ref": {"page": 5}},
        _heading("1.2 Replacing the Phase Datum Pad", 6),
        {"type": "text", "text": "steps...", "source_ref": {"page": 6}},
        _heading("1.3 Replacing the Pad of Phase Indexing Unit", 7),
    ]
    nodes = _outline_from_headings(blocks)
    assert nodes is not None
    titles = {n["title"]: n for n in nodes}
    assert titles["1.1 Replacing the Workpiece Holder"]["parent_id"] == \
           titles["1. CHANGING THE SETUP OF WORKPIECE HOLDER AND PHASE INDEXING"]["node_id"]
    assert titles["1.2 Replacing the Phase Datum Pad"]["page_start"] == 6
    assert all(n["source"] == "heading_detect" for n in nodes)


def test_outline_from_headings_fewer_than_min_returns_none():
    blocks = [_heading("Only One Heading", 1), _heading("Second", 2)]
    assert _outline_from_headings(blocks) is None


def test_outline_from_headings_ignores_non_heading_blocks():
    blocks = [
        _heading("1.", 1), _heading("1.1", 1), _heading("1.2", 2),
        {"type": "text", "text": "not a heading", "source_ref": {"page": 1}},
        {"type": "table", "text": "not a heading either", "source_ref": {"page": 1}},
    ]
    nodes = _outline_from_headings(blocks)
    assert len(nodes) == 3


def test_outline_from_headings_blank_or_pageless_headings_skipped():
    blocks = [
        _heading("1.", 1), _heading("", 2), _heading("1.1", None), _heading("1.2", 2),
        _heading("1.3", 3),
    ]
    nodes = _outline_from_headings(blocks)
    titles = [n["title"] for n in nodes]
    assert titles == ["1.", "1.2", "1.3"]


# ── _outline_from_bookmarks (real fitz PDF) ───────────────────────────────────

@pytest.fixture
def real_pdf_with_bookmarks(tmp_path):
    path = str(tmp_path / "doc.pdf")
    doc = fitz.open()
    for _ in range(10):
        doc.new_page(width=612, height=792)
    doc.set_toc([
        [1, "Chapter 1", 1],
        [2, "1.1 Section", 2],
        [2, "1.2 Section", 4],
        [1, "Chapter 2", 6],
    ])
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def real_pdf_no_bookmarks(tmp_path):
    path = str(tmp_path / "flat.pdf")
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()
    return path


def test_outline_from_bookmarks_real_pdf(real_pdf_with_bookmarks):
    nodes = _outline_from_bookmarks(real_pdf_with_bookmarks)
    assert nodes is not None
    assert len(nodes) == 4
    assert all(n["source"] == "pdf_bookmark" for n in nodes)
    by_title = {n["title"]: n for n in nodes}
    assert by_title["1.1 Section"]["parent_id"] == by_title["Chapter 1"]["node_id"]
    assert by_title["Chapter 1"]["page_end"] == 5


def test_outline_from_bookmarks_no_toc_returns_none(real_pdf_no_bookmarks):
    assert _outline_from_bookmarks(real_pdf_no_bookmarks) is None


def test_outline_from_bookmarks_missing_file_returns_none():
    assert _outline_from_bookmarks("/nonexistent/path/does_not_exist.pdf") is None


def test_outline_from_bookmarks_empty_path_returns_none():
    assert _outline_from_bookmarks("") is None
    assert _outline_from_bookmarks(None) is None


# ── build_outline (bookmark preferred over heading detection) ────────────────

def test_build_outline_prefers_bookmarks_when_present(real_pdf_with_bookmarks):
    blocks = [_heading("Unrelated heading A", 1), _heading("Unrelated heading B", 2),
              _heading("Unrelated heading C", 3)]
    nodes = build_outline(real_pdf_with_bookmarks, blocks)
    assert nodes[0]["source"] == "pdf_bookmark"


def test_build_outline_falls_back_to_headings_when_no_bookmarks(real_pdf_no_bookmarks):
    blocks = [_heading("1.", 1), _heading("1.1", 1), _heading("1.2", 1)]
    nodes = build_outline(real_pdf_no_bookmarks, blocks)
    assert nodes is not None
    assert nodes[0]["source"] == "heading_detect"


def test_build_outline_none_when_neither_source_yields_anything():
    assert build_outline(None, [{"type": "text", "text": "no headings here"}]) is None


# ── write_document_outline / get_outline_children (Postgres mocked) ──────────

def _fake_pg():
    pg = MagicMock()
    pg.conn = MagicMock()
    return pg


def test_write_document_outline_noop_on_none_or_empty():
    with patch("backend.storage.postgres_store.PostgresStore") as MockPg:
        write_document_outline("doc-1", None)
        write_document_outline("doc-1", [])
    MockPg.assert_not_called()


def test_write_document_outline_deletes_stale_then_inserts():
    pg = _fake_pg()
    nodes = [{"node_id": "n0", "parent_id": None, "title": "Ch 1", "level": 1,
              "page_start": 1, "page_end": 5, "source": "heading_detect"}]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=pg):
        write_document_outline("doc-1", nodes)

    calls = [c.args[0] for c in pg.conn.execute.call_args_list]
    assert any("DELETE FROM document_outline" in c for c in calls)
    assert any("INSERT INTO document_outline" in c for c in calls)
    pg.close.assert_called_once()


def test_get_outline_children_root_query_uses_parent_id_is_null():
    pg = _fake_pg()
    pg.conn.execute.return_value.fetchall.return_value = []
    with patch("backend.storage.postgres_store.PostgresStore", return_value=pg):
        get_outline_children("doc-1", node_id=None)

    sql, params = pg.conn.execute.call_args_list[-1].args
    assert "parent_id IS NULL" in sql
    assert params == ["doc-1"]


def test_get_outline_children_with_node_id():
    pg = _fake_pg()
    pg.conn.execute.return_value.fetchall.return_value = [
        ("n1", "1.1 Section", 2, 2, 3),
    ]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=pg):
        result = get_outline_children("doc-1", node_id="n0")

    sql, params = pg.conn.execute.call_args_list[-1].args
    assert "parent_id = %s" in sql
    assert params == ["doc-1", "n0"]
    assert result == [{"node_id": "n1", "title": "1.1 Section", "level": 2,
                       "page_start": 2, "page_end": 3}]
