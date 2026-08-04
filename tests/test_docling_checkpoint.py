"""Tests for _checkpoint_page_blocks() -- real gap found 3-Aug: extract_docling had
ZERO persistence of its own during the per-page conversion loop (only a status-string
progress label via _update_page_progress, not the actual blocks), so a crash/kill
partway through a large document (e.g. page 900 of 1147) lost ALL progress, not just
the tail. This checkpoints each page's text/table blocks to Postgres as they're
produced, mirroring the pattern route_and_rescue already uses successfully.

Run: pytest tests/test_docling_checkpoint.py
"""
from unittest.mock import MagicMock, patch

from backend.extraction.docling_pdf.docling_extract import _checkpoint_page_blocks


def test_writes_page_blocks_via_postgres_store():
    mock_store = MagicMock()
    with patch("backend.storage.postgres_store.PostgresStore", return_value=mock_store):
        _checkpoint_page_blocks("doc1", 42, [{"block_id": "b1", "type": "text"}])

    mock_store.write_page_blocks.assert_called_once_with(
        "doc1", 42, [{"block_id": "b1", "type": "text"}])
    mock_store.close.assert_called_once()


def test_noop_when_no_blocks():
    with patch("backend.storage.postgres_store.PostgresStore") as mock_cls:
        _checkpoint_page_blocks("doc1", 1, [])
    mock_cls.assert_not_called()


def test_store_closed_even_when_write_raises():
    mock_store = MagicMock()
    mock_store.write_page_blocks.side_effect = RuntimeError("db down")
    with patch("backend.storage.postgres_store.PostgresStore", return_value=mock_store):
        _checkpoint_page_blocks("doc1", 1, [{"block_id": "b1"}])  # must not raise
    mock_store.close.assert_called_once()


def test_connection_failure_swallowed_not_raised():
    with patch("backend.storage.postgres_store.PostgresStore",
              side_effect=RuntimeError("connection refused")):
        _checkpoint_page_blocks("doc1", 1, [{"block_id": "b1"}])  # must not raise
