"""Tests for backend.categorize.id_graph -- deterministic drawing/CAD-sheet ID
extraction and cross-document linking. Patterns validated against REAL text
pulled from the actual Toyoda/JTEKT files during the 3-Aug session (title blocks,
CAD sheet filenames), not made up -- see each test's docstring for the source.

Run: pytest tests/test_id_graph.py
"""
from unittest.mock import MagicMock, patch

from backend.categorize.id_graph import (
    document_id_summary,
    extract_ids,
    find_documents_by_id,
    tag_blocks_with_ids,
)


def test_drawing_number_matches_real_jtekt_format():
    # from the circuit diagram's own title block, page 1
    text = "DRAWING No.:00-83547001-0\nOUTPUT: 21/03/02"
    ids = extract_ids(text)
    assert ids["drawing_number"] == ["00-83547001-0"]


def test_cad_sheet_id_matches_real_filename_and_title_block():
    # from MS03AAA789AB-spindle assembly.pdf's own title block
    text = "SPINDLE ASSEMBLY\nMS03AAA789AB\nM----BS17"
    ids = extract_ids(text)
    assert ids["cad_sheet_id"] == ["MS03AAA789AB"]


def test_multiple_distinct_ids_in_one_string_deduped_in_order():
    text = "See MS03AAA789AB and also MS03AAA981AA. Ref MS03AAA789AB again."
    ids = extract_ids(text)
    assert ids["cad_sheet_id"] == ["MS03AAA789AB", "MS03AAA981AA"]


def test_no_match_returns_empty_dict_not_empty_lists():
    assert extract_ids("just some ordinary sentence, nothing technical here") == {}


def test_empty_string_returns_empty_dict():
    assert extract_ids("") == {}
    assert extract_ids(None) == {}


def test_short_alarm_style_codes_do_not_false_positive_as_drawing_numbers():
    # "1-1" (Q1's real alarm code) and "B920" (Q2's) must NOT match
    # drawing_number (\d{2}-\d{7,9}-\d) -- too short, wrong shape.
    ids = extract_ids("alarm code 1-1 occurred; see also B920 on the panel")
    assert "drawing_number" not in ids
    assert "cad_sheet_id" not in ids


def test_connector_part_numbers_do_not_false_positive_as_drawing_numbers():
    # real connector part numbers from page 60/61 of the circuit diagram --
    # different shape (2-1747822-2 has only 7 digits in the middle group,
    # drawing_number requires 7-9) so this is a genuine edge; confirm no match.
    ids = extract_ids("CONNECTOR IN BOX: 2-1747822-2 (REC HOUSING) 316040-2 (REC CONTACT)")
    assert "drawing_number" not in ids


# ---------------------------------------------------------------------------
# tag_blocks_with_ids / document_id_summary
# ---------------------------------------------------------------------------

def _block(text, block_id="b1"):
    return {"block_id": block_id, "type": "text", "text": text, "metadata": {}}


def test_tag_blocks_adds_nested_and_flat_metadata():
    blocks = [_block("Ref drawing 00-83547001-0 and sheet MS03AAA789AB")]
    tag_blocks_with_ids(blocks)
    meta = blocks[0]["metadata"]
    assert meta["mentioned_ids"] == {
        "drawing_number": ["00-83547001-0"],
        "cad_sheet_id": ["MS03AAA789AB"],
    }
    assert sorted(meta["mentioned_ids_flat"]) == sorted(["00-83547001-0", "MS03AAA789AB"])


def test_tag_blocks_leaves_no_match_blocks_untouched():
    blocks = [_block("nothing technical here")]
    tag_blocks_with_ids(blocks)
    assert "mentioned_ids" not in blocks[0]["metadata"]


def test_tag_blocks_skips_non_dict_entries_without_crashing():
    blocks = [_block("MS03AAA789AB"), None]
    tag_blocks_with_ids(blocks)   # must not raise
    assert blocks[0]["metadata"]["mentioned_ids"]


def test_document_id_summary_merges_across_blocks_deduped():
    blocks = [
        _block("sheet MS03AAA789AB", "b1"),
        _block("also MS03AAA789AB and 00-83547001-0", "b2"),
        _block("no ids here", "b3"),
    ]
    tag_blocks_with_ids(blocks)
    summary = document_id_summary(blocks)
    assert summary["cad_sheet_id"] == ["MS03AAA789AB"]
    assert summary["drawing_number"] == ["00-83547001-0"]


# ---------------------------------------------------------------------------
# find_documents_by_id (DB-mocked)
# ---------------------------------------------------------------------------

def test_find_documents_by_id_queries_flat_array_containment():
    mock_store = MagicMock()
    mock_store.conn.execute.return_value.fetchall.return_value = [
        ("doc-a", "blk-1", "text", "sheet MS03AAA789AB", {"page": 3}),
    ]
    with patch("backend.storage.postgres_store.PostgresStore", return_value=mock_store):
        rows = find_documents_by_id("MS03AAA789AB")

    assert rows == [{"document_id": "doc-a", "block_id": "blk-1", "type": "text",
                     "text": "sheet MS03AAA789AB", "source_ref": {"page": 3}}]
    sql, params = mock_store.conn.execute.call_args.args
    assert "mentioned_ids_flat" in sql
    assert params[0] == '["MS03AAA789AB"]'
    mock_store.close.assert_called_once()


def test_find_documents_by_id_scopes_to_one_document_when_given():
    mock_store = MagicMock()
    mock_store.conn.execute.return_value.fetchall.return_value = []
    with patch("backend.storage.postgres_store.PostgresStore", return_value=mock_store):
        find_documents_by_id("MS03AAA789AB", document_id="doc-a")

    sql, params = mock_store.conn.execute.call_args.args
    assert "document_id::text = %s" in sql
    assert params[1] == "doc-a"
