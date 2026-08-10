"""Tests for backend.categorize.id_graph -- deterministic drawing/CAD-sheet ID
extraction and cross-document linking. Patterns validated against REAL text
pulled from the actual Toyoda/JTEKT files during the 3-Aug session (title blocks,
CAD sheet filenames), not made up -- see each test's docstring for the source.

Run: pytest tests/test_id_graph.py
"""
from unittest.mock import MagicMock, patch

from backend.categorize.id_graph import (
    document_id_summary,
    extract_id_from_filename,
    extract_ids,
    find_documents_by_id,
    tag_blocks_with_filename_id,
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
# parts_drawing_no -- 3rd real ID format, ADDED 10-Aug. All 27 sample values
# pulled directly from a real parts-list Excel's "Drawing No" column
# (.../1.Expendable parts drawing/20230831_99Y_NEW-TIGG303_E.xlsx).
# ---------------------------------------------------------------------------

def test_parts_drawing_no_matches_real_sampled_values():
    real_values = [
        "KB-AE000213-A", "KE-MC000954-G", "KE-MC000957-A", "KE-MC000D93-A",
        "KE-MC000D94-A", "KG-DE100891-A", "KK-FA000R19-A", "KK-FA000T95-B",
        "KL-BC001822-A", "KL-BC001J25-A", "KN-CC000T19-A", "KN-CC000T20-A",
        "PA-GA006010-C", "T2-11LG0005-A", "TA-21NG0001-D", "TA-2N1G0001-C",
        "TA-4M3G0007-C", "TA-61BG0002-B", "TA-61BG0004-D", "TA-61BG0005-C",
        "TA-61BG0013-A", "TA-9W5G0001-D", "TA-B8WG0001-B",
    ]
    for v in real_values:
        ids = extract_ids(f"Drawing No: {v}")
        assert ids.get("parts_drawing_no") == [v], f"missed {v}"


def test_parts_drawing_no_does_not_double_match_the_digits_only_family():
    # Same real Excel column also has plain-digits values ("38-55670039-4") --
    # already covered by drawing_number (trailing digit); parts_drawing_no
    # (trailing LETTER) must not also claim them.
    ids = extract_ids("Drawing No: 38-55670039-4")
    assert ids.get("drawing_number") == ["38-55670039-4"]
    assert "parts_drawing_no" not in ids


def test_parts_drawing_no_no_false_positive_on_real_manual_prose():
    text = ("Place the [Operation mode] selection switch to the position of "
            "[MANU.]. Turn the EMERGENCY-STOP button clockwise to unlock it.")
    assert extract_ids(text) == {}


# ---------------------------------------------------------------------------
# extract_id_from_filename / tag_blocks_with_filename_id -- ADDED 10-Aug
# ---------------------------------------------------------------------------

def test_extract_id_from_filename_real_cad_pdf_name():
    # Real filename from the corpus: the CAD PDF is literally named after its
    # own drawing number.
    assert extract_id_from_filename("20230831_99Y_KE-MC000954-G.pdf") == "KE-MC000954-G"


def test_extract_id_from_filename_no_match_returns_none():
    assert extract_id_from_filename("20230831_99Y_02_G0892V10_Operation manual.pdf") is None


def test_extract_id_from_filename_empty_returns_none():
    assert extract_id_from_filename("") is None
    assert extract_id_from_filename(None) is None


def test_tag_blocks_with_filename_id_merges_not_overwrites():
    blocks = [_block("sheet MS03AAA789AB")]
    tag_blocks_with_ids(blocks)  # text-based match first, same as real pipeline order
    tag_blocks_with_filename_id(blocks, "20230831_99Y_KE-MC000954-G.pdf")
    flat = blocks[0]["metadata"]["mentioned_ids_flat"]
    assert "MS03AAA789AB" in flat
    assert "KE-MC000954-G" in flat
    assert blocks[0]["metadata"]["mentioned_ids"]["filename"] == ["KE-MC000954-G"]


def test_tag_blocks_with_filename_id_noop_when_filename_has_no_id():
    blocks = [_block("nothing technical")]
    tag_blocks_with_filename_id(blocks, "manual.pdf")
    assert "mentioned_ids_flat" not in blocks[0]["metadata"]


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
