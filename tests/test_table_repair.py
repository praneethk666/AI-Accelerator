"""Hard-table LLM repair tests (no infra — the LLM call is mocked).
Run: pytest tests/test_table_repair.py

Covers the gate (_classify_table_for_repair), the header-collision fix
(_disambiguate_headers), the anchor-based row validator (_validate_repaired_rows —
this is the piece that replaces plain word-traceability checking, which validated
live on 21-Jul let a misattributed row pass because every word it used was real,
just from the wrong row), and the end-to-end repair path with tags.summary wired so
answerer._is_thin() won't blow the repair away.
"""
import json
from unittest.mock import MagicMock, patch

from backend.chunking.chunk_tool import (
    _classify_table_for_repair,
    _disambiguate_headers,
    _validate_repaired_rows,
    _repair_table_with_llm,
    chunk_blocks,
)


# ── _disambiguate_headers ───────────────────────────────────────────────────────

def test_disambiguate_headers_leaves_unique_headers_untouched():
    assert _disambiguate_headers(["Cause", "Action"]) == ["Cause", "Action"]


def test_disambiguate_headers_suffixes_true_duplicates():
    out = _disambiguate_headers(["", "Factor 1", "Factor 1", "Factor 2"])
    assert out == ["column", "Factor 1 (1)", "Factor 1 (2)", "Factor 2"]


# ── _classify_table_for_repair ──────────────────────────────────────────────────

def _table_block(headers, rows):
    return {"block_id": "b1", "table_data": {"headers": headers, "rows": rows}}


def test_well_formed_table_with_no_code_like_context_does_not_need_repair():
    # No alarm-code-shaped token anywhere in the surrounding context, so there's
    # nothing for the table's cells to be missing -> _lacks_context_anchor is moot.
    block = _table_block(["Factor", "Cause", "Action"],
                          [["Factor 1", "Servo unit failure", "Replace servo unit."]] * 3)
    result = _classify_table_for_repair(block, "Corrective actions", "See the section below")
    assert result["needs_llm"] is False


def test_well_formed_table_still_needs_repair_if_context_anchor_missing_from_cells():
    # Same clean structure as above, but the section names a specific alarm code
    # ('11H') that the table's own cells never repeat — this is the real case
    # validated live where a structurally fine table still loses its context.
    block = _table_block(["Factor", "Cause", "Action"],
                          [["Factor 1", "Servo unit failure", "Replace servo unit."]] * 3)
    result = _classify_table_for_repair(block, "Alarm code 11H", "Corrective action")
    assert result["needs_llm"] is True
    assert result["reasons"] == ["missing_context_anchor"]


def test_duplicate_headers_trigger_repair():
    block = _table_block(["", "Factor 1", "Factor 1"], [["a", "1", "2"]] * 3)
    result = _classify_table_for_repair(block, "", "")
    assert result["needs_llm"] is True
    assert "header_anomaly" in result["reasons"]


def test_ragged_rows_trigger_repair():
    block = _table_block(["a", "b"], [["1", "2"], ["1", "2", "3"]])
    result = _classify_table_for_repair(block, "", "")
    assert result["needs_llm"] is True
    assert "ragged_rows" in result["reasons"]


def test_orphan_table_triggers_repair():
    block = _table_block(["a", "b"], [["1", "2"]])
    result = _classify_table_for_repair(block, "", "")
    assert result["needs_llm"] is True
    assert "orphan_table" in result["reasons"]


def test_missing_context_anchor_triggers_repair():
    # section names alarm code 11H, but the table's own cells never mention it
    block = _table_block(["Factor", "Cause", "Action"],
                          [["Factor 1", "Servo unit failure", "Replace servo unit."],
                           ["Factor 2", "Short-circuit", "Check cables."],
                           ["Factor 3", "Motor failure", "Replace motor."]])
    result = _classify_table_for_repair(block, "Power device failure Alarm code 11H", "")
    assert result["needs_llm"] is True
    assert "missing_context_anchor" in result["reasons"]


def test_context_anchor_present_in_cells_does_not_trigger():
    block = _table_block(["Code", "Cause"], [["11H", "Motor failure"]] * 3)
    result = _classify_table_for_repair(block, "Alarm code 11H", "")
    assert result["needs_llm"] is False


# ── _validate_repaired_rows ──────────────────────────────────────────────────────

_HEADERS = ["Factor", "Cause", "Action"]
_ROWS = [
    ["Factor 1", "Servo unit failure", "Replace servo unit."],
    ["Factor 2", "Short-circuit in motor power wiring", "Check cables."],
    ["Factor 3", "Motor failure", "Replace motor."],
]


def _good_parsed():
    return [
        {"row_index": 0, "chunk_text": "Factor 1: Servo unit failure. Action: Replace servo unit.",
         "structured": {"Factor": "Factor 1", "Cause": "Servo unit failure", "Action": "Replace servo unit."}},
        {"row_index": 1, "chunk_text": "Factor 2: Short-circuit in motor power wiring. Action: Check cables.",
         "structured": {"Factor": "Factor 2", "Cause": "Short-circuit in motor power wiring", "Action": "Check cables."}},
        {"row_index": 2, "chunk_text": "Factor 3: Motor failure. Action: Replace motor.",
         "structured": {"Factor": "Factor 3", "Cause": "Motor failure", "Action": "Replace motor."}},
    ]


def test_valid_repair_passes_and_preserves_order():
    ordered, err = _validate_repaired_rows(_good_parsed(), _HEADERS, _ROWS, "sec", "prec")
    assert err == ""
    assert [o["row_index"] for o in ordered] == [0, 1, 2]


def test_out_of_order_response_still_validates_and_gets_reordered():
    shuffled = [_good_parsed()[2], _good_parsed()[0], _good_parsed()[1]]
    ordered, err = _validate_repaired_rows(shuffled, _HEADERS, _ROWS, "sec", "prec")
    assert err == ""
    assert [o["row_index"] for o in ordered] == [0, 1, 2]


def test_row_count_mismatch_fails():
    parsed = _good_parsed()[:2]
    ordered, err = _validate_repaired_rows(parsed, _HEADERS, _ROWS, "sec", "prec")
    assert ordered is None
    assert "row count mismatch" in err


def test_duplicate_row_index_fails():
    parsed = _good_parsed()
    parsed[1]["row_index"] = 0  # duplicate of parsed[0]
    ordered, err = _validate_repaired_rows(parsed, _HEADERS, _ROWS, "sec", "prec")
    assert ordered is None
    assert "invalid or duplicate row_index" in err


def test_out_of_range_row_index_fails():
    parsed = _good_parsed()
    parsed[0]["row_index"] = 99
    ordered, err = _validate_repaired_rows(parsed, _HEADERS, _ROWS, "sec", "prec")
    assert ordered is None


def test_misattributed_row_content_fails_traceability():
    # row_index 0 (Factor 1 / Servo unit failure) but the text actually describes
    # row 2's content (Motor failure / Replace motor) — exactly the real failure
    # mode found live: right words, wrong row. A whole-table word-traceability
    # check would have let this through since every word IS in the table somewhere.
    parsed = _good_parsed()
    parsed[0]["chunk_text"] = "Factor 1: Motor failure. Action: Replace motor."
    ordered, err = _validate_repaired_rows(parsed, _HEADERS, _ROWS, "sec", "prec")
    assert ordered is None
    assert "row 0" in err


def test_hallucinated_words_fail_traceability():
    parsed = _good_parsed()
    parsed[0]["chunk_text"] = "Factor 1 costs $2,499.00 and ships in a Laptop Computer box."
    ordered, err = _validate_repaired_rows(parsed, _HEADERS, _ROWS, "sec", "prec")
    assert ordered is None


# ── _repair_table_with_llm (end-to-end, LLM mocked) ──────────────────────────────

def _mock_llm(json_payload):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(json_payload))
    return llm


def test_successful_repair_sets_summary_tag_and_chunk_type():
    block = {
        "block_id": "b1", "document_id": "d1",
        "table_data": {"headers": _HEADERS, "rows": _ROWS},
        "source_ref": {"filename": "manual.pdf", "page": 61},
    }
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=_mock_llm(_good_parsed())):
        chunks = _repair_table_with_llm(block, {"chunking": {}}, "Alarm code 11H", "prec ctx", lambda r: r)
    assert len(chunks) == 3
    for c in chunks:
        assert c["tags"]["repaired"] is True
        assert c["tags"]["chunk_type"] == "llm_repaired_table_row"
        # the whole point of this tag: _is_thin() in answerer.py must see a summary
        assert c["tags"]["summary"] == c["text"]
        assert c["tags"]["summary"].strip()


def test_invalid_llm_json_fails_closed_returns_none():
    block = {
        "block_id": "b1", "document_id": "d1",
        "table_data": {"headers": _HEADERS, "rows": _ROWS},
        "source_ref": {"filename": "manual.pdf", "page": 61},
    }
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=_mock_llm({"not": "a list"})):
        chunks = _repair_table_with_llm(block, {"chunking": {}}, "sec", "prec", lambda r: r)
    assert chunks is None


def test_llm_call_exception_fails_closed_returns_none():
    block = {
        "block_id": "b1", "document_id": "d1",
        "table_data": {"headers": _HEADERS, "rows": _ROWS},
        "source_ref": {"filename": "manual.pdf", "page": 61},
    }
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider down")
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=llm):
        chunks = _repair_table_with_llm(block, {"chunking": {}}, "sec", "prec", lambda r: r)
    assert chunks is None


def test_validation_failure_fails_closed_returns_none():
    bad = _good_parsed()
    bad[0]["chunk_text"] = "Factor 1: Motor failure. Action: Replace motor."  # misattributed
    block = {
        "block_id": "b1", "document_id": "d1",
        "table_data": {"headers": _HEADERS, "rows": _ROWS},
        "source_ref": {"filename": "manual.pdf", "page": 61},
    }
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=_mock_llm(bad)):
        chunks = _repair_table_with_llm(block, {"chunking": {}}, "sec", "prec", lambda r: r)
    assert chunks is None


# ── chunk_blocks() integration ──────────────────────────────────────────────────

def test_chunk_blocks_repairs_hard_table_when_enabled():
    blocks = [
        {"type": "heading", "text": "Alarm code 11H", "document_id": "d1"},
        {"type": "table", "document_id": "d1",
         "table_data": {"headers": _HEADERS, "rows": _ROWS},
         "source_ref": {"filename": "manual.pdf", "page": 61}},
    ]
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=_mock_llm(_good_parsed())):
        chunks = chunk_blocks(blocks, repair_hard_tables=True, config={"chunking": {}})
    repaired = [c for c in chunks if c.get("tags", {}).get("repaired")]
    assert len(repaired) == 3
    assert all(c["tags"]["summary"] for c in repaired)


def test_chunk_blocks_repair_disabled_by_default():
    blocks = [
        {"type": "heading", "text": "Alarm code 11H", "document_id": "d1"},
        {"type": "table", "document_id": "d1",
         "table_data": {"headers": _HEADERS, "rows": _ROWS},
         "source_ref": {"filename": "manual.pdf", "page": 61}},
    ]
    # repair_hard_tables defaults to False -> get_llm_for must never be called
    with patch("backend.chunking.chunk_tool.get_llm_for") as mock_get_llm:
        chunk_blocks(blocks, config={"chunking": {}})
        mock_get_llm.assert_not_called()


def test_chunk_blocks_falls_back_to_deterministic_when_repair_fails():
    blocks = [
        {"type": "heading", "text": "Alarm code 11H", "document_id": "d1"},
        {"type": "table", "document_id": "d1",
         "table_data": {"headers": ["Factor", "Cause", "Action"], "rows": _ROWS},
         "source_ref": {"filename": "manual.pdf", "page": 61}},
    ]
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider down")
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=llm):
        chunks = chunk_blocks(blocks, repair_hard_tables=True, config={"chunking": {}})
    # repair failed closed -> falls through to the deterministic troubleshooting_row
    # extractor (Cause/Action headers) rather than dropping the table
    assert len(chunks) == 3
    assert all(c.get("tags", {}).get("chunk_type") == "troubleshooting_row" for c in chunks)
