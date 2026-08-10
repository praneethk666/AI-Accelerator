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
    _row_traceability_error,
    _row_is_continuation,
    _stem,
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


def test_blank_continuation_rows_trigger_repair():
    # Real shape produced by table_source=auto's pymupdf ruled-line extraction on a
    # merged/rowspan source cell (validated live, 24-Jul, servo manual alarm table):
    # every row has the SAME column count (so _has_ragged_rows misses it entirely),
    # but rows after the first leave the identifying leading columns blank.
    block = _table_block(
        ["Alarm name", "Code", "Details", "Remarks"],
        [
            ["Motor model setting error", "F7H", "Setting inconsistency detected.", "SU"],
            ["", "", "Detect error when motor code being set is wrong.", ""],
            ["", "", "Clear by re-powering on.", ""],
        ],
    )
    result = _classify_table_for_repair(block, "5.3 Alarm List", "")
    assert result["needs_llm"] is True
    assert "blank_continuation_rows" in result["reasons"]


def test_table_with_a_genuinely_blank_first_column_header_is_not_flagged():
    # A real, well-formed table can legitimately have a blank leading label column
    # (e.g. a row-header-less matrix) as long as MOST rows aren't blank-leading —
    # only a systematic pattern (>=15% of rows) should trigger repair.
    block = _table_block(
        ["", "Factor 1"],
        [["Occurred on power-on.", "○"], ["Occurred on servo-on.", "○"],
         ["Occurred after motor start.", "○"]],
    )
    result = _classify_table_for_repair(block, "", "")
    assert "blank_continuation_rows" not in result["reasons"]


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


def test_repaired_chunks_carry_structural_tags():
    # ADDED 10-Aug: mentioned_ids/folder tags (id_graph.py/folder_router.py) must
    # reach repaired-table chunks too, not just the deterministic extractors --
    # this is the one bypass path that needs the LLM call mocked to test.
    block = {
        "block_id": "b1", "document_id": "d1",
        "table_data": {"headers": _HEADERS, "rows": _ROWS},
        "source_ref": {"filename": "manual.pdf", "page": 61},
        "metadata": {
            "mentioned_ids_flat": ["MS03AAA789AB"],
            "folder": {"machine": "120_CYLINDRICAL GRINDER", "component": "Spindlehead"},
        },
    }
    with patch("backend.chunking.chunk_tool.get_llm_for", return_value=_mock_llm(_good_parsed())):
        chunks = _repair_table_with_llm(block, {"chunking": {}}, "Alarm code 11H", "prec ctx", lambda r: r)
    for c in chunks:
        assert c["tags"]["mentioned_ids"] == ["MS03AAA789AB"]
        assert c["tags"]["machine"] == "120_CYLINDRICAL GRINDER"
        assert c["tags"]["component"] == "Spindlehead"


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


# ── word-form + header + continuation-row anchor fixes (found live, 27-Jul, on the
# servo manual's real F7H-FFH alarm table -- see docling_extract.py/_table_has_span
# investigation: this table's blank-continuation-row repair kept failing traceability
# even when the LLM correctly followed the "denormalize into each row" instruction) ──

def test_stem_strips_common_suffixes():
    assert _stem("cleared") == "clear"
    assert _stem("powering") == "power"
    assert _stem("occurs") == "occur"


def test_stem_does_not_over_strip_short_words():
    # Stripping "s" from "as" would leave "a" -- guarded by the min-remainder-length check.
    assert _stem("as") == "as"


def test_traceability_allows_verb_conjugation_of_a_cells_own_word():
    # Real failure: cell says "Clear by re-powering on.", natural prose says "cleared
    # by re-powering on" -- exact word-form matching alone flagged "cleared" as
    # fabricated even though it's just a conjugation of the cell's own word. Kept to
    # only the cell's own vocabulary (plus stopwords) so this isolates JUST the verb-
    # conjugation behavior, not the separate proportional-slack allowance.
    err = _row_traceability_error(
        "It is cleared by re-powering on.",
        ["", "", "Clear by re-powering on.", "", ""],
        "",
    )
    assert err is None


def test_traceability_allows_referencing_a_column_header_via_context():
    # Real failure: chunk_text said "...with code 'F7H'..." and "code" is never a
    # literal CELL value anywhere -- it's the column's HEADER name, which
    # _validate_repaired_rows now folds into the context string passed here.
    err = _row_traceability_error(
        "The alarm has code F7H.",
        ["Motor model setting error", "F7H"],
        "code",  # headers get folded into context by the caller, not this function
    )
    assert err is None


def test_traceability_still_rejects_a_genuinely_invented_word():
    err = _row_traceability_error(
        "Factor 1 costs $2,499.00 and ships in a Laptop Computer box.",
        ["Factor 1", "Servo unit failure", "Replace servo unit."],
        "",
    )
    assert err is not None


_ALARM_HEADERS = ["Alarm name", "Code", "Details"]
_ALARM_ROWS = [
    ["Motor model setting error", "F7H", "Setting inconsistency detected."],
    ["", "", "Detect error when motor code being set is wrong."],
    ["", "", "Clear by re-powering on."],
]


def test_row_is_continuation_true_for_blank_leading_cells():
    assert _row_is_continuation(["", "", "Clear by re-powering on."]) is True


def test_row_is_continuation_false_for_a_new_header_row():
    assert _row_is_continuation(["Parameter error 1", "F8H", "..."]) is False


def test_continuation_row_may_repeat_its_anchor_rows_identifying_value():
    # The repair prompt explicitly instructs denormalizing a continuation row by
    # repeating the anchor (header) row's identifying value into its own chunk_text
    # -- this is the whole point of the repair, so it must not be flagged as
    # fabrication just because "Motor model setting error"/"F7H" aren't in THIS
    # row's own (blank) cells.
    parsed = [
        {"row_index": 0, "chunk_text": "Alarm 'Motor model setting error' (F7H): Setting inconsistency detected.",
         "structured": {}},
        {"row_index": 1, "chunk_text": "Alarm 'Motor model setting error' (F7H): Detect error when motor code being set is wrong.",
         "structured": {}},
        {"row_index": 2, "chunk_text": "Alarm 'Motor model setting error' (F7H): Clear by re-powering on.",
         "structured": {}},
    ]
    ordered, err = _validate_repaired_rows(parsed, _ALARM_HEADERS, _ALARM_ROWS, "sec", "prec")
    assert err == ""
    assert len(ordered) == 3


def test_continuation_row_still_rejected_if_it_borrows_a_different_alarms_identity():
    # Two distinct alarms back to back; row 1 (a continuation of alarm 0) must not
    # get away with referencing alarm 2's name/code just because it appears
    # somewhere else in the table -- only the IMMEDIATE anchor's cells are allowed.
    rows = [
        ["Motor model setting error", "F7H", "Setting inconsistency detected."],
        ["", "", "Clear by re-powering on."],
        ["Parameter error 1", "F8H", "System parameter error detected."],
    ]
    parsed = [
        {"row_index": 0, "chunk_text": "Alarm 'Motor model setting error' (F7H): Setting inconsistency detected.",
         "structured": {}},
        {"row_index": 1, "chunk_text": "Alarm 'Parameter error 1' (F8H): Clear by re-powering on.",
         "structured": {}},
        {"row_index": 2, "chunk_text": "Alarm 'Parameter error 1' (F8H): System parameter error detected.",
         "structured": {}},
    ]
    ordered, err = _validate_repaired_rows(parsed, _ALARM_HEADERS, rows, "sec", "prec")
    assert ordered is None
    assert "row 1" in err


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
