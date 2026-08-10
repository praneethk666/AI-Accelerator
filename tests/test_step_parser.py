"""Tests for backend.agent.step_parser -- deterministic "(N) ..." step-list
parsing for guided procedure walkthroughs. Real-transcript test uses the exact
text captured directly from the real manual this feature was designed against
(.../3.INSTRUCTION MANUAL/EN/20230831_99Y_05_G0797V10_Changeover of work
holder&phase Indexing.pdf, page 5-7), not invented text.

Run: pytest tests/test_step_parser.py
"""
from backend.agent.step_parser import parse_procedure_from_blocks


def _heading(text, page, level_hint_bbox=None):
    return {"type": "heading", "text": text, "source_ref": {"page": page, "bbox": level_hint_bbox}}


def _text(text, page):
    return {"type": "text", "text": text, "source_ref": {"page": page}}


# ── Real transcript: the changeover manual's own steps 1-13, verbatim ────────

_REAL_MANUAL_BLOCKS = [
    _heading("1. CHANGING THE SETUP OF WORKPIECE HOLDER AND PHASE INDEXING", 5),
    _heading("1.1 Replacing the Workpiece Holder", 5),
    _text(
        "(1) Place the [Operation mode] selection switch to the position of [MANU.].\n"
        "(2) Place the [CNC MODE] selection switch in the [MEMORY] position.\n"
        "(3) Press the [GR.WHEEL SPINDLE STOP] button and stop the wheel rotation.\n"
        "(4) Open the front door.\n"
        "(5) Press the [EMERGENCY STOP] button.\n"
        "(6) Loosen the bolts and remove the workpiece \nholder.\n"
        "(7) Attach the workpiece holder for the workpiece to \nbe machined next, and tighten the bolts to fix \ndown. \n"
        "(8) Turn the [EMERGENCY STOP] button clockwise \nto unlock it.\n"
        "(9) Press the [MASTER ON] button.\n"
        "(10) Load the workpiece on the workpiece holder.\n"
        "(11) Advance both centers.\n"
        "(12) Press the [EMERGENCY STOP] button.\n"
        "(13) Make sure that there is a clearance between the \nworkpiece and the workpiece holder, using such \nas a thickness gauge.\n",
        5,
    ),
    _heading("1.2 Replacing the Phase Datum Pad", 6),
    _text(
        "(1) Turn the [EMERGENCY STOP] button clockwise to unlock it.\n"
        "(2) Press the [MASTER ON] button.\n"
        "(3) Place the [Operation mode] selection switch to the position of [MANU.].\n"
        "(4) Place the [CNC MODE] selection switch in the [MEMORY] position.\n"
        "(5) Press the [GR.WHEEL SPINDLE STOP] button and stop the wheel rotation.\n"
        "(6) Retract the phase datum.\n"
        "(7) Open the front door.\n"
        "(8) Press the [EMERGENCY STOP] button.\n"
        "(9) Loosen the bolts and remove the pad. \n"
        "(10) Attach the appropriate pad for the workpiece to \nbe machined next and tighten the bolts to \nsecure.\n",
        6,
    ),
    _heading("1.3 Replacing the Pad of Phase Indexing Unit", 7),
    _text("(1) Turn the [EMERGENCY STOP] button clockwise to unlock it.\n(2) Press the [MASTER ON] button.\n", 7),
]


def test_real_manual_section_1_1_parses_all_13_steps_in_order():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Workpiece Holder")
    assert result is not None
    assert result["section_title"] == "1.1 Replacing the Workpiece Holder"
    assert list(result["steps"].keys()) == [str(i) for i in range(1, 14)]
    assert result["first_step"] == "1"
    assert "Place the [Operation mode]" in result["steps"]["1"]["text"]
    assert "thickness gauge" in result["steps"]["13"]["text"]


def test_real_manual_section_stops_before_next_heading_same_or_higher_level():
    # 1.1's step text must NOT bleed into 1.2's steps (both use "(1)...(N)").
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Workpiece Holder")
    assert len(result["steps"]) == 13
    assert "Retract the phase datum" not in " ".join(s["text"] for s in result["steps"].values())


def test_real_manual_step_chain_links_next_correctly():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Workpiece Holder")
    steps = result["steps"]
    assert steps["1"]["next"] == "2"
    assert steps["12"]["next"] == "13"
    assert steps["13"]["next"] is None  # last step in the section


def test_real_manual_page_range_correct():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Workpiece Holder")
    assert result["page_range"] == [5, 5]


def test_real_manual_second_section_parses_independently():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Phase Datum Pad")
    assert result["section_title"] == "1.2 Replacing the Phase Datum Pad"
    assert len(result["steps"]) == 10
    assert result["page_range"] == [6, 6]


def test_real_manual_disambiguates_similarly_named_sections():
    # The real risk this feature exists for: three "Replacing..." procedures in
    # one manual. A loose/partial hint must still pick the exact right one.
    r1 = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Phase Indexing Unit")
    assert r1["section_title"] == "1.3 Replacing the Pad of Phase Indexing Unit"
    assert len(r1["steps"]) == 2  # only 2 steps given in this truncated fixture


def test_locate_by_start_page_instead_of_hint():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, start_page=6)
    assert result["section_title"] == "1.2 Replacing the Phase Datum Pad"


# ── Fail-closed behavior ──────────────────────────────────────────────────────

def test_no_section_hint_or_start_page_returns_none():
    assert parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS) is None


def test_unmatched_section_hint_returns_none():
    assert parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Nonexistent Section XYZ") is None


def test_section_with_no_step_format_returns_none():
    blocks = [
        _heading("2. GENERAL SAFETY", 1),
        _text("This section describes general safety precautions in prose, with no numbered steps at all.", 1),
    ]
    assert parse_procedure_from_blocks(blocks, section_hint="GENERAL SAFETY") is None


def test_gapped_step_numbers_fail_closed_not_guessed():
    blocks = [
        _heading("3. PROCEDURE", 1),
        _text("(1) Do this.\n(2) Do that.\n(5) Do something much later.\n", 1),
    ]
    assert parse_procedure_from_blocks(blocks, section_hint="PROCEDURE") is None


def test_out_of_order_step_numbers_fail_closed():
    blocks = [
        _heading("3. PROCEDURE", 1),
        _text("(2) Do that first in the text.\n(1) But labelled step one.\n(3) Then this.\n", 1),
    ]
    assert parse_procedure_from_blocks(blocks, section_hint="PROCEDURE") is None


def test_no_headings_at_all_returns_none():
    blocks = [_text("(1) A step with no heading above it at all.\n(2) Another.\n(3) Third.\n", 1)]
    assert parse_procedure_from_blocks(blocks, section_hint="anything") is None


def test_max_steps_guard_rejects_pathological_over_split():
    body = "\n".join(f"({i}) step {i}." for i in range(1, 60))
    blocks = [_heading("4. HUGE PROCEDURE", 1), _text(body, 1)]
    result = parse_procedure_from_blocks(blocks, section_hint="HUGE", max_steps=50)
    assert result is None


# ── Branch detection (best-effort, no confirmed real example) ────────────────

def test_explicit_step_reference_detected_as_branch():
    blocks = [
        _heading("5. COOLANT PROCEDURE", 1),
        _text(
            "(1) Check the coolant level.\n"
            "(2) If the level is low, refill using step 5. Otherwise continue to the next step.\n"
            "(3) Close the cover.\n",
            1,
        ),
    ]
    result = parse_procedure_from_blocks(blocks, section_hint="COOLANT")
    assert result["steps"]["2"].get("branches") == [{"condition": "the level is low, refill using", "next": "5"}]
    assert result["steps"]["2"]["next"] == "3"  # sequential fallback still set


def test_no_branches_key_when_no_conditional_reference_present():
    result = parse_procedure_from_blocks(_REAL_MANUAL_BLOCKS, section_hint="Replacing the Workpiece Holder")
    assert all("branches" not in s for s in result["steps"].values())
