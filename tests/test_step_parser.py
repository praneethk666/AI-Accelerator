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


# ── Real transcript: section 1.4 of the same manual, where Docling merges
# several consecutive numbered steps into ONE block with no newline between
# markers -- real bug found+fixed 11-Aug (a line-anchored regex only caught the
# first marker per block). Verbatim from the real extracted blocks.

_INLINE_MERGED_STEPS_BLOCKS = [
    _heading("1. CHANGING THE SETUP OF WORKPIECE HOLDER AND PHASE INDEXING", 7),
    _heading("1.4 Setting the Phase Indexing Encoder", 7),
    _text(
        "(1) Turn the [EMERGENCY STOP] button clockwise to unlock it. "
        "(2) Press the [MASTER ON] button. "
        "(3) Place the [Operation mode] selection switch to the position of [MANU.]. "
        "(4) Place the [CNC MODE] selection switch in the [MEMORY] position. "
        "(5) Press the [GR.WHEEL SPINDLE STOP] button and stop the wheel rotation. "
        "(6) Load a pre-machined workpiece and execute the operations from the advancing of the phase datum. "
        "(7) Advance the phase indexing unit. "
        "(8) Input the part number at the controller (e.g. VS-T11) in the terminal box provided at the rear of the machine.",
        7,
    ),
    _text("Example: Part No. 7 Press the [MODE] key. Press the [–] key. Press the [+] key twice to display “7”. Press the [SET] key.", 7),
    _text(
        "(9) Press the [ON/OFF] key to turn on the ON indicator. "
        "(10) Press the [TEACH] key to store the present position data. <NOTE> The present position data has no dimension.",
        7,
    ),
    _text(
        "(11) Execute the operations from the retraction of the phase indexing unit to the retraction of both centers and unload the pre-machined workpiece. "
        "(12) Load a machined workpiece and execute the operations from the advancing of both centers to the advancing of the phase datum. "
        "(13) Advance the phase indexing unit. "
        "(14) Input the part number at the controller (e.g. VS-T11) in the terminal box provided at the rear of the machine.",
        7,
    ),
    _text("Example: Part No. 7 Press the [MODE] key. Press the [–] key. Press the [+] key twice to display “7”. Press the [SET] key.", 7),
    _text(
        "(15) Press the [ON/OFF] key to turn on the OFF indicator. "
        "(16) Press the [TEACH] key to store the present position data. "
        "(17) Execute the operations from the retraction of the phase indexing unit to the retraction of both centers and unload the machined workpiece.",
        7,
    ),
]


def test_inline_merged_steps_all_17_parsed_in_order():
    result = parse_procedure_from_blocks(
        _INLINE_MERGED_STEPS_BLOCKS, section_hint="1.4 Setting the Phase Indexing Encoder")
    assert result is not None
    assert sorted(int(k) for k in result["steps"]) == list(range(1, 18))
    assert result["steps"]["1"]["text"] == "Turn the [EMERGENCY STOP] button clockwise to unlock it."
    assert result["steps"]["17"]["text"].startswith("Execute the operations")


def test_inline_merged_steps_unmarked_example_asides_dont_become_their_own_steps():
    # The "Example: Part No. 7 ..." asides carry no "(N)" marker of their own,
    # so they're swept into whichever preceding step's span they fall in
    # (correct -- it's illustrative context for that step, not a separate one).
    # What matters is they don't spuriously inflate the step count or break
    # the contiguous sequence.
    result = parse_procedure_from_blocks(
        _INLINE_MERGED_STEPS_BLOCKS, section_hint="1.4 Setting the Phase Indexing Encoder")
    assert len(result["steps"]) == 17
    assert "Example: Part No." in result["steps"]["8"]["text"]


def test_stray_parenthesized_number_in_prose_does_not_produce_false_step():
    # "(2)" here is a citation-like aside mid-sentence, not a step marker -- the
    # text has no OTHER numbered markers around it, so even if it matched, the
    # contiguous-sequence check (or simply "only one match found") should not
    # yield a usable multi-step result.
    blocks = [
        _heading("3. TORQUE SPECIFICATIONS", 1),
        _text("Tighten the bolt per the reference table (2) shown in the appendix.", 1),
    ]
    result = parse_procedure_from_blocks(blocks, section_hint="3. TORQUE SPECIFICATIONS")
    assert result is None


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
