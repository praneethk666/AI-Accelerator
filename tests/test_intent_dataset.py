"""Offline checks for the intent evaluation set — no LLM, no network, no keys.

The live eval (backend/evaluation/intent_eval.py) needs a provider key and is run
by hand. These tests guard the things that can rot silently: an invalid label, a
duplicate case, a follow-up with no conversation to follow up on, or metric maths
that quietly stops being right.
"""
from __future__ import annotations

from backend.agent.intent_classifier import (
    ACTION,
    DOCUMENT_QUESTION,
    FOLLOW_UP,
    GENERAL,
    VALID_INTENTS,
)
from backend.evaluation.intent_dataset import (
    ALL_LABELS,
    CASES,
    TOOL_REQUIRING_LABELS,
    cases_for,
    to_messages,
)
from backend.evaluation.intent_eval import _summarise


# ── dataset integrity ─────────────────────────────────────────────────────────
def test_every_case_has_a_valid_label():
    for case in CASES:
        assert case.expected in VALID_INTENTS, f"bad label on: {case.message}"


def test_no_duplicate_messages():
    seen = [c.message.strip().lower() for c in CASES]
    dupes = {m for m in seen if seen.count(m) > 1}
    assert not dupes, f"duplicate cases skew accuracy: {dupes}"


def test_follow_ups_always_carry_history():
    # "summarise that" with no prior turn is unanswerable — and unclassifiable.
    for case in cases_for(FOLLOW_UP):
        assert case.history, f"follow-up without history: {case.message}"


def test_non_follow_ups_do_not_need_history():
    # keeps the set honest: history should be the signal for follow-ups, not a
    # confound that leaks into the other classes.
    for label in (DOCUMENT_QUESTION, ACTION, GENERAL):
        for case in cases_for(label):
            assert not case.history, f"unexpected history on {label}: {case.message}"


def test_every_label_is_represented_and_reasonably_balanced():
    counts = {label: len(cases_for(label)) for label in ALL_LABELS}
    assert all(n >= 10 for n in counts.values()), f"thin coverage: {counts}"
    # no class may dominate enough to flatter the headline number
    assert max(counts.values()) <= 3 * min(counts.values()), f"unbalanced: {counts}"


def test_both_routes_are_covered():
    forced = [c for c in CASES if c.expected in TOOL_REQUIRING_LABELS]
    direct = [c for c in CASES if c.expected not in TOOL_REQUIRING_LABELS]
    assert len(forced) >= 15 and len(direct) >= 15


def test_borderline_cases_explain_themselves():
    # a judgement call without a rationale is unreviewable six months later
    for case in CASES:
        if case.note:
            assert len(case.note) > 20, f"note too thin: {case.message}"


def test_history_converts_to_messages():
    case = cases_for(FOLLOW_UP)[0]
    msgs = to_messages(case.history)
    assert len(msgs) == len(case.history)
    assert msgs[0].__class__.__name__.startswith("Human")
    assert msgs[1].__class__.__name__.startswith("AI")


# ── metric maths (the numbers the report is judged on) ───────────────────────
def _row(expected, predicted, ms=10):
    return {"expected": expected, "predicted": predicted,
            "correct": expected == predicted, "fallback": False,
            "requires_tools": predicted in TOOL_REQUIRING_LABELS,
            "had_history": False, "note": "", "ms": ms}


def test_summarise_counts_a_perfect_run():
    rows = [_row(label, label) for label in ALL_LABELS]
    s = _summarise(rows)
    assert s["accuracy"] == 1.0 and s["routing_accuracy"] == 1.0
    assert s["grounding_misses"] == 0


def test_wrong_label_same_route_keeps_routing_accuracy_perfect():
    # general -> follow_up: both allow a direct answer, so behaviour is unchanged
    s = _summarise([_row(GENERAL, FOLLOW_UP)])
    assert s["accuracy"] == 0.0            # the label was wrong
    assert s["routing_accuracy"] == 1.0    # the route was right
    assert s["grounding_misses"] == 0


def test_grounding_miss_is_counted_when_a_document_question_answers_freely():
    s = _summarise([_row(DOCUMENT_QUESTION, GENERAL)])
    assert s["grounding_misses"] == 1
    assert s["routing_accuracy"] == 0.0


def test_over_searching_is_not_a_grounding_miss():
    # general -> document_question wastes a search but stays grounded; it must not
    # be counted against the safety metric
    s = _summarise([_row(GENERAL, DOCUMENT_QUESTION)])
    assert s["grounding_misses"] == 0
    assert s["routing_accuracy"] == 0.0


def test_per_class_precision_and_recall():
    rows = [
        _row(DOCUMENT_QUESTION, DOCUMENT_QUESTION),
        _row(DOCUMENT_QUESTION, ACTION),      # recall hit for document_question
        _row(GENERAL, ACTION),                # false positive for action
    ]
    s = _summarise(rows)
    assert s["per_class"][DOCUMENT_QUESTION]["recall"] == 0.5
    assert s["per_class"][ACTION]["precision"] == 0.0   # 0 true / 2 predicted
    assert s["per_class"][ACTION]["support"] == 0


def test_confusion_matrix_totals_match_the_rows():
    rows = [_row(DOCUMENT_QUESTION, DOCUMENT_QUESTION), _row(GENERAL, FOLLOW_UP)]
    s = _summarise(rows)
    assert sum(sum(r.values()) for r in s["matrix"].values()) == len(rows)
    assert s["matrix"][GENERAL][FOLLOW_UP] == 1
