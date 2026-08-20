"""Watermark arithmetic for persisted rolling summaries — no DB, no network.

The subtle failure mode this guards: a turn folded into the summary twice (drift
and wasted tokens), or a turn that falls out of the verbatim window *and* never
reaches the summary (silently forgotten). Both come from the watermark moving
wrongly, so the arithmetic is tested on its own.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.context_summarizer import (
    compact_session,
    uncovered_window,
)


class _SummaryLLM:
    def __init__(self, summary: str = "running summary"):
        self.summary = summary
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.summary)


class _BoomLLM:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("summariser down")


def _chat(pairs: int, filler: str = "x", tag: str = "q") -> list:
    """`tag` labels a segment so tests can tell old turns from newly-arrived ones."""
    out = []
    for i in range(pairs):
        out.append(HumanMessage(f"{tag}{i} {filler}"))
        out.append(AIMessage(f"a-{tag}{i} {filler}"))
    return out


# ── uncovered_window ─────────────────────────────────────────────────────────
def test_nothing_covered_returns_the_whole_window():
    window = _chat(3)
    assert uncovered_window(window, total_messages=6, covered=0) == window


def test_covered_prefix_is_dropped_from_the_window():
    window = _chat(5)                       # 10 messages, the whole session
    out = uncovered_window(window, total_messages=10, covered=4)
    assert out == window[4:]                # first 4 already summarised


def test_window_offset_is_accounted_for():
    # session has 20 messages; only the last 10 are loaded, so the window starts
    # at absolute index 10. A watermark of 12 covers the window's first 2.
    window = _chat(5)
    out = uncovered_window(window, total_messages=20, covered=12)
    assert out == window[2:]


def test_watermark_behind_the_window_drops_nothing():
    window = _chat(5)
    # covered=6 is before the window start (10) -> window is entirely fresh
    assert uncovered_window(window, total_messages=20, covered=6) == window


def test_fully_covered_window_is_empty():
    window = _chat(3)
    assert uncovered_window(window, total_messages=6, covered=6) == []


def test_empty_window_is_safe():
    assert uncovered_window([], total_messages=0, covered=0) == []
    assert uncovered_window(None, total_messages=10, covered=5) == []


# ── compact_session: watermark movement ──────────────────────────────────────
def test_short_session_does_not_move_the_watermark():
    window = _chat(3)
    result, covered = compact_session(window, llm=_SummaryLLM(), total_messages=6,
                                      covered=0, trigger_tokens=10_000)
    assert result.triggered is False
    assert covered == 0                     # nothing summarised, nothing covered


def test_compaction_advances_the_watermark_to_the_verbatim_boundary():
    window = _chat(20, filler="y" * 200)    # 40 messages
    result, covered = compact_session(window, llm=_SummaryLLM(), total_messages=40,
                                      covered=0, trigger_tokens=500, keep_recent=6)
    assert result.triggered is True
    # everything not kept verbatim is now covered
    assert covered == 40 - len(result.messages)
    assert covered > 0


def test_second_pass_does_not_refold_already_covered_turns():
    llm = _SummaryLLM()
    window = _chat(20, filler="y" * 200)
    _, covered = compact_session(window, llm=llm, total_messages=40, covered=0,
                                 trigger_tokens=500, keep_recent=6)

    # same window comes back next turn; everything before `covered` must be excluded
    llm2 = _SummaryLLM()
    result2, covered2 = compact_session(window, llm=llm2, total_messages=40,
                                        covered=covered, prior_summary="running summary",
                                        trigger_tokens=500, keep_recent=6)
    assert result2.triggered is False        # nothing new aged out
    assert covered2 == covered               # watermark held still
    assert llm2.prompts == []                # and no wasted summarisation call


def test_new_turns_after_a_compaction_are_folded_once():
    llm = _SummaryLLM()
    first = _chat(20, filler="y" * 200, tag="OLD")
    _, covered = compact_session(first, llm=llm, total_messages=40, covered=0,
                                 trigger_tokens=500, keep_recent=6)

    # 20 more messages arrive, tagged so they are distinguishable from the old ones
    grown = first + _chat(10, filler="z" * 200, tag="NEW")
    llm2 = _SummaryLLM("updated summary")
    result2, covered2 = compact_session(grown, llm=llm2, total_messages=60,
                                        covered=covered, prior_summary="running summary",
                                        trigger_tokens=500, keep_recent=6)
    assert result2.triggered is True
    assert covered2 > covered                       # watermark advanced
    folded = llm2.prompts[0]
    assert "running summary" in folded              # prior summary folded in
    assert "NEW0" in folded                         # newly aged-out turns folded
    assert "OLD0" not in folded                     # already-covered turns excluded


def test_failure_does_not_advance_the_watermark():
    # if the watermark moved on failure, those turns would vanish from both the
    # summary and the verbatim window — silently forgotten
    window = _chat(20, filler="y" * 200)
    result, covered = compact_session(window, llm=_BoomLLM(), total_messages=40,
                                      covered=3, trigger_tokens=500, keep_recent=6)
    assert result.fallback is True
    assert covered == 3


def test_no_llm_does_not_advance_the_watermark():
    window = _chat(20, filler="y" * 200)
    _, covered = compact_session(window, llm=None, total_messages=40, covered=7,
                                 trigger_tokens=500, keep_recent=6)
    assert covered == 7


def test_watermark_never_moves_backwards():
    window = _chat(20, filler="y" * 200)
    _, covered = compact_session(window, llm=_SummaryLLM(), total_messages=40,
                                 covered=999, trigger_tokens=500, keep_recent=6)
    assert covered >= 999


def test_total_messages_defaults_to_the_window_when_unknown():
    # a caller that cannot count the session still gets correct behaviour for the
    # common case where the window IS the whole conversation
    window = _chat(20, filler="y" * 200)
    result, covered = compact_session(window, llm=_SummaryLLM(), total_messages=0,
                                      covered=0, trigger_tokens=500, keep_recent=6)
    assert result.triggered is True
    assert covered == len(window) - len(result.messages)
