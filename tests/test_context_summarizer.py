"""Unit tests for rolling-summary compaction — mocked LLM, no network, no DB.

The invariants that matter: recent turns are never paraphrased, compaction only
fires when the chat is genuinely long, the summary stays bounded, and every
failure path degrades to plain truncation rather than breaking the chat.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.context_summarizer import (
    CompactionResult,
    as_context_message,
    compact_history,
    estimate_tokens,
    split_history,
    summarize_messages,
    total_tokens,
)


class _SummaryLLM:
    """Returns a canned summary and records the prompt it was given."""

    def __init__(self, summary: str = "The user asked about brake torque; 42 Nm."):
        self.summary = summary
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.summary)


class _BoomLLM:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("summariser down")


def _chat(pairs: int, filler: str = "x") -> list:
    """`pairs` Q&A exchanges; `filler` pads each message to control token load."""
    out = []
    for i in range(pairs):
        out.append(HumanMessage(f"question {i} {filler}"))
        out.append(AIMessage(f"answer {i} {filler}"))
    return out


# ── token estimation ─────────────────────────────────────────────────────────
def test_estimate_tokens_matches_the_four_chars_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_total_tokens_sums_message_content():
    msgs = [HumanMessage("a" * 400), AIMessage("b" * 400)]
    assert total_tokens(msgs) == 200
    assert total_tokens([]) == 0
    assert total_tokens(None) == 0


# ── splitting: recency is sacred ─────────────────────────────────────────────
def test_split_keeps_the_recent_tail_verbatim():
    msgs = _chat(10)
    older, recent = split_history(msgs, keep_recent=6)
    assert len(recent) >= 6
    assert recent == msgs[-len(recent):]        # untouched, in order
    assert older + recent == msgs               # nothing lost or duplicated


def test_split_starts_the_tail_on_a_user_turn():
    # a tail beginning with a dangling assistant reply reads as a non-sequitur
    older, recent = split_history(_chat(10), keep_recent=5)
    assert recent[0].__class__.__name__.startswith("Human")


def test_short_history_is_never_split():
    msgs = _chat(2)
    older, recent = split_history(msgs, keep_recent=6)
    assert older == [] and recent == msgs


def test_keep_recent_zero_still_returns_everything():
    msgs = _chat(3)
    older, recent = split_history(msgs, keep_recent=0)
    assert older == [] and recent == msgs


# ── trigger threshold ────────────────────────────────────────────────────────
def test_short_chat_is_not_compacted_and_costs_no_llm_call():
    llm = _SummaryLLM()
    msgs = _chat(3)
    out = compact_history(msgs, llm, trigger_tokens=10_000)
    assert out.triggered is False
    assert out.messages == msgs                 # passed through whole
    assert out.summary == ""
    assert llm.prompts == []                    # no model call made


def test_long_chat_is_compacted():
    llm = _SummaryLLM()
    msgs = _chat(20, filler="y" * 200)
    out = compact_history(msgs, llm, trigger_tokens=500, keep_recent=6)
    assert out.triggered is True and out.fallback is False
    assert out.compacted > 0
    assert out.summary == llm.summary
    assert len(out.messages) < len(msgs)        # older turns dropped
    assert out.messages == msgs[-len(out.messages):]   # the tail is verbatim


def test_prior_summary_is_carried_forward_when_not_triggered():
    # a session compacted earlier must not lose its summary on a later short turn
    out = compact_history(_chat(2), _SummaryLLM(), prior_summary="earlier context",
                          trigger_tokens=10_000)
    assert out.summary == "earlier context" and out.triggered is False


def test_prior_summary_is_folded_in_not_stacked():
    llm = _SummaryLLM()
    msgs = _chat(20, filler="y" * 200)
    compact_history(msgs, llm, prior_summary="PRIOR CONTEXT",
                    trigger_tokens=500, keep_recent=6)
    prompt = llm.prompts[0]
    assert "PRIOR CONTEXT" in prompt            # the old summary went in
    assert prompt.count("PRIOR CONTEXT") == 1   # once — folded, not appended


# ── failure paths degrade to truncation ──────────────────────────────────────
def test_summariser_failure_falls_back_to_truncation():
    msgs = _chat(20, filler="y" * 200)
    out = compact_history(msgs, _BoomLLM(), trigger_tokens=500, keep_recent=6)
    assert out.fallback is True and out.triggered is True
    assert out.summary == ""                    # no bogus summary invented
    assert out.messages == msgs[-len(out.messages):]   # recent turns still intact


def test_no_llm_available_falls_back_to_truncation():
    msgs = _chat(20, filler="y" * 200)
    out = compact_history(msgs, None, trigger_tokens=500, keep_recent=6)
    assert out.fallback is True
    assert len(out.messages) < len(msgs)


def test_failure_preserves_a_prior_summary():
    # losing the running summary on a transient error would silently forget the chat
    msgs = _chat(20, filler="y" * 200)
    out = compact_history(msgs, _BoomLLM(), prior_summary="KEEP ME",
                          trigger_tokens=500, keep_recent=6)
    assert out.summary == "KEEP ME"


def test_empty_history_is_safe():
    out = compact_history([], _SummaryLLM())
    assert out.messages == [] and out.summary == "" and out.triggered is False


# ── summarize_messages directly ──────────────────────────────────────────────
def test_summarize_returns_empty_string_on_failure():
    assert summarize_messages(_chat(2), _BoomLLM()) == ""


def test_summarize_of_nothing_is_empty_without_calling_the_model():
    llm = _SummaryLLM()
    assert summarize_messages([], llm) == ""
    assert llm.prompts == []


def test_summary_prompt_contains_the_transcript_with_roles():
    llm = _SummaryLLM()
    summarize_messages([HumanMessage("what is the torque?"),
                        AIMessage("42 Nm per the manual.")], llm)
    prompt = llm.prompts[0]
    assert "User: what is the torque?" in prompt
    assert "Assistant: 42 Nm per the manual." in prompt


def test_summary_length_instruction_scales_with_budget():
    llm = _SummaryLLM()
    summarize_messages(_chat(2), llm, max_tokens=400)
    assert "200 words" in llm.prompts[0]


# ── the context message handed to the model ──────────────────────────────────
def test_as_context_message_is_a_system_message_naming_the_summary():
    msg = as_context_message("  the user asked about brakes  ")
    assert isinstance(msg, SystemMessage)
    assert "the user asked about brakes" in msg.content
    assert "earlier conversation" in msg.content.lower()


def test_result_is_immutable():
    out = CompactionResult(summary="s", messages=[])
    try:
        out.summary = "changed"
    except Exception:
        return
    raise AssertionError("CompactionResult should be frozen")
