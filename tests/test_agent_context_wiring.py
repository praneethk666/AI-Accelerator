"""Context compaction through run_agent — mocked LLMs, no network, no DB.

Phases 1-2 tested compaction and the watermark in isolation; this proves the
wiring: a long chat reaches the model as summary + recent turns, a short one is
untouched, and a summariser failure still answers the question.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.executor import run_agent
from backend.agent.intent_classifier import GENERAL

_CONFIG = {
    "query": {"agent": {
        "max_iterations": 5,
        "write_tools": ["ingest_document"],
        "context": {"enabled": True, "trigger_tokens": 500, "keep_recent": 6},
    }}
}
_NO_COMPACT = {
    "query": {"agent": {
        "max_iterations": 5,
        "write_tools": ["ingest_document"],
        "context": {"enabled": False},
    }}
}

SUMMARY = "Earlier: the user asked about brake torque; the manual gives 42 Nm."


class _Bound:
    def __init__(self, parent, tool_choice):
        self._parent, self._tool_choice = parent, tool_choice

    def invoke(self, messages, **kwargs):
        self._parent.invocations.append(list(messages))
        return self._parent.responses.pop(0)


class _ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations: list[list] = []

    def bind_tools(self, tools, **kwargs):
        return _Bound(self, kwargs.get("tool_choice"))

    def invoke(self, messages, **kwargs):
        self.invocations.append(list(messages))
        return self.responses.pop(0)


class _IntentLLM:
    def invoke(self, messages, **kwargs):
        return AIMessage(content=GENERAL)     # keep the loop to one turn


class _SummaryLLM:
    def __init__(self, summary=SUMMARY):
        self.summary, self.calls = summary, 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=self.summary)


class _BoomSummaryLLM:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("summariser down")


def _long_history(pairs=20, filler="y" * 200):
    out = []
    for i in range(pairs):
        out.append(HumanMessage(f"OLDQ{i} {filler}"))
        out.append(AIMessage(f"OLDA{i} {filler}"))
    return out


def _run(history, summary_llm, config=_CONFIG):
    llm = _ScriptedLLM([AIMessage(content="ok")])
    result = run_agent(
        "what is 2+2?",
        config=config,
        registry={},
        llm=llm,
        intent_llm=_IntentLLM(),
        summary_llm=summary_llm,
        conversation_history=history,
    )
    return result, llm


def _prompt_text(llm: _ScriptedLLM) -> str:
    return " ".join(str(getattr(m, "content", "")) for m in llm.invocations[0])


def _system_texts(llm: _ScriptedLLM) -> list[str]:
    return [m.content for m in llm.invocations[0] if isinstance(m, SystemMessage)]


# ── long chat is compacted ───────────────────────────────────────────────────
def test_long_history_is_summarised_into_the_prompt():
    summariser = _SummaryLLM()
    result, llm = _run(_long_history(), summariser)

    assert result["status"] == "done"
    assert summariser.calls == 1                      # summarised once
    assert any(SUMMARY in s for s in _system_texts(llm))   # summary reached the model
    assert "earlier conversation" in " ".join(_system_texts(llm)).lower()


def test_oldest_turns_are_dropped_but_recent_ones_survive_verbatim():
    _, llm = _run(_long_history(), _SummaryLLM())
    text = _prompt_text(llm)
    assert "OLDQ0" not in text          # oldest turn compressed away
    assert "OLDQ19" in text             # newest turn still verbatim


def test_compaction_shrinks_the_prompt():
    history = _long_history()
    _, compacted = _run(history, _SummaryLLM())
    _, plain = _run(history, _SummaryLLM(), config=_NO_COMPACT)
    assert len(_prompt_text(compacted)) < len(_prompt_text(plain))


# ── short chat untouched ─────────────────────────────────────────────────────
def test_short_history_is_not_summarised():
    summariser = _SummaryLLM()
    history = [HumanMessage("hello there"), AIMessage("hi, how can I help?")]
    _, llm = _run(history, summariser)

    assert summariser.calls == 0                      # no wasted LLM call
    text = _prompt_text(llm)
    assert "hello there" in text and "hi, how can I help?" in text
    assert SUMMARY not in text


def test_disabled_by_config_skips_compaction_entirely():
    summariser = _SummaryLLM()
    _, llm = _run(_long_history(), summariser, config=_NO_COMPACT)
    assert summariser.calls == 0                      # no summarisation attempted
    text = _prompt_text(llm)
    assert SUMMARY not in text                        # no summary injected
    # Turned off, the old behaviour stands: the last max_history (20) messages are
    # kept and everything before them is simply dropped, uncompressed.
    assert "OLDQ19" in text and "OLDQ10" in text
    assert "OLDQ0" not in text


# ── failure paths still answer ───────────────────────────────────────────────
def test_summariser_failure_falls_back_to_truncation_and_still_answers():
    result, llm = _run(_long_history(), _BoomSummaryLLM())
    assert result["status"] == "done" and result["answer"] == "ok"
    assert SUMMARY not in _prompt_text(llm)           # no bogus summary
    assert "OLDQ19" in _prompt_text(llm)              # recent turns preserved


def test_no_summariser_still_answers():
    # summary_llm=None with no usable config model -> truncate, don't crash
    result, _ = _run(_long_history(), None)
    assert result["status"] == "done"


def test_empty_history_is_safe():
    result, llm = _run([], _SummaryLLM())
    assert result["status"] == "done"
    assert "earlier conversation" not in " ".join(_system_texts(llm)).lower()
