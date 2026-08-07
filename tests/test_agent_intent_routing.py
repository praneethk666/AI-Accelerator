"""Intent routing through run_agent — mocked LLMs, no network, no keys, no DB.

Phase 1 tested the classifier in isolation; this proves the WIRING: the label
actually changes what the agent does. Two things must both flip, or the feature is
invisible:
  1. tool_choice on turn 0 (required -> auto)
  2. the system prompt's ALWAYS-SEARCH mandate (relaxed for direct-answer turns)
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.executor import DIRECT_ANSWER_OVERRIDE, run_agent
from backend.agent.intent_classifier import (
    DOCUMENT_QUESTION,
    FOLLOW_UP,
    GENERAL,
)

_CONFIG = {"query": {"agent": {"max_iterations": 5, "write_tools": ["ingest_document"]}}}


class _Bound:
    """One bind_tools() result. The executor pre-binds BOTH an `auto` and a
    `required` variant and picks between them per turn, so the meaningful signal
    is which binding actually got invoked — not which were created."""

    def __init__(self, parent: "_ScriptedLLM", tool_choice):
        self._parent = parent
        self._tool_choice = tool_choice

    def invoke(self, messages, **kwargs):
        self._parent.used_tool_choice.append(self._tool_choice)
        self._parent.invocations.append(list(messages))
        return self._parent.responses.pop(0)


class _ScriptedLLM:
    """Agent model: hands out tagged bindings, pops canned responses in order."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.invocations: list[list] = []
        self.used_tool_choice: list = []

    def bind_tools(self, tools, **kwargs):
        return _Bound(self, kwargs.get("tool_choice"))

    def invoke(self, messages, **kwargs):  # unbound calls (e.g. fast paths)
        self.invocations.append(list(messages))
        return self.responses.pop(0)


class _IntentLLM:
    """Classifier model: always returns the label it was constructed with."""

    def __init__(self, label: str):
        self.label = label

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self.label)


class _FakeSearchTool:
    name = "search_documents"
    description = "Search the ingested documents and answer a question."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": "42", "citations": []}


def _run(label: str, message: str, responses: list[AIMessage], **kw):
    llm = _ScriptedLLM(responses)
    search = _FakeSearchTool()
    result = run_agent(
        message,
        config=_CONFIG,
        registry={"search_documents": search},
        llm=llm,
        intent_llm=_IntentLLM(label),
        **kw,
    )
    return result, llm, search


def _turn0_tool_choice(llm: _ScriptedLLM):
    """Which binding the agent actually used on turn 0: 'required' or None (auto)."""
    return llm.used_tool_choice[0] if llm.used_tool_choice else None


def _system_prompt(llm: _ScriptedLLM) -> str:
    return llm.invocations[0][0].content


# ── document_question keeps today's grounded behaviour ────────────────────────
def test_document_question_forces_a_tool_and_keeps_the_mandate():
    result, llm, _ = _run(
        DOCUMENT_QUESTION,
        "what is the warranty period?",
        [
            AIMessage(content="", tool_calls=[{"name": "search_documents",
                                               "args": {"query": "warranty"},
                                               "id": "c1", "type": "tool_call"}]),
            AIMessage(content="42 months."),
        ],
    )
    assert result["intent"] == DOCUMENT_QUESTION
    assert _turn0_tool_choice(llm) == "required"          # turn 0 forced a tool
    assert DIRECT_ANSWER_OVERRIDE not in _system_prompt(llm)   # mandate intact


# ── general / follow_up allow a direct answer ─────────────────────────────────
def test_general_does_not_force_a_tool_and_relaxes_the_mandate():
    result, llm, search = _run(
        GENERAL, "what is 2+2?", [AIMessage(content="4.")],
    )
    assert result["intent"] == GENERAL
    assert result["answer"] == "4."
    assert search.calls == []                            # no pointless RAG
    assert _turn0_tool_choice(llm) is None                # turn 0 not forced (auto)
    assert DIRECT_ANSWER_OVERRIDE in _system_prompt(llm)  # mandate relaxed


def test_follow_up_answers_from_conversation_without_searching():
    history = [
        HumanMessage("what does the manual say about brakes?"),
        AIMessage("The manual specifies a 42 Nm torque."),
    ]
    result, llm, search = _run(
        FOLLOW_UP, "summarise that",
        [AIMessage(content="Brakes: 42 Nm torque.")],
        conversation_history=history,
    )
    assert result["intent"] == FOLLOW_UP
    assert search.calls == []                            # did not re-retrieve
    assert DIRECT_ANSWER_OVERRIDE in _system_prompt(llm)
    # the prior turns must actually reach the model, or it cannot answer
    contents = " ".join(str(getattr(m, "content", "")) for m in llm.invocations[0])
    assert "brakes" in contents.lower() and "42 Nm" in contents


# ── the relaxation permits, it does not forbid ────────────────────────────────
def test_direct_answer_turn_may_still_search_if_the_model_chooses():
    result, _, search = _run(
        GENERAL, "remind me what the manual said",
        [
            AIMessage(content="", tool_calls=[{"name": "search_documents",
                                               "args": {"query": "manual"},
                                               "id": "c1", "type": "tool_call"}]),
            AIMessage(content="It said 42."),
        ],
    )
    assert search.calls == [{"query": "manual"}]          # tool still available
    assert result["status"] == "done"


# ── failure path: classifier down -> pre-classifier behaviour ────────────────
def test_classifier_failure_falls_back_to_forcing_tools():
    class _BoomLLM:
        def invoke(self, messages, **kwargs):
            raise RuntimeError("classifier down")

    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "search_documents",
                                           "args": {"query": "x"}, "id": "c1",
                                           "type": "tool_call"}]),
        AIMessage(content="done"),
    ])
    result = run_agent(
        "what is 2+2?",                                   # would be `general`
        config=_CONFIG,
        registry={"search_documents": _FakeSearchTool()},
        llm=llm,
        intent_llm=_BoomLLM(),
    )
    assert result["intent"] == DOCUMENT_QUESTION and result["intent_fallback"] is True
    assert _turn0_tool_choice(llm) == "required"          # grounded behaviour kept
    assert DIRECT_ANSWER_OVERRIDE not in _system_prompt(llm)
