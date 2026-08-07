"""Headless tests for intent classification — mocked LLM, no network, no keys.

Covers: each label maps to the right tool requirement, parsing tolerates messy
model replies, follow-ups see the conversation, and every failure path falls back
to document_question (tools required) so a broken classifier can never make the
agent less grounded.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.intent_classifier import (
    ACTION,
    DEFAULT_INTENT,
    DOCUMENT_QUESTION,
    FOLLOW_UP,
    GENERAL,
    classify_intent,
    parse_intent,
)


class _ScriptedLLM:
    """Fake chat model: invoke() returns a canned label and records the prompt."""

    def __init__(self, reply: str):
        self._reply = reply
        self.invocations: list[list] = []

    def invoke(self, messages, **kwargs):
        self.invocations.append(list(messages))
        return AIMessage(content=self._reply)


class _BoomLLM:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("provider exploded")


def _classify(reply: str, message: str = "q", **kw):
    return classify_intent(message, llm=_ScriptedLLM(reply), **kw)


# ── label -> tool requirement ─────────────────────────────────────────────────
def test_document_question_requires_tools():
    out = _classify(DOCUMENT_QUESTION, "what is the torque spec?")
    assert out.intent == DOCUMENT_QUESTION
    assert out.requires_tools is True and out.fallback is False


def test_action_requires_tools():
    out = _classify(ACTION, "ingest uploads/report.pdf")
    assert out.intent == ACTION and out.requires_tools is True


def test_follow_up_does_not_require_tools():
    out = _classify(FOLLOW_UP, "summarise that")
    assert out.intent == FOLLOW_UP and out.requires_tools is False


def test_general_does_not_require_tools():
    out = _classify(GENERAL, "what is 2+2?")
    assert out.intent == GENERAL and out.requires_tools is False


# ── parsing tolerance ─────────────────────────────────────────────────────────
def test_parse_intent_handles_messy_replies():
    assert parse_intent(" GENERAL ") == GENERAL              # case + whitespace
    assert parse_intent('"follow_up".') == FOLLOW_UP          # quotes + punctuation
    assert parse_intent("Label: action") == ACTION            # preamble
    assert parse_intent("document_question is best here") == DOCUMENT_QUESTION


def test_parse_intent_picks_the_first_label_mentioned():
    # a trailing restatement must not override the actual answer
    assert parse_intent("general, not document_question") == GENERAL


def test_parse_intent_rejects_unknown():
    assert parse_intent("banana") is None
    assert parse_intent("") is None
    assert parse_intent(None) is None


def test_unparseable_reply_falls_back_to_tools_required():
    out = _classify("I think it depends")
    assert out.intent == DEFAULT_INTENT
    assert out.requires_tools is True and out.fallback is True


# ── failure paths all fail OPEN (tools required = today's behaviour) ──────────
def test_llm_error_falls_back():
    out = classify_intent("anything", llm=_BoomLLM())
    assert out.intent == DEFAULT_INTENT
    assert out.requires_tools is True and out.fallback is True


def test_empty_message_falls_back():
    out = classify_intent("   ", llm=_ScriptedLLM(GENERAL))
    assert out.intent == DEFAULT_INTENT and out.fallback is True


def test_disabled_by_config_falls_back_without_calling_the_model():
    llm = _ScriptedLLM(GENERAL)
    out = classify_intent("what is 2+2?", llm=llm,
                          agent_cfg={"intent": {"enabled": False}})
    assert out.intent == DEFAULT_INTENT and out.requires_tools is True
    assert llm.invocations == []          # no call made when switched off


# ── conversation history reaches the classifier ───────────────────────────────
def test_history_is_included_for_follow_up_detection():
    llm = _ScriptedLLM(FOLLOW_UP)
    classify_intent(
        "summarise that",
        llm=llm,
        conversation_history=[
            HumanMessage("what does the manual say about brakes?"),
            AIMessage("The manual specifies a 42 Nm torque."),
        ],
    )
    prompt = llm.invocations[0][-1].content
    assert "brakes" in prompt and "42 Nm" in prompt       # both turns present
    assert "User:" in prompt and "Assistant:" in prompt   # roles labelled


def test_no_history_still_classifies():
    llm = _ScriptedLLM(GENERAL)
    out = classify_intent("what is 2+2?", llm=llm, conversation_history=[])
    assert out.intent == GENERAL
    assert "Conversation so far" not in llm.invocations[0][-1].content
