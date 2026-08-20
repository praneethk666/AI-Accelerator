"""Intent classification for the agent loop.

One cheap LLM call that labels what the user's message actually wants, so the
executor stops forcing a tool call on every non-greeting turn.

The label decides ONE thing: must a tool run on turn 0?
    
    document_question -> yes   needs the corpus ("what's the torque spec?")
    action            -> yes   wants a tool that DOES something ("ingest this file")
    follow_up         -> no    answerable from the conversation ("summarise that")
    general           -> no    model's own knowledge / chit-chat ("what is 2+2?")

"no" means tool_choice="auto", NOT "tools forbidden" — the model may still search
if it judges that necessary. So a misclassification degrades gracefully.

Fail-open: any error, timeout, or unparseable reply falls back to
document_question (tools required) — i.e. exactly the behaviour before this
module existed. A broken classifier must never make the agent LESS grounded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.core.llm_client import clean_message_content, get_llm_for

logger = logging.getLogger(__name__)

# Labels that force a tool call on turn 0 (i.e. keep today's grounded behaviour).
DOCUMENT_QUESTION = "document_question"
ACTION = "action"
FOLLOW_UP = "follow_up"
GENERAL = "general"

TOOL_REQUIRING = frozenset({DOCUMENT_QUESTION, ACTION})
VALID_INTENTS = frozenset({DOCUMENT_QUESTION, ACTION, FOLLOW_UP, GENERAL})

# Safe default: if we cannot classify, behave exactly as the agent did before.
DEFAULT_INTENT = DOCUMENT_QUESTION

CLASSIFIER_PROMPT = """You classify what a user's message needs. Reply with EXACTLY \
one of these labels and nothing else:

document_question - needs information from the user's uploaded documents \
(specs, manuals, reports, invoices, drawings, data in their files).
action - asks the system to DO something to the corpus or database: ingest, upload, \
index or re-process a file; list, count or check the status of ingested files; run a \
database query. Not merely mentioning a file, and not reformatting an answer.
follow_up - refers to the conversation so far and can be answered from it, including \
restating, reformatting, shortening, tabulating or translating a previous answer, e.g. \
"summarise that", "explain your last answer", "put that in a table".
general - general knowledge, reasoning, chit-chat, maths, or definitions that need \
neither the documents nor the prior conversation. Only choose this when the question \
would have the same answer for any user; if it turns on THIS user's equipment, \
paperwork or data, it is a document_question.

Tie-breakers, in order:
1. A terse phrase or bare noun phrase with no conversational reference (e.g. "model \
name", "max load", "warranty terms") is a document_question — the user is naming a \
field they want looked up in their files.
2. "the" before a thing the user owns or filed ("the contract", "the battery", "the \
manual") points at their documents even with no filename given — "who signed the \
contract?" is a document_question, not general knowledge.
3. Asking whether something is present or stated ("are there any warnings about X?", \
"does it mention Y?") is a document_question: the files must be searched to answer, \
and "no" is still an answer that requires looking.
4. A question about recommended values, intervals, procedures or requirements for \
equipment is a document_question — those live in the user's manuals. Choose general \
only when the user explicitly signals they want generic or industry-wide context \
("generally", "in general terms", "what's a reasonable X to buy").
5. If the message plausibly needs the user's documents at all, choose \
document_question. Answering a document question without the documents is the worst \
outcome; an unnecessary search is merely wasteful.
Reply with the label only."""


@dataclass(frozen=True)
class IntentResult:
    """Outcome of one classification.

    intent:         one of VALID_INTENTS
    requires_tools: whether turn 0 should force tool_choice="required"
    fallback:       True when classification failed and DEFAULT_INTENT was used
    """

    intent: str
    requires_tools: bool
    fallback: bool = False


def _result(intent: str, *, fallback: bool = False) -> IntentResult:
    return IntentResult(intent, intent in TOOL_REQUIRING, fallback)


def parse_intent(raw: str | None) -> str | None:
    """Map a model reply to a known label, or None if it matches nothing.

    Tolerant on purpose: models add quotes, punctuation, or a short preamble.
    """
    if not raw:
        return None
    text = raw.strip().strip("\"'`.").lower()
    if text in VALID_INTENTS:
        return text
    # Longer reply — take the first label mentioned (earliest position wins, so a
    # trailing restatement can't override the actual answer).
    hits = [(text.find(label), label) for label in VALID_INTENTS if label in text]
    return min(hits)[1] if hits else None


def _history_snippet(conversation_history, max_turns: int = 4, max_chars: int = 400) -> str:
    """Compact recent turns so the classifier can recognise follow-ups.

    Only role + truncated text: the classifier needs the shape of the exchange,
    not its full content, and short prompts keep this call cheap.
    """
    if not conversation_history:
        return ""
    lines = []
    for msg in list(conversation_history)[-max_turns:]:
        content = clean_message_content(getattr(msg, "content", "") or "")
        if not content:
            continue
        role = "User" if msg.__class__.__name__.startswith("Human") else "Assistant"
        lines.append(f"{role}: {content[:max_chars]}")
    return "\n".join(lines)


def classify_intent(
    message: str,
    *,
    conversation_history=None,
    config: dict | None = None,
    agent_cfg: dict | None = None,
    llm=None,
) -> IntentResult:
    """Classify one user message. Never raises — falls back to DEFAULT_INTENT.

    Args:
        message: the user's message.
        conversation_history: recent BaseMessages, used to spot follow-ups.
        config: full loaded config (for the model factory).
        agent_cfg: config["query"]["agent"]; its optional `intent` block may
            override the model/provider so this can run on something cheap.
        llm: prebuilt chat model — tests inject a mock; skips config entirely.

    Returns:
        IntentResult(intent, requires_tools, fallback)
    """
    if not message or not message.strip():
        return _result(DEFAULT_INTENT, fallback=True)

    intent_cfg = (agent_cfg or {}).get("intent") or {}
    if not intent_cfg.get("enabled", True):
        # Disabled -> keep the pre-classifier behaviour (always force tools).
        return _result(DEFAULT_INTENT, fallback=True)

    try:
        if llm is None:
            llm = get_llm_for(config or {}, {**(agent_cfg or {}), **intent_cfg},
                              max_tokens=intent_cfg.get("max_tokens", 12))

        history = _history_snippet(conversation_history)
        user_block = f"Conversation so far:\n{history}\n\n" if history else ""
        prompt = f"{user_block}Message to classify:\n{message}"

        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke([SystemMessage(CLASSIFIER_PROMPT), HumanMessage(prompt)])
        intent = parse_intent(clean_message_content(getattr(response, "content", "")))

        if intent is None:
            logger.warning("intent classifier returned an unusable label; defaulting")
            return _result(DEFAULT_INTENT, fallback=True)

        logger.info("[INTENT] %s (tools_required=%s)", intent, intent in TOOL_REQUIRING)
        return _result(intent)

    except Exception as exc:  # network, auth, provider error — degrade, don't break
        logger.warning("intent classification failed (%s); defaulting to %s",
                       exc, DEFAULT_INTENT)
        return _result(DEFAULT_INTENT, fallback=True)
