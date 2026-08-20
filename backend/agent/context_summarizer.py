"""Rolling-summary compaction for long chats.

The agent currently drops old turns: history is sliced to the last N messages, so
turn 21 silently forgets turn 1. This compresses the dropped part instead —
older turns become a running summary, recent turns stay verbatim.

    [older turns] -> summary paragraph -> prepended as context
    [recent K turns] -> passed through untouched

Design rules, in priority order:

1. Recency is never summarised. The last `keep_recent` messages always pass
   through verbatim — paraphrase is lossy, and follow-ups ("summarise that",
   "why did you say 42 Nm?") depend on the exact prior wording.
2. Compaction triggers on a TOKEN estimate, not a message count. Twenty one-line
   turns and twenty page-long answers are not the same context load.
3. Fails open. Any error returns the plain truncated history — i.e. exactly
   today's behaviour. A summariser outage must never break a chat.
4. The summary is bounded. Feeding a prior summary back in (`prior_summary`)
   folds it into the new one rather than stacking, so it cannot grow forever.

This module is pure: no DB, no config lookups, no network beyond the `llm` it is
handed. Caching the summary between turns is the caller's job (see Phase 2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Match backend/guardrails/token_budget.py so the two agree on what a token costs.
_CHARS_PER_TOKEN = 4

# Defaults chosen to be conservative: compact only once a chat is genuinely long,
# and keep a generous verbatim tail so follow-ups still work.
DEFAULT_TRIGGER_TOKENS = 3000   # compact once history exceeds this
DEFAULT_KEEP_RECENT = 6         # messages (≈3 Q&A pairs) kept verbatim
DEFAULT_SUMMARY_TOKENS = 400    # cap on the generated summary

SUMMARY_PROMPT = """You maintain a running summary of a conversation between a user \
and a document-intelligence assistant, so earlier turns can be dropped without \
losing what matters.

Write a compact summary that preserves:
- what the user is trying to accomplish, and any constraints they gave
- documents, files, page numbers, part numbers, and other identifiers mentioned
- specific facts and figures the assistant reported (keep numbers and units exact)
- decisions made and questions left open

Omit pleasantries and restatements. Use plain prose, third person, no preamble and \
no headings. Be specific: "the user asked about front brake torque; the manual gives \
42 Nm (Service Manual p.87)" is useful, "the user asked about a specification" is not."""


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token), matching the guardrails estimator."""
    return len(text) // _CHARS_PER_TOKEN if text else 0


def _content(message) -> str:
    value = getattr(message, "content", "")
    return value if isinstance(value, str) else str(value)


def total_tokens(messages) -> int:
    """Estimated token load of a message list."""
    return sum(estimate_tokens(_content(m)) for m in messages or [])


def _role(message) -> str:
    return "User" if message.__class__.__name__.startswith("Human") else "Assistant"


def _transcript(messages) -> str:
    return "\n".join(f"{_role(m)}: {_content(m)}" for m in messages if _content(m))


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of one compaction pass.

    summary:     running summary of the dropped turns ("" when none was made)
    messages:    the turns to send verbatim
    compacted:   how many messages the summary replaced
    triggered:   whether compaction actually ran
    fallback:    True when summarisation failed and history was merely truncated
    """

    summary: str
    messages: list
    compacted: int = 0
    triggered: bool = False
    fallback: bool = False


def split_history(messages, keep_recent: int = DEFAULT_KEEP_RECENT):
    """Split into (older, recent). `recent` is never summarised.

    Splits on a Human message where possible so a kept tail starts with a user
    turn rather than a dangling assistant reply.
    """
    messages = list(messages or [])
    if keep_recent <= 0 or len(messages) <= keep_recent:
        return [], messages

    cut = len(messages) - keep_recent
    # walk forward to the next user turn so the verbatim tail reads as whole exchanges
    for i in range(cut, len(messages)):
        if messages[i].__class__.__name__.startswith("Human"):
            cut = i
            break
    return messages[:cut], messages[cut:]


def summarize_messages(messages, llm, *, prior_summary: str = "",
                       max_tokens: int = DEFAULT_SUMMARY_TOKENS) -> str:
    """Summarise `messages`, folding in `prior_summary`. Returns "" on failure.

    Folding rather than stacking is what keeps the summary bounded across a long
    chat: each pass rewrites one summary, it never appends a new one.
    """
    transcript = _transcript(messages)
    if not transcript.strip() and not prior_summary.strip():
        return ""

    from langchain_core.messages import HumanMessage, SystemMessage

    parts = []
    if prior_summary.strip():
        parts.append(f"Summary of the conversation so far:\n{prior_summary.strip()}")
    if transcript.strip():
        parts.append(f"Newer turns to fold in:\n{transcript}")
    parts.append(
        f"Write the updated summary in at most {max_tokens // 2} words. "
        "It must stand alone — the turns above will be discarded."
    )

    try:
        response = llm.invoke([
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage("\n\n".join(parts)),
        ])
        content = getattr(response, "content", "")
        return content.strip() if isinstance(content, str) else str(content).strip()
    except Exception as exc:
        logger.warning("context summarisation failed (%s); keeping recent turns only", exc)
        return ""


def compact_history(
    messages,
    llm=None,
    *,
    prior_summary: str = "",
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    max_summary_tokens: int = DEFAULT_SUMMARY_TOKENS,
) -> CompactionResult:
    """Compact history if it is long enough to warrant it.

    Under the trigger, or with no model available, history passes through
    unchanged — carrying any `prior_summary` forward so a previously compacted
    session keeps its context.
    """
    messages = list(messages or [])

    if total_tokens(messages) <= trigger_tokens:
        return CompactionResult(summary=prior_summary, messages=messages)

    older, recent = split_history(messages, keep_recent)
    if not older:
        # everything is inside the verbatim window; nothing to compress
        return CompactionResult(summary=prior_summary, messages=recent)

    if llm is None:
        # no summariser available -> truncate, which is the pre-existing behaviour
        logger.info("context over budget but no summariser configured; truncating")
        return CompactionResult(summary=prior_summary, messages=recent,
                                compacted=len(older), triggered=True, fallback=True)

    summary = summarize_messages(older, llm, prior_summary=prior_summary,
                                 max_tokens=max_summary_tokens)
    if not summary:
        return CompactionResult(summary=prior_summary, messages=recent,
                                compacted=len(older), triggered=True, fallback=True)

    logger.info("compacted %d messages into a %d-token summary",
                len(older), estimate_tokens(summary))
    return CompactionResult(summary=summary, messages=recent,
                            compacted=len(older), triggered=True)


def as_context_message(summary: str):
    """Wrap a summary as the system message that carries it into the prompt."""
    from langchain_core.messages import SystemMessage

    return SystemMessage(
        "Summary of earlier conversation (the turns themselves have been dropped "
        f"to save context):\n{summary.strip()}"
    )


def uncovered_window(window, total_messages: int, covered: int) -> list:
    """Drop the part of a loaded window the summary already accounts for.

    Only the last `len(window)` of a session's `total_messages` are loaded, so the
    window starts at absolute index `total_messages - len(window)`. Anything before
    `covered` is already in the summary; folding it in again would double-count it
    and let the summary drift.
    """
    window = list(window or [])
    if covered <= 0 or not window:
        return window
    window_start = max(0, total_messages - len(window))
    already_in_summary = covered - window_start
    if already_in_summary <= 0:
        return window
    return window[already_in_summary:] if already_in_summary < len(window) else []


def compact_session(
    window,
    *,
    llm=None,
    total_messages: int = 0,
    prior_summary: str = "",
    covered: int = 0,
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    max_summary_tokens: int = DEFAULT_SUMMARY_TOKENS,
) -> tuple[CompactionResult, int]:
    """Compact a persisted session's loaded window.

    Wraps compact_history with the watermark arithmetic, returning the result plus
    the NEW covered count to persist. The caller does the storage I/O so this stays
    testable without a database.

    Returns:
        (result, new_covered)
    """
    window = list(window or [])
    total_messages = max(total_messages, len(window))

    fresh = uncovered_window(window, total_messages, covered)
    # Everything already summarised is gone from `fresh`, so the trigger is judged
    # on what is actually about to be sent.
    result = compact_history(
        fresh, llm,
        prior_summary=prior_summary,
        trigger_tokens=trigger_tokens,
        keep_recent=keep_recent,
        max_summary_tokens=max_summary_tokens,
    )

    if not result.triggered or result.fallback:
        # nothing new was folded in -> the watermark must not move, or those turns
        # would be lost from both the summary and the verbatim window
        return result, covered

    new_covered = total_messages - len(result.messages)
    return result, max(covered, new_covered)
