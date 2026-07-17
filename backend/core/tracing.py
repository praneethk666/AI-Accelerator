"""Shared Langfuse tracing helpers for grouping a whole request under ONE trace.

Problem this solves: backend/core/llm_client.py attaches a fresh
`langfuse.langchain.CallbackHandler()` to every LLM call. Langfuse v4 is
OTEL-based — when a CallbackHandler starts a run and there is NO currently
active OTEL span, it opens a brand new root trace for that single call. That's
why, before this module existed, every LLM call (agent tool-picking, the
search_documents answerer, query_planner, retrieval's hyp expansion, ...) in
one user turn showed up as its own disconnected trace in Langfuse Cloud —
there was no way to see "everything that happened for this one request".

The fix doesn't touch llm_client.py or vision_client.py at all. Per the
CallbackHandler's own root-context logic, if a span is already active in the
current OTEL context, the handler attaches its generation to THAT span instead
of starting a new trace. So we just need to open one root span per request
(`traced_request`, e.g. wrapping run_agent()'s whole tool-calling loop) and
every nested LLM call and tool invocation that happens inside that `with`
block — however deep the call stack — automatically nests under it as long as
it runs in the same thread/async task (contextvars propagate down the call
stack; that's the same mechanism backend/core/usage.py's token sink relies on).

`traced_tool` wraps each individual tool dispatch (tools_node's tool.run(...))
in its own child span, so tools that don't call an LLM themselves (list_documents,
sql_read) still show up as a distinct step in the trace, not just a gap.

Both are no-ops (yield None) when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't
set — same on/off switch already used by llm_client._langfuse_callbacks() and
vision_client._langfuse_enabled(), so tracing stays fully optional.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """Only "on" when both keys are set — see vision_client._langfuse_enabled()
    for why we gate this (unconfigured langfuse v4 otherwise starts an OTEL
    exporter against localhost:3001 and retries forever)."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _client():
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        logger.debug("langfuse client unavailable", exc_info=True)
        return None


@contextmanager
def traced_request(name: str, *, input: Any = None, metadata: dict | None = None):
    """Open ONE root span for an entire request (e.g. one /agent/chat turn).

    Everything invoked inside this `with` block — the agent's own tool-picking
    LLM calls, each dispatched tool (wrap those with `traced_tool` too), and any
    LLM calls those tools make internally (search_documents' query_planner /
    retrieval / answerer) — nests under this one trace, so Langfuse Cloud shows
    a single trace_id per request with every tool call as a child span.

    Yields a dict the caller can mutate: {"trace_id": str|None}. Stays None when
    Langfuse isn't configured — callers should treat that as "tracing is off" and
    simply not surface trace info, not as an error.
    """
    info: dict[str, Any] = {"trace_id": None}
    client = _client()
    if client is None:
        yield info
        return

    try:
        cm = client.start_as_current_observation(
            name=name, as_type="agent", input=input, metadata=metadata,
        )
        cm.__enter__()
    except Exception:
        logger.debug("failed to open langfuse root span %r", name, exc_info=True)
        yield info
        return

    try:
        info["trace_id"] = client.get_current_trace_id()
        yield info
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("failed to close langfuse root span %r", name, exc_info=True)


@contextmanager
def traced_tool(name: str, *, input: Any = None):
    """Child span for ONE tool dispatch, nested under the active `traced_request`
    (no-op, standalone span, if there isn't one — still fine, just unparented).
    Mutate the yielded box's "output" key before the block ends to record the
    tool's result on the span."""
    box: dict[str, Any] = {"output": None}
    client = _client()
    if client is None:
        yield box
        return

    try:
        cm = client.start_as_current_observation(name=name, as_type="tool", input=input)
        span = cm.__enter__()
    except Exception:
        logger.debug("failed to open langfuse tool span %r", name, exc_info=True)
        yield box
        return

    try:
        yield box
    finally:
        try:
            if span is not None:
                span.update(output=box["output"])
        except Exception:
            logger.debug("failed to record output on tool span %r", name, exc_info=True)
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("failed to close langfuse tool span %r", name, exc_info=True)