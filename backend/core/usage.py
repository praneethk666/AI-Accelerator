"""Per-run LLM/vision token accounting.

A pipeline run opens a sink (`using_sink()`); the model-call helpers record token
usage into it via `record(kind, input, output)` — or, more conveniently,
`record_from_message(kind, ai_message)` which pulls everything off a LangChain
AIMessage's `usage_metadata` automatically. The sink lives in a ContextVar so
it's naturally visible to the graph nodes (same thread), and the vision/enrichment
tools propagate it into their worker threads by wrapping submitted tasks with
`contextvars.copy_context().run(...)` (see `copy_ctx`).

Kinds we record: "categorize" (cover-page vision), "vision" (image captions),
"enrichment" (chunk summaries), "agent" (agent tool-picking), "query_planner",
"hyde", "answer" (chat — query side). Totals are attached to the pipeline state
as state["token_usage"] and surfaced in the UI.

Fields in totals():
  input_tokens / output_tokens — summed across every call in the run.
  reasoning_tokens             — summed "thinking"/reasoning tokens, when the
                                  provider reports them (output_token_details.
                                  reasoning — e.g. Claude extended thinking,
                                  OpenAI o-series). These are a BREAKDOWN of
                                  output_tokens, not additional tokens, so they
                                  aren't added again into total_tokens.
  total_tokens                 — input_tokens + output_tokens.
  context_tokens                — the LARGEST single call's input_tokens seen in
                                  this run: i.e. how much context (conversation
                                  history + tool results) was actually sent to
                                  the model at its fullest point this turn. NOT a
                                  fraction of some model's max context window —
                                  no provider returns that per-call, so we don't
                                  claim one.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from contextvars import ContextVar

_SINK: ContextVar = ContextVar("token_usage_sink", default=None)


class _Sink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_kind: dict[str, dict] = {}
        self._context_tokens = 0  # largest single call's input_tokens this run

    def add(self, kind: str, input_tokens=0, output_tokens=0,
             reasoning_tokens=0, context_tokens=None) -> None:
        with self._lock:
            k = self.by_kind.setdefault(kind, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
            })
            it = int(input_tokens or 0)
            ot = int(output_tokens or 0)
            rt = int(reasoning_tokens or 0)
            k["calls"] += 1
            k["input_tokens"] += it
            k["output_tokens"] += ot
            k["reasoning_tokens"] += rt

            ctx = int(context_tokens if context_tokens is not None else it)
            if ctx > self._context_tokens:
                self._context_tokens = ctx

    def totals(self) -> dict:
        with self._lock:
            ti = sum(v["input_tokens"] for v in self.by_kind.values())
            to = sum(v["output_tokens"] for v in self.by_kind.values())
            tr = sum(v["reasoning_tokens"] for v in self.by_kind.values())
            calls = sum(v["calls"] for v in self.by_kind.values())
            return {
                "calls": calls,
                "input_tokens": ti,
                "output_tokens": to,
                "reasoning_tokens": tr,
                "total_tokens": ti + to,
                "context_tokens": self._context_tokens,
                "by_kind": {k: dict(v) for k, v in self.by_kind.items()},
            }


@contextmanager
def using_sink():
    """Scope a fresh usage sink for one pipeline run."""
    sink = _Sink()
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def record(kind: str, input_tokens=0, output_tokens=0,
           reasoning_tokens=0, context_tokens=None) -> None:
    """Record one model call's token usage into the active sink (no-op if none)."""
    sink = _SINK.get()
    if sink is not None:
        sink.add(kind, input_tokens, output_tokens, reasoning_tokens, context_tokens)


def record_from_message(kind: str, message) -> None:
    """Convenience: pull `usage_metadata` off a LangChain AIMessage and record it.
    Safe to call even when the provider didn't return usage info (records zeros).
    `context_tokens` is set to that call's own input_tokens — the size of the
    context actually sent for THIS call — so the sink can track the largest one
    across the whole run."""
    um = getattr(message, "usage_metadata", None) or {}
    input_tokens = um.get("input_tokens", 0) or 0
    output_tokens = um.get("output_tokens", 0) or 0
    reasoning_tokens = (um.get("output_token_details") or {}).get("reasoning", 0) or 0
    record(kind, input_tokens, output_tokens,
           reasoning_tokens=reasoning_tokens, context_tokens=input_tokens)


def copy_ctx():
    """Snapshot the current context (incl. the sink) to run worker-thread tasks in,
    so usage recorded off-thread still lands in this run's sink. Use as:
        ctx = copy_ctx(); executor.submit(ctx.run, fn, *args)"""
    return contextvars.copy_context()