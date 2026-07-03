"""Per-run LLM/vision token accounting.

A pipeline run opens a sink (`using_sink()`); the model-call helpers record token
usage into it via `record(kind, input, output)`. The sink lives in a ContextVar so
it's naturally visible to the graph nodes (same thread), and the vision/enrichment
tools propagate it into their worker threads by wrapping submitted tasks with
`contextvars.copy_context().run(...)` (see `copy_ctx`).

Kinds we record: "categorize" (cover-page vision), "vision" (image captions),
"enrichment" (chunk summaries), "answer" (chat — query side). Totals are attached
to the pipeline state as state["token_usage"] and surfaced in the UI.
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

    def add(self, kind: str, input_tokens, output_tokens) -> None:
        with self._lock:
            k = self.by_kind.setdefault(
                kind, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            k["calls"] += 1
            k["input_tokens"] += int(input_tokens or 0)
            k["output_tokens"] += int(output_tokens or 0)

    def totals(self) -> dict:
        with self._lock:
            ti = sum(v["input_tokens"] for v in self.by_kind.values())
            to = sum(v["output_tokens"] for v in self.by_kind.values())
            calls = sum(v["calls"] for v in self.by_kind.values())
            return {
                "calls": calls,
                "input_tokens": ti,
                "output_tokens": to,
                "total_tokens": ti + to,
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


def record(kind: str, input_tokens=0, output_tokens=0) -> None:
    """Record one model call's token usage into the active sink (no-op if none)."""
    sink = _SINK.get()
    if sink is not None:
        sink.add(kind, input_tokens, output_tokens)


def copy_ctx():
    """Snapshot the current context (incl. the sink) to run worker-thread tasks in,
    so usage recorded off-thread still lands in this run's sink. Use as:
        ctx = copy_ctx(); executor.submit(ctx.run, fn, *args)"""
    return contextvars.copy_context()
