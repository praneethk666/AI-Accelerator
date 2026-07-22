"""backend/core/tracing.py — OpenTelemetry tracing, exported to Grafana Cloud Tempo.

Public API is unchanged from the Langfuse version so callers don't need to
change their call sites:

    with traced_request(name, input=..., metadata=...) as trace_info:
        trace_info["trace_id"]          # str, always available

    with traced_tool(name, input=...) as span:
        span["output"] = result          # optional, becomes a span attribute

New: record_handled_error(...) — call this at any point where an exception
is caught and appended to state["errors"] instead of raised, so it still
shows up in Grafana even though the pipeline swallows it and continues.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

_INITIALIZED = False


def _init_tracer_provider() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        # No Grafana Cloud config present -> tracing is a no-op, mirroring the
        # old "no-op unless LANGFUSE_* env vars are set" behavior.
        _INITIALIZED = True
        return

    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "ai-accelerator"),
        "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "backend"),
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter()  # reads endpoint/headers from OTEL_EXPORTER_OTLP_* env vars
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument every LangChain model call (agent tool-picking, answerer,
    # query_planner, retrieval's hyp expansion, ...) so each becomes its own
    # child span of whatever span is currently active — this is the OTel
    # replacement for llm_client.py's old per-call Langfuse CallbackHandler.
    # Optional dependency: pip install opentelemetry-instrumentation-langchain.
    # Guarded the same way the old _langfuse_callbacks() guarded its import, so
    # its absence never breaks a call, just skips the per-LLM-call span level.
    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor
        LangchainInstrumentor().instrument()
    except Exception:
        logger.debug("opentelemetry-instrumentation-langchain not installed; "
                      "LLM calls will still be captured under their parent "
                      "tool/step span, just without their own dedicated span.")

    _INITIALIZED = True


def _tracer():
    _init_tracer_provider()
    return trace.get_tracer("ai-accelerator")


def _summarize(value: Any, limit: int = 2000) -> str:
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class _SpanHandle(dict):
    """Dict-like wrapper so call sites can do span['output'] = result, matching
    the old Langfuse span object's interface."""

    def __init__(self, otel_span):
        super().__init__()
        self._otel_span = otel_span

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "output":
            self._otel_span.set_attribute("tool.output_summary", _summarize(value))
        elif key == "error":
            self._otel_span.set_attribute("error.message", _summarize(value))


@contextlib.contextmanager
def traced_request(
    name: str,
    *,
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[dict]:
    """Root span for one full request/turn (agent chat, ingest, query)."""
    tracer = _tracer()
    metadata = metadata or {}
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("input_summary", _summarize(input))
        for k, v in metadata.items():
            span.set_attribute(f"metadata.{k}", _summarize(v, 200))

        ctx = span.get_span_context()
        trace_id_hex = format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None
        trace_info = {"trace_id": trace_id_hex}

        try:
            yield trace_info
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


@contextlib.contextmanager
def traced_tool(name: str, *, input: Any = None) -> Iterator[_SpanHandle]:
    """Child span for one tool/step/LLM-call dispatch."""
    tracer = _tracer()
    with tracer.start_as_current_span(name) as otel_span:
        tool_name = name.split(":", 1)[-1] if ":" in name else name
        otel_span.set_attribute("tool.name", tool_name)
        otel_span.set_attribute("tool.input_summary", _summarize(input))

        handle = _SpanHandle(otel_span)
        try:
            yield handle
        except Exception as exc:
            otel_span.record_exception(exc)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            if "error" in handle:
                otel_span.set_status(Status(StatusCode.ERROR, str(handle["error"])))


def record_handled_error(error_type: str, message: str, **attrs: Any) -> None:
    """Call at the point a caught exception is appended to state['errors']
    instead of raised (graph.py steps, answerer.py, classifier.py). Marks the
    CURRENT active span as errored so handled failures are visible in Grafana
    even though the pipeline continues past them."""
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        logger.warning("record_handled_error with no active span: %s: %s", error_type, message)
        return
    span.set_attribute("error.type", error_type)
    span.set_attribute("error.message", _summarize(message, 500))
    for k, v in attrs.items():
        span.set_attribute(k, _summarize(v, 200))
    span.add_event("handled_error", {"error.type": error_type, "error.message": _summarize(message, 500)})
    span.set_status(Status(StatusCode.ERROR, message))