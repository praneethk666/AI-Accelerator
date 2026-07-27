"""backend/core/tracing.py — OpenTelemetry traces + metrics + logs, exported to
Grafana Cloud (Tempo / Prometheus / Loki).

Public API unchanged for existing callers: traced_request(), traced_tool(),
record_handled_error(). New in this version:

  record_llm_usage(component, input_tokens, output_tokens, model=, provider=)
      Call this from backend/core/usage.py's record()/record_from_message()
      to attach token counts to the CURRENT span and increment the
      llm_tokens_total metric.

  Session tagging is automatic: traced_request(..., metadata={"session_id": x})
  stores it in a contextvar for the life of the request, and traced_tool
  stamps every child span with session_id too — every span for one chat/query
  turn becomes filterable by session_id in Tempo with zero other call-site
  changes.

  Every tool_call also emits a tool_calls_total counter and a
  tool_call_duration_ms histogram (both tagged tool.name + status), which is
  what makes Grafana dashboards (not just single-trace inspection) possible.

  Root requests also emit request_calls_total and request_duration_ms, tagged
  by request.name + status, so Grafana can show request volume and latency by
  agent turn / query / ingest path.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import time
from typing import Any, Iterator

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

_INITIALIZED = False
_REQUEST_CALLS = None
_REQUEST_DURATION = None
_TOOL_CALLS = None
_TOOL_DURATION = None
_LLM_TOKENS = None

# Set once per request (traced_request) and read by every nested traced_tool,
# so session_id lands on every span in the turn without threading it through
# every call site by hand.
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "otel_session_id", default=None
)

_request_name_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "otel_request_name", default=None
)

def _resource() -> Resource:
    return Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "ai-accelerator"),
        "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "backend"),
    })


def _init() -> None:
    global _INITIALIZED, _REQUEST_CALLS, _REQUEST_DURATION, _TOOL_CALLS, _TOOL_DURATION, _LLM_TOKENS
    if _INITIALIZED:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        # No Grafana Cloud config -> everything below stays a no-op, same as before.
        _INITIALIZED = True
        return

    resource = _resource()

    # ---- traces ----
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tp)

    # ---- metrics ----
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=15000)
    mp = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(mp)
    meter = metrics.get_meter("ai-accelerator")
    _REQUEST_CALLS = meter.create_counter(
        "request_calls_total", description="Root request count", unit="1")
    _REQUEST_DURATION = meter.create_histogram(
        "request_duration_ms", description="Root request duration", unit="ms")
    _TOOL_CALLS = meter.create_counter(
        "tool_calls_total", description="Tool/step dispatch count", unit="1")
    _TOOL_DURATION = meter.create_histogram(
        "tool_call_duration_ms", description="Tool/step duration", unit="ms")
    _LLM_TOKENS = meter.create_counter(
        "llm_tokens_total", description="LLM/vision tokens consumed", unit="1")

    # ---- logs (correlated to traces automatically via the active span context) ----
    try:
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry import _logs as otel_logs_api

        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        otel_logs_api.set_logger_provider(lp)
        handler = LoggingHandler(level=logging.INFO, logger_provider=lp)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)  # ADDS alongside your existing handlers
    except Exception:
        logger.debug(
            "OTLP log export unavailable (check opentelemetry-exporter-otlp-proto-http "
            "version) — traces/metrics still work fine without it.",
            exc_info=True,
        )

    # ---- LangChain LLM call auto-instrumentation ----
    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor
        LangchainInstrumentor().instrument()
    except Exception:
        logger.debug("opentelemetry-instrumentation-langchain not installed; "
                      "LLM calls still nest under their parent span, just without "
                      "their own dedicated span.")

    _INITIALIZED = True


def _tracer():
    _init()
    return trace.get_tracer("ai-accelerator")


def _summarize(value: Any, limit: int = 2000) -> str:
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class _SpanHandle(dict):
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
def traced_request(name: str, *, input: Any = None, metadata: dict | None = None) -> Iterator[dict]:
    tracer = _tracer()
    metadata = metadata or {}
    session_id = metadata.get("session_id")
    current_span = trace.get_current_span()
    parent_ctx = current_span.get_span_context() if current_span is not None else None
    is_root = not parent_ctx or not parent_ctx.is_valid
    start = time.perf_counter()
    status = "ok"
    token = _session_id_var.set(session_id) if session_id else None
    req_token = _request_name_var.set(name)
    try:
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("input_summary", _summarize(input))
            if session_id:
                span.set_attribute("session_id", session_id)
            for k, v in metadata.items():
                span.set_attribute(f"metadata.{k}", _summarize(v, 200))

            ctx = span.get_span_context()
            trace_id_hex = format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None
            trace_info = {"trace_id": trace_id_hex}
            logger.info("request %s starting, trace_id=%s", name, trace_id_hex)
            try:
                yield trace_info
            except Exception as exc:
                status = "error"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
    finally:
        if is_root and _REQUEST_CALLS is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _REQUEST_CALLS.add(1, {"request.name": name, "status": status})
            _REQUEST_DURATION.record(elapsed_ms, {"request.name": name, "status": status})
        if token is not None:
            _session_id_var.reset(token)
            _request_name_var.reset(req_token)


@contextlib.contextmanager
def traced_tool(name: str, *, input: Any = None) -> Iterator[_SpanHandle]:
    tracer = _tracer()
    start = time.perf_counter()
    tool_name = name.split(":", 1)[-1] if ":" in name else name
    status = "ok"
    with tracer.start_as_current_span(name) as otel_span:
        otel_span.set_attribute("tool.name", tool_name)
        otel_span.set_attribute("tool.input_summary", _summarize(input))
        session_id = _session_id_var.get()
        if session_id:
            otel_span.set_attribute("session_id", session_id)
        request_name = _request_name_var.get()
        if request_name:
            otel_span.set_attribute("request_name", request_name)

        handle = _SpanHandle(otel_span)
        try:
            yield handle
        except Exception as exc:
            status = "error"
            otel_span.record_exception(exc)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            if "error" in handle:
                status = "error"
                otel_span.set_status(Status(StatusCode.ERROR, str(handle["error"])))
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if _TOOL_CALLS is not None:
                attrs = {"tool.name": tool_name, "status": status}
                if request_name:
                    attrs["request_name"] = request_name
                _TOOL_CALLS.add(1, attrs)
                _TOOL_DURATION.record(elapsed_ms, attrs)


def record_llm_usage(component: str, input_tokens: int, output_tokens: int,
                      *, model: str | None = None, provider: str | None = None) -> None:
    """Call from usage.py's record()/record_from_message() (or directly from a
    call site) to attach token counts to the CURRENT span and the
    llm_tokens_total metric. No-op if tracing isn't configured."""
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        if model:
            span.set_attribute("model", model)
        if provider:
            span.set_attribute("provider", provider)
    if _LLM_TOKENS is not None:
        attrs = {"component": component}
        if model:
            attrs["model"] = model
        _LLM_TOKENS.add(input_tokens, {**attrs, "token_type": "input"})
        _LLM_TOKENS.add(output_tokens, {**attrs, "token_type": "output"})


def record_handled_error(error_type: str, message: str, **attrs: Any) -> None:
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
