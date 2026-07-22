"""Config-driven LangGraph pipeline.

- nodes = tools found in the registry by the names in config.steps
- a step runs only if it is BOTH listed in config.steps AND registered (toggle)
- the `extract` placeholder expands to the concrete extractors; at runtime exactly
  one runs, chosen by (in priority order):
    1. route_extractors  — a route overrides extraction entirely (e.g. cad_route
       -> cad_extract uses a vision prompt instead of the normal pdf extractor)
    2. pdf_extractors     — file_type=="pdf" dispatches by state["pdf_kind"]
       (digital/scanned/mixed), set by the categorize step's detector
    3. extractors         — file_type -> tool for the simple 1:1 cases (excel, ppt)
- tools never call each other — they read/write the shared state; the graph routes
- one tool failing -> appended to state["errors"], never kills the run
"""

from __future__ import annotations
import logging
import time
from langgraph.graph import END, START, StateGraph
from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState, Tool
from backend.core.tracing import traced_tool, record_handled_error

logger = logging.getLogger(__name__)

# Placeholder step in a route's `steps`; resolved to a file-type-specific extractor.
EXTRACT_PLACEHOLDER = "extract"


def _make_node(tool: Tool, raw_config: dict):
    # wrap a Tool as a graph node: graceful failure + per-step metrics. config
    # bound in via closure.
    #
    # Each step also gets its own OTel child span (traced_tool), nested under
    # the single root span ingest_document() opens for the whole run (see
    # backend/pipeline/ingest.py). That's what lets one Grafana trace show every
    # ingestion step — categorize, extract, chunk, enrich, embed, index, ... — as
    # its own child, with any LLM/vision calls a step makes internally (they use
    # get_llm()/vision_client, which attach via the currently-active OTEL context)
    # nesting one level deeper still. Same mechanism the agent executor's
    # tools_node uses for query-side tools — see tracing.py's module docstring.
    def node(state: PipelineState) -> PipelineState:
        state.setdefault("errors", [])
        start = time.perf_counter()
        status, error = "ok", None
        trace_input = {
            "document_id": state.get("document_id"),
            "file_type": state.get("file_type"),
            "route": state.get("route"),
        }
        with traced_tool(f"step:{tool.name}", input=trace_input) as span:
            try:
                result = tool.run(state, raw_config)
            except Exception as exc:  # one tool must not kill the run
                status, error = "error", str(exc)
                state["errors"].append(f"{tool.name}: {exc}")
                result = state
                record_handled_error(
                    "step_failure", str(exc), **{"tool.name": tool.name}
                )

            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            entry = {"step": tool.name, "ms": elapsed_ms, "status": status}
            if error:
                entry["error"] = error
            # a tool may attach a per-step decision report (e.g. docling_pdf's
            # extraction routing) — surface it on the step metric so it's
            # persisted + visible, and record it on the span too.
            if isinstance(result, dict) and result.get("extraction_report") is not None:
                entry["report"] = result["extraction_report"]
            span["output"] = entry

        logger.info("step %s %s %.1fms", tool.name, status, elapsed_ms)

        # Return the tool's updates, but own the `metrics` channel: strip any
        # inherited metrics from the update and contribute exactly one entry
        # (the add-reducer on PipelineState.metrics appends it).
        update = dict(result) if isinstance(result, dict) else {}
        update.pop("metrics", None)
        update["metrics"] = [entry]
        return update

    return node


def _resolve_steps(config: PipelineConfig, registry: ToolRegistry):
    """Expand `extract` into registered extractors; return (steps, dispatch).

    dispatch holds the three gate maps that decide which single extractor runs:
      extractor_ft       tool -> file_type            (excel/ppt 1:1)
      pdf_kind_of        tool -> pdf kind             (digital/scanned/mixed)
      route_extractor_of tool -> {routes}             (route override, e.g. CAD)
      override_routes    set of routes that have a route_extractor — on those
                         routes the normal file_type/pdf extractors are suppressed
    A concrete extractor listed directly in `steps` (not via `extract`) always runs.
    """
    extractor_map = config.raw.get("extractors", {})        # file_type -> tool
    pdf_extractors = config.raw.get("pdf_extractors", {})    # pdf kind  -> tool
    route_extractors = config.raw.get("route_extractors", {})  # route   -> tool

    steps: list[str] = []
    extractor_ft: dict[str, str] = {}
    pdf_kind_of: dict[str, set[str]] = {}   # tool -> {pdf kinds it serves}
    route_extractor_of: dict[str, set[str]] = {}

    for s in config.steps:
        if s == EXTRACT_PLACEHOLDER:
            # 1. route-override extractors (a tool may serve several routes)
            for route, tool_name in route_extractors.items():
                if tool_name in registry:
                    if tool_name not in route_extractor_of:
                        steps.append(tool_name)
                        route_extractor_of[tool_name] = set()
                    route_extractor_of[tool_name].add(route)
            # 2. file_type extractors (excel, ppt, ...)
            for file_type, tool_name in extractor_map.items():
                if tool_name in registry:
                    steps.append(tool_name)
                    extractor_ft[tool_name] = file_type
            # 3. pdf sub-kind extractors (digital/scanned/mixed). One tool may serve
            # several kinds (e.g. docling_pdf handles all three) — add the node ONCE
            # and collect every kind it serves, so it runs for any of them.
            for kind, tool_name in pdf_extractors.items():
                if tool_name in registry:
                    if tool_name not in pdf_kind_of:
                        steps.append(tool_name)
                        pdf_kind_of[tool_name] = set()
                    pdf_kind_of[tool_name].add(kind)
        elif s in registry:  # toggle: listed AND registered
            steps.append(s)

    dispatch = {
        "extractor_ft": extractor_ft,
        "pdf_kind_of": pdf_kind_of,
        "route_extractor_of": route_extractor_of,
        "override_routes": set(route_extractors.keys()),
    }
    return steps, dispatch


def build_pipeline(registry: ToolRegistry, config: PipelineConfig):
    """Compile a runnable graph from a registry + config. Returns graph.invoke-able."""
    steps, dispatch = _resolve_steps(config, registry)
    extractor_ft = dispatch["extractor_ft"]
    pdf_kind_of = dispatch["pdf_kind_of"]
    route_extractor_of = dispatch["route_extractor_of"]
    override_routes = dispatch["override_routes"]
    gates = config.raw.get("route_gates", {})  # {step: [routes that keep it]}

    def _is_extractor(step: str) -> bool:
        return (
            step in extractor_ft
            or step in pdf_kind_of
            or step in route_extractor_of
        )

    def _enabled(step: str, state: PipelineState) -> bool:
        route = state.get("route") or config.route

        # route-override extractor: runs only on its route(s)
        routes = route_extractor_of.get(step)
        if routes is not None:
            return route in routes

        # file_type extractor: matches file_type, unless this route is overridden
        ft = extractor_ft.get(step)
        if ft is not None:
            if route in override_routes:
                return False
            return state.get("file_type") == ft

        # pdf sub-kind extractor: file_type==pdf AND detected kind is one this tool serves
        kinds = pdf_kind_of.get(step)
        if kinds is not None:
            if route in override_routes:
                return False
            return state.get("file_type") == "pdf" and state.get("pdf_kind") in kinds

        # route gate: gated step runs only for its allowed routes; ungated always runs
        allowed = gates.get(step)
        if allowed is None:
            return True
        return route in allowed

    def _conditional(step: str) -> bool:
        # skippable -> a branch point precedes it (route gate or extractor dispatch)
        return step in gates or _is_extractor(step)

    sg = StateGraph(PipelineState)
    for step in steps:
        sg.add_node(step, _make_node(registry.get(step), config.raw))

    # router: next enabled step after position i (or END). i=-1 means "from START".
    def next_from(i: int):
        def route(state: PipelineState) -> str:
            for s in steps[i + 1 :]:
                if _enabled(s, state):
                    return s
            return END

        return route

    # reachable targets from position i: walk forward, stop at the first
    # UNCONDITIONAL step (it always runs, so nothing past it is reachable). END
    # only if all the rest are skippable. Keeps the graph a clean chain with
    # branches only where a gate or file-type dispatch can skip a step.
    def dests_after(i: int) -> dict:
        d = {}
        for s in steps[i + 1 :]:
            d[s] = s
            if not _conditional(s):  # unconditional -> always runs, can't skip past it
                return d
        d[END] = END
        return d

    if not steps:
        sg.add_edge(START, END)
    else:
        sg.add_conditional_edges(START, next_from(-1), dests_after(-1))
        for i, step in enumerate(steps):
            sg.add_conditional_edges(step, next_from(i), dests_after(i))

    return sg.compile()


def run_pipeline(tools, state: PipelineState, config, on_step=None) -> PipelineState:
    """Build + run in one call. Accepts dicts (legacy) or a registry/PipelineConfig.

    on_step: optional callback invoked as on_step(entry, snapshot) after EACH step
    completes, where entry is the step's {step, ms, status} metric. Lets the API
    report live progress while ingestion runs (the UI polls it). When omitted the
    graph is invoked normally (no streaming overhead)."""
    registry = tools if isinstance(tools, ToolRegistry) else _registry_from_dict(tools)
    cfg = (
        config
        if isinstance(config, PipelineConfig)
        else PipelineConfig.from_dict(config)
    )
    graph = build_pipeline(registry, cfg)

    from backend.core import usage  # per-run LLM/vision token accounting

    if on_step is None:
        with usage.using_sink() as sink:
            final = graph.invoke(state)
        if isinstance(final, dict):
            final["token_usage"] = sink.totals()
        return final

    # stream_mode="values" yields the full accumulated state after each step;
    # the metrics channel grows by one entry per step, so emit the new ones.
    final = state
    seen = 0
    with usage.using_sink() as sink:
        for snapshot in graph.stream(state, stream_mode="values"):
            final = snapshot
            metrics = snapshot.get("metrics", []) or []
            while seen < len(metrics):
                try:
                    on_step(metrics[seen], snapshot)
                except Exception:  # progress reporting must never break ingestion
                    pass
                seen += 1
    if isinstance(final, dict):
        final["token_usage"] = sink.totals()
    return final


def _registry_from_dict(tools: dict) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools.values():
        reg.register(tool)
    return reg