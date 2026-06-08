"""Config-driven LangGraph pipeline.

- nodes = tools found in the registry by the names in config.steps
- a step runs only if it is BOTH listed in config.steps AND registered (toggle)
- the `extract` placeholder expands to the concrete extractors (pdf/excel/ppt/
  image); at runtime only the one matching state["file_type"] runs
- tools never call each other — they read/write the shared state; the graph routes
- one tool failing -> appended to state["errors"], never kills the run
"""

from __future__ import annotations
from langgraph.graph import END, START, StateGraph
from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState, Tool

# Placeholder step in a route's `steps`; resolved to a file-type-specific extractor.
EXTRACT_PLACEHOLDER = "extract"


def _make_node(tool: Tool, raw_config: dict):
    # wrap a Tool as a graph node + graceful failure; config bound in via closure
    def node(state: PipelineState) -> PipelineState:
        state.setdefault("errors", [])
        try:
            return tool.run(state, raw_config)
        except Exception as exc:  # one tool must not kill the run
            state["errors"].append(f"{tool.name}: {exc}")
            return state

    return node


def _resolve_steps(config: PipelineConfig, registry: ToolRegistry):
    """Expand `extract` into registered extractors; return (steps, extractor_ft).

    extractor_ft maps an expanded extractor's tool name -> the file_type that
    selects it. Only steps from expanding `extract` are file-type-gated; a
    concrete extractor listed directly in `steps` always runs (a deliberate choice).
    """
    extractor_map = config.raw.get("extractors", {})  # file_type -> tool name
    steps: list[str] = []
    extractor_ft: dict[str, str] = {}
    for s in config.steps:
        if s == EXTRACT_PLACEHOLDER:
            for file_type, tool_name in extractor_map.items():
                if tool_name in registry:  # toggle: only registered extractors
                    steps.append(tool_name)
                    extractor_ft[tool_name] = file_type
        elif s in registry:  # toggle: listed AND registered
            steps.append(s)
    return steps, extractor_ft


def build_pipeline(registry: ToolRegistry, config: PipelineConfig):
    """Compile a runnable graph from a registry + config. Returns graph.invoke-able."""
    steps, extractor_ft = _resolve_steps(config, registry)
    gates = config.raw.get("route_gates", {})  # {step: [routes that keep it]}

    def _enabled(step: str, state: PipelineState) -> bool:
        # expanded extractor: only the one matching the document's file_type runs
        ft = extractor_ft.get(step)
        if ft is not None:
            return state.get("file_type") == ft
        # route gate: gated step runs only for its allowed routes; ungated always runs
        allowed = gates.get(step)
        if allowed is None:
            return True
        return (state.get("route") or config.route) in allowed

    def _conditional(step: str) -> bool:
        # skippable -> a branch point precedes it (route gate or file-type dispatch)
        return step in gates or step in extractor_ft

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


def run_pipeline(tools, state: PipelineState, config) -> PipelineState:
    """Build + run in one call. Accepts dicts (legacy) or a registry/PipelineConfig."""
    registry = tools if isinstance(tools, ToolRegistry) else _registry_from_dict(tools)
    cfg = (
        config
        if isinstance(config, PipelineConfig)
        else PipelineConfig.from_dict(config)
    )
    graph = build_pipeline(registry, cfg)
    return graph.invoke(state)


def _registry_from_dict(tools: dict) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools.values():
        reg.register(tool)
    return reg
