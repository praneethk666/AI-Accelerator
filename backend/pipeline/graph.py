"""Config-driven LangGraph pipeline.

- nodes = tools found in the registry by the names in config.steps
- a step runs only if it is BOTH listed in config.steps AND registered (toggle)
- tools never call each other — they read/write the shared state; the graph routes
- one tool failing -> appended to state["errors"], never kills the run
"""
from __future__ import annotations
from langgraph.graph import END, START, StateGraph
from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState, Tool


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


def _enabled(step: str, route: str, gates: dict) -> bool:
    # gated step runs only for its allowed routes; ungated always runs
    allowed = gates.get(step)
    return allowed is None or route in allowed


def build_pipeline(registry: ToolRegistry, config: PipelineConfig):
    """Compile a runnable graph from a registry + config. Returns graph.invoke-able."""
    steps = [s for s in config.steps if s in registry]  # toggle: listed AND registered
    gates = config.raw.get("route_gates", {})  # {step: [routes that keep it]}

    sg = StateGraph(PipelineState)
    for step in steps:
        sg.add_node(step, _make_node(registry.get(step), config.raw))

    # router: next enabled step after position i (or END). i=-1 means "from START".
    def next_from(i: int):
        def route(state: PipelineState) -> str:
            current_route = state.get("route") or config.route
            for s in steps[i + 1:]:
                if _enabled(s, current_route, gates):
                    return s
            return END
        return route

    # reachable targets from position i: walk forward, stop at the first UNGATED
    # step (it always runs, so nothing past it is reachable). END only if all the
    # rest are skippable. Keeps the graph a clean chain with branches only at gates.
    def dests_after(i: int) -> dict:
        d = {}
        for s in steps[i + 1:]:
            d[s] = s
            if s not in gates:  # ungated -> always runs, can't skip past it
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
    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig.from_dict(config)
    graph = build_pipeline(registry, cfg)
    return graph.invoke(state)


def _registry_from_dict(tools: dict) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools.values():
        reg.register(tool)
    return reg
