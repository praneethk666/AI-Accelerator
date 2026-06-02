"""Pipeline runner.

STUB: runs registered tools in the order given by config["steps"].

TODO (Karthii): replace this sequential stub with a real LangGraph graph —
nodes = tools, edges built from config, and a CONDITIONAL edge on
state["route"] (set by the categorize tool). Keep the Tool interface in
core.tool unchanged so nobody is blocked while you swap the engine.
"""
from __future__ import annotations
from backend.core.tool import PipelineState


def run_pipeline(tools: dict, state: PipelineState, config: dict) -> PipelineState:
    state.setdefault("errors", [])
    for step in config.get("steps", []):
        tool = tools.get(step)
        if tool is None:
            state["errors"].append(f"no tool registered for step '{step}'")
            continue
        try:
            state = tool.run(state, config)
        except Exception as exc:  # fail gracefully — one tool must not kill the run
            state["errors"].append(f"{step}: {exc}")
    return state
