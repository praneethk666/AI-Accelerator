"""Agent executor — the tool-calling loop.

Takes a request, advertises the registered agent tools (backend/agent_tools.py)
to the model, lets the model pick, runs the calls, feeds results back, and
returns the final answer. Headless: no UI, no server — just a function.

    request -> model sees tools -> tool_calls? -> run tools -> results back
            -> model again ... -> no tool_calls -> final answer

Safety: read tools run directly; a tool marked `writes = True` must be approved
first via the `approve` callback. No callback provided => the write is refused
(with an explanatory tool result the model can react to), never silently run.

The model comes from config (agent.llm block, falling back to the global llm
block) so dev runs free Groq and prod flips to gpt-4o-mini by config only.
Tests inject a mock via the `llm` param — no network needed.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from backend.agent_tools import build_agent_registry
from backend.core.llm_client import get_llm

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a document-intelligence assistant. Use the available tools to "
    "fulfill the user's request. Call tools when needed; when you have enough "
    "information, reply with a concise final answer. If a tool call is refused "
    "or fails, explain what happened instead of retrying it blindly. "
    "Tool arguments must match the JSON schema exactly — omit optional "
    "parameters entirely unless you have a concrete value of the right type."
)


def _tool_spec(tool) -> dict:
    """AgentTool -> OpenAI-format tool spec (what bind_tools advertises)."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _agent_llm(config: dict):
    """Model from the agent.llm block, falling back to the global llm block.

    Dedicated block => the agent's provider/model swap independently of the
    pipeline's (dev: free Groq; prod: point it at gpt-4o-mini). get_llm reads
    config["llm"], so substitute the agent block in.
    """
    llm_cfg = (config.get("agent") or {}).get("llm") or config["llm"]
    return get_llm({**config, "llm": llm_cfg})


def _dispatch(tool, args: dict, approve) -> tuple[dict, bool]:
    """Run one tool call, enforcing the write-approval gate.

    Returns (result_payload, ran). Refusals/errors come back as payloads (not
    raises) so they land in the transcript and the model can react.
    """
    if getattr(tool, "writes", False):
        if approve is None:
            return {"error": f"{tool.name} writes data and no approval mechanism "
                             "is available; the call was refused."}, False
        if not approve(tool.name, args):
            return {"error": f"{tool.name} was not approved by the user."}, False
    try:
        return tool.run(**args), True
    except TypeError as exc:  # bad/missing args from the model
        return {"error": f"invalid arguments for {tool.name}: {exc}"}, False
    except Exception as exc:  # tool failed — surface it, don't kill the loop
        logger.exception("agent tool %s failed", tool.name)
        return {"error": f"{tool.name} failed: {exc}"}, False


def run_agent(
    request: str,
    *,
    registry: dict | None = None,
    config: dict | None = None,
    llm=None,
    approve=None,
    max_iterations: int | None = None,
) -> dict:
    """Run the tool-calling loop for one request.

    Args:
        request: the user's request in plain language.
        registry: {name: AgentTool}; defaults to build_agent_registry().
        config: loaded config dict (model + max_iterations come from it).
        llm: pre-built chat model (tests inject a mock); default from config.
        approve: callback(tool_name, args) -> bool for tools marked writes=True.
            None => write calls are refused (safe headless default).
        max_iterations: cap on model turns; default query.agent.max_iterations.

    Returns:
        {"answer": str, "tool_calls": [{tool, args, ok, result}], "iterations": int,
         "stopped": "final" | "max_iterations"}
    """
    if config is None and llm is None:
        from backend.core.config import load_config
        import os
        config = load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))

    tools = registry if registry is not None else build_agent_registry()
    model = llm if llm is not None else _agent_llm(config)
    if max_iterations is None:
        max_iterations = ((config or {}).get("query", {}).get("agent", {})
                          .get("max_iterations", DEFAULT_MAX_ITERATIONS))

    bound = model.bind_tools([_tool_spec(t) for t in tools.values()])
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(request)]
    trace: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        try:
            response = bound.invoke(messages)
        except Exception as exc:
            # Some providers (e.g. Groq) validate tool calls SERVER-side and 400
            # the whole request when the model emits malformed args. Feed the
            # validation error back so the model can correct itself next turn.
            if "tool_use_failed" in str(exc) or "tool call validation failed" in str(exc):
                logger.warning("model emitted an invalid tool call; retrying: %s", exc)
                messages.append(HumanMessage(
                    "Your last tool call was rejected by the API because its "
                    f"arguments did not match the tool's JSON schema: {exc}. "
                    "Retry with arguments that match the schema exactly; omit "
                    "optional parameters unless you have a valid value."))
                continue
            raise
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:  # no more tools wanted -> final answer
            return {"answer": response.content or "", "tool_calls": trace,
                    "iterations": iteration, "stopped": "final"}

        for call in tool_calls:
            name, args = call["name"], call.get("args") or {}
            tool = tools.get(name)
            if tool is None:  # model hallucinated a tool name
                result, ok = {"error": f"unknown tool: {name}"}, False
            else:
                result, ok = _dispatch(tool, args, approve)
            trace.append({"tool": name, "args": args, "ok": ok, "result": result})
            messages.append(ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=call.get("id") or name,
            ))

    logger.warning("agent hit max_iterations=%s without a final answer", max_iterations)
    return {"answer": "", "tool_calls": trace,
            "iterations": max_iterations, "stopped": "max_iterations"}
