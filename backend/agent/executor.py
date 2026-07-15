"""Agent-executor core — the loop that picks and calls agent tools.

Thin, hand-rolled tool-calling loop on ONE LangGraph graph (agent node <-> tools
node), using the provider's native tool-calling via `llm.bind_tools()`. Deliberately
NOT LangChain's AgentExecutor/create_tool_calling_agent — we already depend on
LangGraph for the ingestion/query pipelines, and keeping the write-approval gate in
our own dispatch code (not buried in a framework abstraction) is the whole point
(see PLAN.md Phase 2, locked decision #2).

Reads run immediately. Any tool listed in `query.agent.write_tools` (config) is
NOT executed on first ask — the loop stops and returns status="needs_approval"
with the pending call(s). The caller (API/CLI) shows the user what's about to run
and, if they say yes, re-invokes run_agent(..., approved_writes=True) with the SAME
message; the model re-proposes the same call and this time it executes. This is a
restart-based approval, not a mid-graph checkpoint/interrupt() — simpler to reason
about for v1; graduating to a real LangGraph interrupt()+checkpointer so approval
can resume mid-conversation (instead of replaying) is the natural next step once
this is proven out.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from backend.agent_tools import AgentTool, build_agent_registry
from backend.core.llm_client import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the assistant for a document intelligence system. Documents are ALREADY "
    "ingested and searchable before this conversation starts — search_documents queries "
    "an existing index, it does not need a file path. NEVER tell the user you need a "
    "file path, or that no file has been provided, in order to answer a question — "
    "that only applies to ingest_document, and only when they explicitly want to add a "
    "NEW file. For ANY question — including vague ones like 'this invoice' or 'the "
    "document' with no filename given — call search_documents first; do not assume "
    "nothing is ingested just because the message doesn't name a file.\n\n"
    "Tools available: ingest_document (ingest a NEW file the user gives a path for), "
    "search_documents (answer a question from already-ingested documents with "
    "citations; accepts an optional document_scope to restrict to specific document "
    "ids), list_documents (list what's already ingested: id, filename, document_type, "
    "industry, status), sql_read (read-only SQL against the configured database).\n\n"
    "Disambiguation: if a question references a document ambiguously (e.g. 'this "
    "invoice') and you're not sure which one is meant, call list_documents first. If "
    "there's exactly one ingested document, or exactly one obvious match, just use it "
    "(pass its id as document_scope to search_documents) without asking. If several "
    "documents could plausibly match, don't guess and don't silently search across all "
    "of them — reply in plain text listing the candidates (filename + document_type) "
    "and ask the user which one they mean before calling search_documents.\n\n"
    "Don't call a tool you don't need. Once you have enough information, answer "
    "directly in plain text — do not call a tool just to restate its result."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    pending_approval: list[dict] | None
    approved_writes: bool


def _to_openai_tool(tool: AgentTool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _invoke_with_retry(llm_with_tools, messages, attempts: int = 3):
    """Groq/Llama-3.3 tool-calling occasionally emits its raw text function-call
    syntax instead of a structured tool call, and the API rejects the whole turn
    with a 'tool_use_failed' 400 — a provider-side flake (reproduced independent of
    prompt/tool content), not a logic error here. One retry clears it in practice;
    anything else is a real error and must not be swallowed."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            msg = str(exc)
            if "tool_use_failed" not in msg and "Failed to call a function" not in msg:
                raise
            last_exc = exc
            logger.warning("agent LLM malformed tool call (attempt %d/%d): %s", attempt + 1, attempts, msg[:200])
    raise last_exc


def _build_graph(llm_with_tools, registry: dict[str, AgentTool], write_tools: set[str],
                  max_iterations: int):
    def agent_node(state: AgentState) -> dict:
        response = _invoke_with_retry(llm_with_tools, state["messages"])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        pending: list[dict] = []
        for call in getattr(last, "tool_calls", None) or []:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

            if name in write_tools and not state.get("approved_writes"):
                pending.append({"id": call_id, "name": name, "args": args})
                tool_messages.append(ToolMessage(
                    content="blocked: this action writes data and needs human "
                            "approval before it can run.",
                    tool_call_id=call_id,
                ))
                continue

            tool = registry.get(name)
            if tool is None:
                result: Any = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    result = tool.run(**args)
                except Exception as exc:  # a bad tool call must not kill the loop
                    logger.warning("agent tool %s failed: %s", name, exc)
                    result = {"error": str(exc)}
            tool_messages.append(ToolMessage(
                content=json.dumps(result, default=str), tool_call_id=call_id,
            ))

        return {"messages": tool_messages, "pending_approval": pending or None}

    def route_after_agent(state: AgentState) -> str:
        if state.get("iterations", 0) >= max_iterations:
            return END
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def route_after_tools(state: AgentState) -> str:
        return END if state.get("pending_approval") else "agent"

    sg = StateGraph(AgentState)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", tools_node)
    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    sg.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return sg.compile()


def _extract_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Pair each requested tool call with its result (if it ran), in call order."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            by_id[call["id"]] = {"id": call["id"], "name": call["name"], "args": call.get("args") or {}}
            order.append(call["id"])
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id in by_id:
            by_id[m.tool_call_id]["result"] = m.content
    return [by_id[i] for i in order]


def run_agent(
    message: str,
    *,
    config: dict,
    registry: dict[str, AgentTool] | None = None,
    llm=None,
    conversation_history: list[BaseMessage] | None = None,
    approved_writes: bool = False,
) -> dict:
    """Run one turn of the agent loop.

    Returns:
      {"status": "done", "answer": str, "tool_calls": [...], "messages": [...]}
      {"status": "needs_approval", "pending": [{"id","name","args"}], "tool_calls": [...],
       "answer": None, "messages": [...]}

    `messages` in the return is the growing LangChain message list — pass it back
    in as `conversation_history` (plus the next user message) to continue the
    conversation with memory of what was asked/called before.
    """
    registry = registry if registry is not None else build_agent_registry()
    agent_cfg = (config.get("query") or {}).get("agent") or {}
    max_iterations = agent_cfg.get("max_iterations", 5)
    write_tools = set(agent_cfg.get("write_tools") or [])

    if llm is None:
        llm_cfg = {**config.get("llm", {})}
        if agent_cfg.get("provider"):
            llm_cfg["provider"] = agent_cfg["provider"]
        if agent_cfg.get("model"):
            llm_cfg["model"] = agent_cfg["model"]
        if agent_cfg.get("base_url"):
            llm_cfg["base_url"] = agent_cfg["base_url"]
        # agent-specific api_key (e.g. ${DEEPSEEK_API_KEY}) takes priority;
        # otherwise null out the main LLM's key if providers differ.
        if agent_cfg.get("api_key"):
            llm_cfg["api_key"] = agent_cfg["api_key"]
        elif agent_cfg.get("provider") and agent_cfg["provider"] != config.get("llm", {}).get("provider"):
            llm_cfg["api_key"] = None
        llm = get_llm({**config, "llm": llm_cfg})

    tool_schemas = [_to_openai_tool(t) for t in registry.values()]
    llm_with_tools = llm.bind_tools(tool_schemas)

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    messages += conversation_history or []
    messages.append(HumanMessage(message))

    graph = _build_graph(llm_with_tools, registry, write_tools, max_iterations)
    final_state = graph.invoke({
        "messages": messages,
        "iterations": 0,
        "pending_approval": None,
        "approved_writes": approved_writes,
    })

    tool_calls = _extract_tool_calls(final_state["messages"])
    if final_state.get("pending_approval"):
        return {
            "status": "needs_approval",
            "pending": final_state["pending_approval"],
            "tool_calls": tool_calls,
            "answer": None,
            "messages": final_state["messages"],
        }

    last = final_state["messages"][-1]
    answer = last.content if isinstance(last, AIMessage) else ""
    return {
        "status": "done",
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": final_state["messages"],
    }
