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
from backend.core import usage
from backend.core.llm_client import get_llm, clean_message_content
from backend.core.tracing import traced_request, traced_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document intelligence assistant. Your job is to answer questions "
    "by searching the ingested document corpus — not from your own knowledge.\n\n"

    "## DEFAULT BEHAVIOR — ALWAYS SEARCH FIRST\n"
    "For EVERY question, definition, factual query, or topic request, call "
    "search_documents FIRST. Never answer from memory. The documents are the "
    "source of truth. If the document has an answer, report it with citations. "
    "If the document does not cover it, say so clearly and briefly.\n\n"

    "## TOOLS\n"
    "- search_documents(query, document_scope?): Search ingested docs. Pass the "
    "user's question as `query`. Only pass `document_scope` (array of doc ids) "
    "when the user clearly refers to a specific document — otherwise omit it.\n"
    "- list_documents(): List all ingested documents (id, filename, type, status). "
    "Call this when the user asks what documents exist, or when they mention a "
    "filename you need to look up.\n"
    "- ingest_document(file_path): Ingest a new file the user provides a path for. "
    "Only call this when the user explicitly attaches or names a file to ingest.\n"
    "- sql_read(query): Read-only SQL against the database. Use only when asked.\n\n"

    "## WHEN NOT TO SEARCH\n"
    "Skip search_documents ONLY for:\n"
    "1. Pure greetings/sign-off (hello, thanks, bye) — reply naturally in plain text.\n"
    "2. Requests to list documents — call list_documents instead.\n"
    "3. Requests to ingest a file — call ingest_document instead.\n\n"

    "## AFTER AN INGEST\n"
    "When the user attaches a file: ingest it first. Then if they ask a question "
    "about it, search restricted to that document's id.\n\n"

    "## DISAMBIGUATION\n"
    "If multiple documents match and the user hasn't specified which one, list the "
    "candidates by filename and ask the user to pick — do not guess.\n\n"

    "## STYLE\n"
    "- Be direct and concise. Do not narrate your reasoning or tool usage.\n"
    "- Never say 'I don't have access to files' or 'please provide a file path' "
    "— documents are already ingested and searchable.\n"
    "- Never call the same tool twice with different guesses. One call, one answer.\n"
    "- Never output raw function calls or XML/markdown tags like <function=...> in your text content. Always use the native tool calling feature."
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


def _invoke_with_retry(llm_with_tools, messages, attempts: int = 2):
    """Groq/Llama-3.3 tool-calling occasionally emits its raw text function-call
    syntax instead of a structured tool call, and the API rejects the whole turn
    with a 'tool_use_failed' 400 — a provider-side flake (reproduced independent of
    prompt/tool content), not a logic error here. One retry clears it in practice;
    anything else is a real error and must not be swallowed."""
    import re
    import uuid
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            msg = str(exc)
            if "tool_use_failed" not in msg and "Failed to call a function" not in msg:
                raise

            # Intercept and recover raw tool call format if parsed successfully
            match = re.search(r'<function=(\w+)\(?(.*?)\)?\s*</function>', msg)
            if match:
                name = match.group(1)
                args_str = match.group(2).strip()
                if args_str.endswith('>'):
                    args_str = args_str[:-1].strip()
                if args_str.startswith('(') and args_str.endswith(')'):
                    args_str = args_str[1:-1].strip()

                args = None
                # 1. Try standard JSON parsing
                try:
                    clean_str = args_str.strip().strip("'").strip('"')
                    args = json.loads(clean_str)
                except Exception:
                    try:
                        args = json.loads(clean_str.replace("'", '"').replace("None", "null").replace("null", "null"))
                    except Exception:
                        pass

                # 2. Try parsing keyword arguments (like query="something", document_scope=None)
                if args is None:
                    kwargs = {}
                    for k, v in re.findall(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[^,\s\)]+)', args_str):
                        val = v.strip().strip("'").strip('"')
                        if val.lower() == 'none' or val.lower() == 'null':
                            val = None
                        elif val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        kwargs[k] = val
                    if kwargs:
                        args = kwargs

                if args is not None:
                    logger.info("agent LLM tool call successfully parsed from raw generation string for tool: %s", name)
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": name,
                            "args": args,
                            "id": f"call_{uuid.uuid4().hex}",
                            "type": "tool_call"
                        }]
                    )

            last_exc = exc
            logger.warning("agent LLM malformed tool call (attempt %d/%d): %s", attempt + 1, attempts, msg[:200])
    raise last_exc


def _build_graph(llm_with_tools, registry: dict[str, AgentTool], write_tools: set[str],
                  max_iterations: int):
    def agent_node(state: AgentState) -> dict:
        response = _invoke_with_retry(llm_with_tools, state["messages"])
        usage.record_from_message("agent", response)
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
            # Each tool dispatch gets its own child span under the request's root
            # trace (see traced_request in run_agent below) — this is what lets a
            # single Langfuse trace show every tool the agent called for this
            # request, not just isolated per-call traces.
            with traced_tool(f"tool:{name}", input=args) as span:
                if tool is None:
                    result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        result = tool.run(**args)
                    except Exception as exc:  # a bad tool call must not kill the loop
                        logger.warning("agent tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}
                span["output"] = result
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
    session_id: str = "",
) -> dict:
    """Run one turn of the agent loop.

    Returns:
      {"status": "done", "answer": str, "tool_calls": [...], "messages": [...],
       "token_usage": {...}, "trace_id": str|None}
      {"status": "needs_approval", "pending": [{"id","name","args"}], "tool_calls": [...],
       "answer": None, "messages": [...], "token_usage": {...}, "trace_id": str|None}

    `messages` in the return is the growing LangChain message list — pass it back
    in as `conversation_history` (plus the next user message) to continue the
    conversation with memory of what was asked/called before.

    The whole turn (agent tool-picking LLM calls, every dispatched tool, and any
    LLM calls those tools make internally — e.g. search_documents' query_planner /
    retrieval / answerer) runs inside ONE Langfuse root span (`traced_request`,
    no-op unless LANGFUSE_* env vars are set) and ONE token-usage sink, so a
    single request shows up as a single trace_id in Langfuse with every tool as a
    child span, and "tokens used" in the API response covers the whole turn, not
    just the agent's own tool-picking calls.
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
        # an agent-specific provider needs its own key (e.g. GROQ_API_KEY), not the
        # main llm.api_key placeholder, which may point at a different provider.
        if agent_cfg.get("provider") and agent_cfg["provider"] != config.get("llm", {}).get("provider"):
            llm_cfg["api_key"] = None
        llm = get_llm({**config, "llm": llm_cfg})

    tool_schemas = [_to_openai_tool(t) for t in registry.values()]
    llm_with_tools = llm.bind_tools(tool_schemas)

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    messages += conversation_history or []
    messages.append(HumanMessage(message))

    graph = _build_graph(llm_with_tools, registry, write_tools, max_iterations)

    # ONE root span + ONE token-usage sink for the whole turn — every LLM call
    # and tool dispatch below (however deep, e.g. search_documents' internal
    # query_planner/retrieval/answerer LLM calls) nests under this single trace
    # and accumulates into this single usage total. See tracing.py / usage.py.
    with traced_request(
        "agent_chat", input=message,
        metadata={"session_id": session_id, "approved_writes": approved_writes},
    ) as trace_info, usage.using_sink() as sink:
        final_state = graph.invoke({
            "messages": messages,
            "iterations": 0,
            "pending_approval": None,
            "approved_writes": approved_writes,
        })

    token_usage = sink.totals()
    tool_calls = _extract_tool_calls(final_state["messages"])
    if final_state.get("pending_approval"):
        return {
            "status": "needs_approval",
            "pending": final_state["pending_approval"],
            "tool_calls": tool_calls,
            "answer": None,
            "messages": final_state["messages"],
            "token_usage": token_usage,
            "trace_id": trace_info["trace_id"],
        }

    last = final_state["messages"][-1]
    answer = clean_message_content(last.content) if isinstance(last, AIMessage) else ""
    return {
        "status": "done",
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": final_state["messages"],
        "token_usage": token_usage,
        "trace_id": trace_info["trace_id"],
    }