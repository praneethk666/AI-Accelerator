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
from backend.core.llm_client import get_llm
from backend.core.tracing import traced_request, traced_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the assistant for a document intelligence system. Documents are already "
    "ingested and searchable before this conversation starts — search_documents queries "
    "an existing index and needs no file path. Never say you need a file path or that "
    "nothing's been provided in order to answer a question; that only applies to "
    "ingest_document, and only when the user is adding a NEW file.\n\n"
 
    "TOOLS:\n"
    "- ingest_document: ingest a NEW file at a given path.\n"
    "- search_documents: answer a question from the ingested corpus, with citations. "
    "It takes a query and an optional document_scope — the list of document ids to "
    "restrict the search to. When you want to restrict the search, pass an array "
    "of one or more ids. When you don't want to restrict it — the normal case — "
    "just leave document_scope out of the call entirely rather than filling it in "
    "with anything.\n"
    "- list_documents: list ingested docs (id, filename, document_type, industry, "
    "status), optionally filtered by document_type and/or industry. These are real "
    "categories (e.g. pdf, invoice, finance) set at ingest time — never put a "
    "document's title, topic, or any phrase from inside its content into these "
    "fields hoping it'll match; it won't, and a failed filter guess is not a sign to "
    "retry with another guess. When you don't need to filter, call it with no "
    "arguments at all rather than filling either field in. If a call to "
    "list_documents itself errors out (a real tool failure, not just an empty "
    "result), don't call it again — move straight to search_documents over the "
    "whole corpus instead.\n"
    "- sql_read: read-only SQL.\n\n"
    "Tool results are ground truth — never answer from your own training knowledge "
    "about a document's content, a fact, a definition, or vocabulary; search first. "
    "If list_documents returns rows, report that exact list, verbatim, at that length "
    "— never say 'none' when it returned rows, and never fetch everything then filter/"
    "tally it yourself in prose.\n\n"
 
    "1. GREETINGS / SMALL TALK — NO TOOLS:\n"
    "If the message is purely conversational (hello, thanks, how are you, bye, etc.) "
    "with no actual question or file, just reply in plain text. Don't call any tool.\n\n"
 
    "2. A FILE IS ATTACHED — the message contains an attached-file marker naming the "
    "file and its path:\n"
    "This always means ingest first — no matter what other wording surrounds it "
    "('this file', 'this invoice', 'please ingest this', or anything else). Ingest "
    "that file before anything else, every single time such a marker is present. "
    "Then look at the rest of the message's intent: if it's empty or just a generic "
    "placeholder like 'here' or 'attached', confirm the ingest and stop there, don't "
    "search. If there's a real question or request about the file's content, then "
    "search for it too, restricted to only the document you just ingested — nothing "
    "else.\n\n"
 
    "3. NEXT TURN AFTER AN INGEST — the user says 'this'/'it'/'that file' with "
    "nothing new attached:\n"
    "Keep restricting the search to the same document you ingested earlier in this "
    "conversation. Don't drop that restriction and don't broaden to the whole corpus "
    "just because the new message has no attachment marker of its own — it carries "
    "over until the user clearly moves on to something else.\n\n"
 
    "4. THE QUERY NAMES OR RESEMBLES A FILENAME, not a fresh attachment — e.g. 'the "
    "hybrid rag file', a filename with or without its extension, a shortened version "
    "of one:\n"
    "List the ingested documents once, with no filters, and match by comparing each "
    "filename in the result against the name or words the user used — never use "
    "document_type or industry to try to find it, never list a second time with a "
    "different guessed filter, and never list again if that call errored. If exactly "
    "one document clearly matches, restrict the search to it. If nothing matches at "
    "all, fall back to searching the whole corpus rather than continuing to guess. If "
    "more than one document plausibly matches the same name or description, that's an "
    "ambiguous case — see rule 6: name the candidates and ask which one instead of "
    "picking.\n\n"
 
    "5. LISTING REQUESTS — 'what documents do I have', 'list my invoices', 'show docs "
    "in finance', etc.:\n"
    "List the ingested documents, filtering by document_type and/or industry when the "
    "request names a real category (e.g. 'list my invoices' should filter by the "
    "invoice type). Report the result exactly as returned.\n\n"
 
    "6. AMBIGUITY — ASK, DON'T GUESS:\n"
    "If a question could plausibly apply to more than one ingested document and "
    "nothing narrows it down — e.g. ten invoices are ingested and the user asks for a "
    "value like an HSN number with no document named — stop and ask which document "
    "they mean, naming the candidates by filename and type, instead of picking one, "
    "merging all of them, or guessing. Same if the request itself is unclear enough "
    "that you genuinely don't know what's being asked — ask a short clarifying "
    "question rather than calling a tool on a guess.\n\n"
 
    "7. VAGUE OR GENERIC REFERENCE, OR NO REFERENCE AT ALL — 'this document', 'the "
    "invoice', just a document type, or nothing specific, with no recent attachment "
    "and no clear single candidate:\n"
    "Search across the whole corpus, with no restriction. Don't list documents just "
    "to ask the user to pick unless rule 6's ambiguity actually applies.\n\n"
 
    "GENERAL RULES:\n"
    "- Always pass the user's own question to search_documents, verbatim or lightly "
    "cleaned up — never a placeholder.\n"
    "- Talk naturally; never narrate your own reasoning (e.g. don't say 'since there's "
    "no real question, there's nothing to answer' — just give a short confirmation "
    "like 'Got it — ingested invoice.pdf, ready to search.').\n"
    "- Don't call a tool you don't need, and don't call the same tool twice in a row "
    "with a different guessed argument hoping one sticks — once you have enough to "
    "answer (or enough to know you can't, or enough to know you should ask), respond "
    "in plain text."
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
    answer = last.content if isinstance(last, AIMessage) else ""
    return {
        "status": "done",
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": final_state["messages"],
        "token_usage": token_usage,
        "trace_id": trace_info["trace_id"],
    }