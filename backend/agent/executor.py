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
from backend.core.llm_client import get_llm, get_llm_for, clean_message_content, resolve_model_provider
from backend.core.tracing import traced_request, traced_tool, record_handled_error

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document intelligence assistant. Your job is to answer questions "
    "by searching the ingested document corpus — NOT from your own knowledge.\n\n"

    "## ABSOLUTE MANDATE — ALWAYS CALL search_documents FIRST\n"
    "You are strictly prohibited from answering any question using internal memory or pre-training knowledge.\n"
    "For EVERY question, numbered item, factual query, or section lookup, you MUST call "
    "search_documents FIRST before returning any text answer. Never output an answer without calling search_documents.\n"
    "If the document context contains the answer, report it with exact inline citations.\n"
    "If the document context does not contain the answer, state plainly: "
    "'I could not find this in the provided documents.'\n\n"

    "## TOOLS\n"
    "- search_documents(query, document_scope?, doc_type?, industry?): Search ingested docs. "
    "Pass the user's question as `query`. Pass `document_scope` (array of doc ids or filenames) "
    "when the user clearly refers to specific documents. Pass `doc_type` (e.g. 'invoice', "
    "'manual') or `industry` ONLY when the question clearly implies a scope (e.g. 'in the "
    "invoices…') and the value matches something you saw via list_documents — otherwise omit "
    "all filters and search everything. A filter should narrow on clear intent, never on a guess.\n"
    "- get_page_context(document_id, page): Fetch a document PAGE's full raw content, "
    "bypassing chunking entirely. Chunking sometimes fragments a page badly (e.g. a "
    "label/code split from the table that explains it) — if a search_documents result "
    "looks incomplete, thin, or cuts off mid-thought (a bare heading or code with no "
    "real content), call this with that source's document_id and page to get the whole "
    "page and re-answer from that. Don't call this speculatively on every search — only "
    "when a returned chunk genuinely looks too fragmented to answer confidently from.\n"
    "- view_page_image(document_id, page, question): Look at the ACTUAL rendered "
    "image of a page — for when the answer depends on something visual that text "
    "extraction can lose (which icon lines up with which table row, a diagram's "
    "layout, a callout's position) and get_page_context's plain text isn't enough. "
    "This is a real vision-model call — try get_page_context FIRST; only use this "
    "when you've read the text and still can't answer confidently because of "
    "something visual. Ask a specific question, not a generic 'describe this page'.\n"
    "- list_documents(): List all ingested documents (id, filename, type, status). "
    "Call this when the user asks what documents exist, or when they mention a "
    "filename you need to look up.\n"
    "- ingest_document(file_path): Ingest a new file the user provides a path for. "
    "Only call this when the user explicitly attaches a file or provides a local path to import a new file.\n"
    "- sql_read(query): Read-only SQL against the database. Use only when asked.\n"
    "- request_clarification(question, options?): Ask the USER to choose when their "
    "request is ambiguous (e.g. several documents match). Prefer this over guessing.\n\n"

    "## WHEN NOT TO SEARCH\n"
    "Skip search_documents ONLY for:\n"
    "1. Pure greetings/sign-off (hello, thanks, bye) — reply naturally in plain text.\n"
    "2. Requests to list documents — call list_documents instead.\n"
    "3. Requests to ingest a new file — call ingest_document instead.\n\n"

    "## FILENAME RESTRICTIONS\n"
    "If the user query or conversation history mentions a specific file name "
    "(e.g., 'major-08.pptx'), you MUST restrict your search strictly to that document "
    "by passing its filename or UUID in the `document_scope` parameter of `search_documents`. "
    "Never search the entire corpus when a specific file is targeted.\n\n"

    "## AFTER AN INGEST\n"
    "When the user attaches a file: ingest it first. Then if they ask a question "
    "about it, search restricted ONLY to that document's id or filename.\n\n"

    "## DISAMBIGUATION\n"
    "If several documents plausibly match and the user hasn't said which, call "
    "list_documents to see the candidates, then call request_clarification with the "
    "question and the candidate filenames as options — do NOT guess or pick one silently.\n\n"

    "## MULTI-STEP\n"
    "You MAY chain tools when a task needs it — e.g. list_documents to find a file, "
    "then search_documents scoped to it; several searches for a multi-part question; "
    "or search_documents -> get_page_context -> view_page_image, escalating only as "
    "far as each step actually requires. Work step by step. Just don't repeat the "
    "SAME call with the same arguments.\n\n"

    "## STYLE\n"
    "- Be direct and concise in your final answer; don't narrate tool mechanics.\n"
    "- Never say 'I don't have access to files' or 'please provide a file path' "
    "— documents are already ingested and searchable.\n"
    "- Never output raw function calls or XML/markdown tags like <function=...> in your text content. Always use the native tool calling feature.\n"
    "- Never call ingest_document just because a filename is mentioned; only call it if the user explicitly attaches a new file or provides a local file path.\n"
    "- Do NOT add your own derivations, reformulations, or 'equivalent forms' of formulas or facts unless that exact form appears verbatim in the retrieved source. Every statement you make must be traceable to a specific cited passage.\n"
    "- For mathematical formulas, ALWAYS use standard Markdown LaTeX syntax: '$$formula$$' for block/display equations and '$formula$' for inline equations. Never output bracket delimiters like '[ ... ]' or '\\[ ... \\]' for math equations."
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    pending_approval: list[dict] | None
    approved_writes: bool
    approved_calls: list[dict] | None
    clarification: dict | None


def _args_key(args: dict | None) -> str:
    """Stable string key for a tool call's args, for approval matching."""
    return json.dumps(args or {}, sort_keys=True, default=str)


def _is_approved(name: str, args: dict, approved_calls: list[dict] | None) -> bool:
    """A write runs only if the human approved a call with the SAME name AND args.
    This binds approval to what was shown — the model can't ingest a different/extra
    file on the approved re-run than the one the user actually authorized."""
    key = _args_key(args)
    return any(
        c.get("name") == name and _args_key(c.get("args")) == key
        for c in (approved_calls or [])
    )


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


_GREETINGS = {"hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", "good morning", "good evening"}


def _is_greeting(text: str) -> bool:
    clean = text.strip().lower().rstrip("!.,")
    return clean in _GREETINGS


def _build_graph(llm, tool_schemas: list[dict], registry: dict[str, AgentTool], write_tools: set[str],
                  clarify_tools: set[str], max_iterations: int, is_question: bool,
                  config: dict, agent_cfg: dict):
    llm_auto = llm.bind_tools(tool_schemas)
    try:
        llm_required = llm.bind_tools(tool_schemas, tool_choice="required")
    except Exception:
        llm_required = llm_auto

    def agent_node(state: AgentState) -> dict:
        iters = state.get("iterations", 0)
        # On iteration 0 for non-greetings, FORCE the model to call a tool (tool_choice="required").
        # On iteration 1+ (after a tool has returned its data), allow auto choice to synthesize text.
        active_llm = llm_required if (iters == 0 and is_question) else llm_auto
        
        # Prune verbose tool messages to save input context tokens for the agent LLM
        pruned_messages = _prune_messages_for_llm(state["messages"])
        response = _invoke_with_retry(active_llm, pruned_messages)

        model_name, provider_name = resolve_model_provider(config, agent_cfg)
        usage.record_from_message("agent", response, model=model_name, provider=provider_name)
        return {"messages": [response], "iterations": iters + 1}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        pending: list[dict] = []
        clarification: dict | None = None
        for call in getattr(last, "tool_calls", None) or []:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

            # Control tool: the agent is asking the USER to choose. Pause the loop and
            # surface it as needs_clarification instead of dispatching anything.
            if name in clarify_tools:
                if clarification is None:
                    clarification = {
                        "question": args.get("question") or "Which option?",
                        "options": args.get("options") or [],
                    }
                tool_messages.append(ToolMessage(
                    content="awaiting user selection", tool_call_id=call_id,
                ))
                continue

            # Write gate: run ONLY if the human approved a call with the same name+args.
            if name in write_tools and not (
                state.get("approved_writes") and _is_approved(name, args, state.get("approved_calls"))
            ):
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
            # single Grafana trace show every tool the agent called for this
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
                        record_handled_error(
                            "tool_failure", str(exc), **{"tool.name": name}
                        )
                span["output"] = result
            tool_messages.append(ToolMessage(
                content=json.dumps(result, default=str), tool_call_id=call_id,
            ))

        return {"messages": tool_messages, "pending_approval": pending or None,
                "clarification": clarification}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        if not has_calls:
            return END
        if state.get("iterations", 0) < max_iterations:
            return "tools"
        # At the iteration cap: still route to tools if the final turn proposed a
        # WRITE or CLARIFY control call, so it surfaces as needs_approval /
        # needs_clarification instead of being silently dropped (AC8). Plain reads
        # at the cap just end — the agent already had its budget.
        control = write_tools | clarify_tools
        wants_control = any(c["name"] in control for c in last.tool_calls)
        return "tools" if wants_control else END

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_approval") or state.get("clarification"):
            return END
        if state.get("iterations", 0) >= max_iterations:
            return END
        return "agent"

    sg = StateGraph(AgentState)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", tools_node)
    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    sg.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return sg.compile()


# Max characters to keep in each ToolMessage payload sent back to the agent LLM.
# search_documents returns full citation JSON with snippets (~3-8 KB per call);
# after 3+ searches the raw payloads alone can exceed 30 KB, which:
#   (a) costs input tokens on every subsequent agent turn, and
#   (b) causes DeepSeek / other models to truncate their own completion, producing
#       content="" which the blank-answer fallback misreads as "nothing found".
# 2000 chars ≈ 500 tokens — enough to retain answer + top citations, trim the rest.
_TOOL_MSG_MAX_CHARS = 2000


def _prune_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Prunes verbose JSON payloads from ToolMessages to minimize input context tokens
    for the Agent LLM: search_documents-shaped results (answer + citations) get
    reconstructed into a compact "Search Answer / Source Map" summary (relevance-sorted
    citations, per-chunk-type snippet budgets) instead of the raw JSON — the agent only
    needs the answer + a citation lookup, not every raw snippet field. Any ToolMessage
    still over _TOOL_MSG_MAX_CHARS after that (or one that isn't answer/citations-shaped
    at all, e.g. a different tool's raw output) gets hard character-truncated as a
    safety net so a single oversized payload can't blow up context regardless of shape."""
    import json

    def truncate_on_word(text: str, max_chars: int) -> str:
        text_strip = text.strip()
        if len(text_strip) <= max_chars:
            return text_strip
        truncated = text_strip[:max_chars]
        if ' ' in truncated:
            return truncated.rsplit(' ', 1)[0].rstrip(".,| ") + "..."
        return truncated + "..."

    def _hard_truncate(msg: ToolMessage) -> ToolMessage:
        if len(msg.content) <= _TOOL_MSG_MAX_CHARS:
            return msg
        truncated = msg.content[:_TOOL_MSG_MAX_CHARS]
        # Try to keep valid JSON by finding the last complete top-level value
        last_brace = max(truncated.rfind("}"), truncated.rfind("]"))
        if last_brace > 0:
            truncated = truncated[: last_brace + 1]
        truncated += "\n...[truncated for context efficiency]"
        return ToolMessage(content=truncated, tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None))

    pruned = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(m.content)
                if isinstance(data, dict) and "answer" in data:
                    answer = data.get("answer") or ""
                    citations = data.get("citations") or []

                    # Sort citations explicitly by score descending (highest-relevance
                    # first) so our character budget goes to the most useful sources.
                    def get_score(cit):
                        try:
                            return float(cit.get("score") or 0.0)
                        except Exception:
                            return 0.0

                    sorted_citations = sorted(citations, key=get_score, reverse=True)

                    source_lines = []
                    total_snippet_chars = 0
                    max_total_snippet_chars = 3000

                    for idx, c in enumerate(sorted_citations):
                        fname = c.get("filename")
                        page = c.get("page")
                        doc_id = c.get("document_id")
                        snippet = c.get("snippet") or ""
                        chunk_type = c.get("chunk_type")

                        if fname:
                            # 1-based index to match inline document citation indices
                            page_suffix = f" (page {page})" if page is not None else ""
                            id_suffix = f" [id: {doc_id}]" if doc_id else ""

                            snippet_suffix = ""
                            if snippet:
                                is_special = chunk_type in ("warning", "alarm", "troubleshooting_row")
                                max_len = 450 if is_special else 200

                                truncated_snippet = truncate_on_word(snippet, max_len)
                                if truncated_snippet and total_snippet_chars + len(truncated_snippet) <= max_total_snippet_chars:
                                    snippet_suffix = f" | Snippet: {truncated_snippet}"
                                    total_snippet_chars += len(truncated_snippet)

                            source_lines.append(f"  [{idx + 1}] = {fname}{page_suffix}{id_suffix}{snippet_suffix}")

                    pruned_content = f"Search Answer: {answer}\n"
                    if source_lines:
                        pruned_content += "Source Map:\n" + "\n".join(source_lines)

                    pruned.append(_hard_truncate(ToolMessage(
                        content=pruned_content,
                        tool_call_id=m.tool_call_id,
                        name=m.name,
                    )))
                    continue
            except Exception:
                pass
            m = _hard_truncate(m)
        pruned.append(m)
    return pruned


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
    approved_calls: list[dict] | None = None,
    session_id: str = "",
) -> dict:
    """Run one turn of the agent loop.

    Returns:
      {"status": "done", "answer": str, "tool_calls": [...], "messages": [...],
       "token_usage": {...}, "trace_id": str|None}
      {"status": "needs_approval", "pending": [{"id","name","args"}], "tool_calls": [...],
       "answer": None, "messages": [...], "token_usage": {...}, "trace_id": str|None}
      {"status": "needs_clarification", "question": str, "options": [str], "answer": question,
       "tool_calls": [...], "messages": [...], "token_usage": {...}, "trace_id": str|None}

    Approval is BOUND to args: to approve a pending write, re-invoke with
    approved_writes=True AND approved_calls=<the pending list you showed the user>;
    a write runs only if its name+args match an approved call.

    `messages` in the return is the growing LangChain message list — pass it back
    in as `conversation_history` (plus the next user message) to continue the
    conversation with memory of what was asked/called before.

    The whole turn (agent tool-picking LLM calls, every dispatched tool, and any
    LLM calls those tools make internally — e.g. search_documents' query_planner /
    retrieval / answerer) runs inside ONE OpenTelemetry root span (`traced_request`,
    no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set) and ONE token-usage sink, so a
    single request shows up as a single trace_id in Grafana Tempo with every tool
    as a child span, and "tokens used" in the API response covers the whole turn,
    not just the agent's own tool-picking calls
    """
    registry = registry if registry is not None else build_agent_registry()
    agent_cfg = (config.get("query") or {}).get("agent") or {}
    max_iterations = agent_cfg.get("max_iterations", 5)
    write_tools = set(agent_cfg.get("write_tools") or [])
    clarify_tools = set(agent_cfg.get("clarify_tools") or ["request_clarification"])

    if llm is None:
        # get_llm_for handles base_url/api_key overrides too (not just provider/model)
        # — needed when the agent points at a DIFFERENT OpenAI-compatible endpoint
        # than the global llm block (e.g. NVIDIA NIM vs z.ai — both provider: openai,
        # different base_url/key; a provider-only diff check would miss this and try
        # NVIDIA's model name against z.ai's endpoint).
        llm = get_llm_for(config, agent_cfg)

    tool_schemas = [_to_openai_tool(t) for t in registry.values()]
    is_question = not _is_greeting(message)
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    messages += conversation_history or []
    messages.append(HumanMessage(message))

    graph = _build_graph(llm, tool_schemas, registry, write_tools, clarify_tools,
                          max_iterations, is_question, config, agent_cfg)

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
            "approved_calls": approved_calls,
            "clarification": None,
        })

    token_usage = sink.totals()
    tool_calls = _extract_tool_calls(final_state["messages"])
    if final_state.get("clarification"):
        clar = final_state["clarification"]
        return {
            "status": "needs_clarification",
            "question": clar.get("question"),
            "options": clar.get("options") or [],
            # answer mirrors the question so simple clients still show something
            "answer": clar.get("question"),
            "tool_calls": tool_calls,
            "messages": final_state["messages"],
            "token_usage": token_usage,
            "trace_id": trace_info["trace_id"],
        }
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
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    # If the last message has tool calls, its content is intermediate monologue/narration
    # (e.g. "Now I have all the data. Let me check..."), NOT a final answer.
    answer = clean_message_content(last.content) if (isinstance(last, AIMessage) and not has_tool_calls) else ""
    if not answer.strip():
        # Diagnostic: log WHY the answer is blank so this is traceable in logs.
        if isinstance(last, AIMessage):
            logger.warning(
                "Agent produced blank answer (content=%r, tool_calls=%s, iters=%d) — "
                "triggering fallback synthesis",
                last.content,
                bool(getattr(last, "tool_calls", None)),
                final_state.get("iterations", 0),
            )
        else:
            logger.warning(
                "Agent ended on a %s (not AIMessage, iters=%d) — triggering fallback",
                type(last).__name__,
                final_state.get("iterations", 0),
            )

        # Fast path: if any prior search_documents tool message already contains an
        # answer field, use it directly instead of making another LLM call.
        # This handles the common case where DeepSeek returns content="" on the
        # synthesis turn after the tool already returned a complete answer.
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, ToolMessage):
                try:
                    tool_result = json.loads(msg.content)
                    if isinstance(tool_result, dict) and tool_result.get("answer", "").strip():
                        answer = tool_result["answer"]
                        logger.info("Recovered answer from prior tool result (fast-path fallback)")
                        break
                except Exception:
                    pass

        # Slow path: ask the LLM to synthesize from history without tools.
        if not answer.strip():
            try:
                fallback_prompt = (
                    "\n[System Note: You have reached the maximum search step limit. "
                    "Based on the tool results above, synthesize a final answer for the user. "
                    "If the documents contained the answer, report it with inline citations. "
                    "If not, state plainly that you could not find it in the provided documents.]"
                )
                fallback_messages = list(final_state["messages"])
                # If the last message has tool calls they must be answered before sending.
                if isinstance(last, AIMessage) and last.tool_calls:
                    for call in last.tool_calls:
                        fallback_messages.append(ToolMessage(
                            content="error: maximum search step limit reached",
                            tool_call_id=call["id"],
                            name=call.get("name")
                        ))
                messages_for_fallback = fallback_messages + [SystemMessage(content=fallback_prompt)]
                fallback_response = llm.invoke(messages_for_fallback)
                answer = clean_message_content(fallback_response.content)
                usage.record_from_message("agent_fallback", fallback_response)
            except Exception as exc:
                logger.warning("Agent fallback LLM invocation failed: %s", exc)

        if not answer.strip():
            answer = (
                "I searched the documents but couldn't produce a complete answer within "
                "the available steps. Please try rephrasing your question or naming the "
                "specific document you want to search."
            )
    return {
        "status": "done",
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": final_state["messages"],
        "token_usage": token_usage,
        "trace_id": trace_info["trace_id"],
    }
