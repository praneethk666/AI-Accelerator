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

from backend.guardrails.config_schema import validate_guardrail_config
from backend.guardrails.input_guard import check_input
from backend.guardrails.output_guard import mask_output
from backend.guardrails.policy_engine import get_engine
from backend.guardrails.session_risk import get_accumulator
from backend.guardrails.event_logger import log_event
from backend.guardrails.guard_decision import GuardDecision, GuardEvidence, PolicyDecision, SAFE_REPLY_MESSAGE
from backend.guardrails.rollout import should_apply_guard
from backend.guardrails.retrieval_guard import scan_tool_output_async

logger = logging.getLogger(__name__)

def add_evidences(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """LangGraph reducer to accumulate guard evidences across nodes."""
    return (left or []) + (right or [])

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    pending_approval: list[dict] | None
    approved_writes: bool
    approved_calls: list[dict] | None
    clarification: dict | None
    # guardrail fields
    guard_blocked: bool
    safe_answer: str | None
    guard_evidences: Annotated[list[dict], add_evidences]
    guard_risk_score: int
    guard_policy: str | None

SYSTEM_PROMPT = (
    "You are a document intelligence assistant. Your job is to answer questions "
    "by searching the ingested document corpus — NOT from your own knowledge.\n\n"

    "## ABSOLUTE MANDATE — ALWAYS CALL search_documents OR excel_tool FIRST\n"
    "You are strictly prohibited from answering any question using internal memory or pre-training knowledge.\n"
    "For EVERY question, you MUST call search_documents (or excel_tool if the query is an Excel calculation) "
    "FIRST before returning any text answer. Never output an answer without calling a tool.\n"
    "If the document context contains the answer, report it with exact inline citations.\n"
    "If the document context does not contain the answer, state plainly: "
    "'I could not find this in the provided documents.'\n\n"

    "## TOOLS\n"
    "- search_documents(query, document_scope?, doc_type?, industry?): Search ingested docs. "
    "Pass the user's question as `query`. Pass `document_scope` (array of doc ids or filenames) "
    "ONLY when the user explicitly quotes a real filename or document_id that you have seen in a "
    "list_documents result — NEVER invent, guess, or make up a filename. "
    "Pass `doc_type` (e.g. 'invoice', 'manual') or `industry` ONLY when the question clearly implies a scope (e.g. 'in the "
    "invoices…') and the value matches something you saw via list_documents — otherwise omit "
    "all filters and search everything. A filter should narrow on clear intent, never on a guess.\n"
    "- get_page_context(document_id, page): Fetch a document PAGE's full raw content, "
    "bypassing chunking entirely. Chunking sometimes fragments a page badly (e.g. a "
    "label/code split from the table that explains it) — if a search_documents result "
    "looks incomplete, thin, or cuts off mid-thought (a bare heading or code with no "
    "real content), call this with that source's document_id and page to get the whole "
    "page and re-answer from that. Don't call this speculatively on every search — only "
    "when a returned chunk genuinely looks too fragmented to answer confidently from.\n"
    "- list_documents(): List all ingested documents (id, filename, type, status). "
    "Call this ONLY when the user explicitly asks 'what files exist?', 'show loaded documents', or 'list files'. "
    "NEVER call list_documents for content questions or short phrases like 'model name' — use search_documents instead.\n"
    "- ingest_document(file_path): Ingest a new file the user provides a path for. "
    "Only call this when the user explicitly attaches a file or provides a local path to import a new file.\n"
    "- sql_read(query): Read-only SQL against the database. Use only when asked.\n"
    "- excel_tool(filename_or_id, code, sheet_name?): Execute Python/Pandas code "
    "directly on an Excel file's data. Use this when a question requires calculation, "
    "filtering, aggregation, comparison, or any data lookup/manipulation on Excel sheets. You MUST follow these rules:\n"
    "      1. Assign the final output to `result` (e.g., `result = ...`). Do NOT use return statements.\n"
    "      2. NEVER use `import` statements or restricted builtins like `dir()`, `globals()`, `locals()`, `hasattr()`, `setattr()`, `getattr()`, `eval()`, `exec()` — they are blocked by the secure sandbox. Pandas (pd), Numpy (np), sqlite3, math, datetime, re, and json are pre-imported and available.\n"
    "      3. To list all sheets, set sheet_name='all' and use `list(dfs.keys())`.\n"
    "      4. To inspect columns/rows of sheet 'Vendor A', use `dfs['Vendor A'].columns.tolist()` or `dfs['Vendor A'].iloc[:5]`.\n"
    "      5. Work step-by-step: first list the sheet names, then inspect columns/rows of relevant sheets to find headers, then perform the final calculation.\n"
    "- request_clarification(question, options?): Ask the USER to choose when their "
    "request is ambiguous (e.g. several documents match). Prefer this over guessing.\n\n"

    "## WHEN NOT TO SEARCH\n"
    "Skip search_documents ONLY for:\n"
    "1. Pure greetings/sign-off (hello, thanks, bye) — reply naturally in plain text.\n"
    "2. Requests to list documents — call list_documents instead.\n"
    "3. Requests to ingest a new file — call ingest_document instead.\n"
    "4. Questions requiring calculation, filtering, comparisons, sorting, aggregation, data lookup, or "
    "looking up text notes/inclusions/exclusions/metadata on Excel files (e.g. 'sum column X', "
    "'which company has highest revenue', 'is scaffolding included', 'list exclusions') — call excel_tool instead. "
    "It runs real Pandas code on the actual data and is far more accurate and token-efficient than searching chunked text.\n"
    "   CRITICAL: Bypassing excel_tool for Excel files and calling search_documents is STRICTLY PROHIBITED. RAG/Search "
    "retrieves massive amounts of raw text/tables, which causes extreme context window bloat (up to 500,000+ tokens) and poor "
    "computational accuracy. First resolve the Excel file name using list_documents/sql_read if needed, then run your "
    "computational/filtering code entirely inside excel_tool.\n\n"

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
    "or search_documents followed by get_page_context on a fragmented result. "
    "Work step by step. Just don't repeat the SAME call with the same arguments.\n\n"

    "## COMPLETION\n"

    "For each user request:\n"

    "1. Perform only the searches necessary to answer the request completely.\n"

    "2. If the retrieved information is sufficient to answer all parts of the current request, STOP calling tools and produce the final answer.\n"

    "3. Do not perform additional searches merely to restate, refine, or expand information you already have.\n"

    "4. Use previous conversation only to resolve references or follow-up questions. Do not continue incomplete answers from previous turns unless the user explicitly asks you to continue.\n"
    "5. Each user message is answered on its own merits. If it is unrelated to a "
    "previous question in this conversation, answer ONLY the new question — do not "
    "repeat, merge with, or re-summarize a prior answer just because it's visible "
    "above. Only pull in prior conversation content when the new question explicitly "
    "references it (e.g. 'and what about...', 'the second one you mentioned').\n\n"

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


async def _ainvoke_with_retry(llm_with_tools, messages, attempts: int = 2):
    import re
    import uuid
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await llm_with_tools.ainvoke(messages)
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
                try:
                    clean_str = args_str.strip().strip("'").strip('"')
                    args = json.loads(clean_str)
                except Exception:
                    try:
                        args = json.loads(clean_str.replace("'", '"').replace("None", "null"))
                    except Exception:
                        pass

                if args is None:
                    kwargs = {}
                    for k, v in re.findall(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[^,\s\)]+)', args_str):
                        val = v.strip().strip("'").strip('"')
                        if val.lower() in ('none', 'null'):
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
                  config: dict, agent_cfg: dict, session_id: str = ""):
    llm_auto = llm.bind_tools(tool_schemas)
    try:
        llm_required = llm.bind_tools(tool_schemas, tool_choice="required")
    except Exception:
        llm_required = llm_auto

    g_cfg = config.get("guardrails") or {}

    def input_guard_node(state: AgentState) -> dict:
        if not g_cfg.get("enabled", True):
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        rollout_pct = g_cfg.get("rollout", {}).get("input_guard_pct", 100)
        if not should_apply_guard(rollout_pct, session_id):
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        raw_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                raw_msg = msg.content
                break

        if not raw_msg:
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        version = g_cfg.get("version", "1.0.0")
        decision: GuardDecision = check_input(
            raw_msg, config=config, session_id=session_id, version=version
        )
        log_event(decision, session_id=session_id, config=config)

        accum = get_accumulator(config)
        session_risk_pct = g_cfg.get("rollout", {}).get("session_risk_pct", 20)
        should_block_session = False
        cumulative_score = decision.risk_score
        
        if should_apply_guard(session_risk_pct, session_id):
            should_block_session, cumulative_score = accum.add_and_check(session_id, decision.risk_score)
            if should_block_session:
                logger.warning("Session %s blocked due to cumulative multi-turn risk", session_id)

        ev = GuardEvidence(
            stage="input",
            risk_score=decision.risk_score,
            events=[decision.event_type] if decision.event_type else [],
            rule_ids=[decision.rule_id] if decision.rule_id else [],
            latency_ms=decision.latency_ms,
            bypassed=decision.bypassed,
            hard_block=decision.hard_block or should_block_session,
        )

        engine = get_engine(config)
        policy_decision = engine.evaluate([ev])

        if policy_decision == PolicyDecision.BLOCK or should_block_session:
            blocked_msg = AIMessage(
                content="Your request contains triggers that violate our safety policies."
            )
            return {
                "messages": [blocked_msg],
                "guard_blocked": True,
                "guard_evidences": [ev.__dict__],
                "guard_risk_score": decision.risk_score,
                "guard_policy": policy_decision.value,
                "safe_answer": blocked_msg.content,
            }

        ret_dict = {
            "guard_blocked": False,
            "guard_evidences": [ev.__dict__],
            "guard_risk_score": decision.risk_score,
            "guard_policy": policy_decision.value,
        }

        if decision.sanitized_value:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    redacted_human_msg = HumanMessage(
                        content=decision.sanitized_value,
                        id=msg.id,
                    )
                    ret_dict["messages"] = [redacted_human_msg]
                    break

        return ret_dict

    def agent_node(state: AgentState) -> dict:
        iters = state.get("iterations", 0)
        active_llm = llm_required if (iters == 0 and is_question) else llm_auto
        
        pruned_messages = _prune_messages_for_llm(state["messages"])
        response = _invoke_with_retry(active_llm, pruned_messages)

        model_name, provider_name = resolve_model_provider(config, agent_cfg)
        usage.record_from_message("agent", response, prompt=pruned_messages, model=model_name, provider=provider_name)
        return {"messages": [response], "iterations": iters + 1}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        pending: list[dict] = []
        clarification: dict | None = None
        new_evidences: list[dict] = []

        for call in getattr(last, "tool_calls", None) or []:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

            if name in clarify_tools:
                if clarification is None:
                    clarification = {
                        "question": args.get("question") or "Which option?",
                        "options": args.get("options") or [],
                    }
                tool_messages.append(ToolMessage(
                    content="awaiting user selection", tool_call_id=call_id, name=name,
                ))
                continue

            if name == "ingest_document":
                import os
                file_path = args.get("file_path", "")
                
                # Check database for existing document matching the filename
                from backend.storage.postgres_store import PostgresStore
                db_doc = None
                try:
                    pg = PostgresStore()
                    try:
                        fname = os.path.basename(file_path)
                        cur = pg.conn.cursor()
                        cur.execute(
                            "SELECT document_id, file_path, status FROM documents WHERE filename = %s OR filename = %s OR file_path = %s ORDER BY created_at DESC LIMIT 1",
                            (file_path, fname, file_path)
                        )
                        row = cur.fetchone()
                        if row:
                            db_doc = {
                                "document_id": str(row[0]),
                                "file_path": row[1],
                                "status": row[2]
                            }
                    finally:
                        pg.close()
                except Exception as db_exc:
                    logger.warning("Failed to query documents table in tools_node: %s", db_exc)

                # Resolve file_path to database file_path if it exists on disk
                resolved_path = file_path
                if db_doc and db_doc["file_path"] and os.path.isfile(db_doc["file_path"]):
                    resolved_path = db_doc["file_path"]
                    args["file_path"] = resolved_path

                # If the document is already ready and file exists, we don't even need approval or tool execution!
                if db_doc and db_doc["status"] == "ready" and os.path.isfile(resolved_path):
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "document_id": db_doc["document_id"],
                            "status": "ready",
                            "message": f"Document {fname!r} is already ingested and ready for query."
                        }, default=str),
                        tool_call_id=call_id, name=name,
                    ))
                    continue

                # If file path doesn't exist on disk, return FileNotFoundError immediately
                if not resolved_path or not os.path.isfile(resolved_path):
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "error": f"FileNotFoundError: File {file_path!r} does not exist. "
                                     f"Please upload/attach the file in the chat input first."
                        }, default=str),
                        tool_call_id=call_id, name=name,
                    ))
                    continue

            if name in write_tools and not (
                state.get("approved_writes") and _is_approved(name, args, state.get("approved_calls"))
            ):
                pending.append({"id": call_id, "name": name, "args": args})
                tool_messages.append(ToolMessage(
                    content="blocked: this action writes data and needs human "
                            "approval before it can run.",
                    tool_call_id=call_id, name=name,
                ))
                continue

            tool = registry.get(name)
            with traced_tool(f"tool:{name}", input=args) as span:
                if tool is None:
                    result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        result = tool.run(**args)
                    except Exception as exc:
                        logger.warning("agent tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}
                        record_handled_error(
                            "tool_failure", str(exc), **{"tool.name": name}
                        )
                
                version = g_cfg.get("version", "1.0.0")
                from backend.guardrails.retrieval_guard import scan_tool_output
                scan_decision = scan_tool_output(
                    result, config=config, session_id=session_id, version=version
                )
                log_event(scan_decision, session_id=session_id, config=config)

                if scan_decision.risk_score > 0 or scan_decision.bypassed:
                    ev = GuardEvidence(
                        stage="retrieval",
                        risk_score=scan_decision.risk_score,
                        events=[scan_decision.event_type] if scan_decision.event_type else [],
                        rule_ids=[scan_decision.rule_id] if scan_decision.rule_id else [],
                        latency_ms=scan_decision.latency_ms,
                        bypassed=scan_decision.bypassed,
                        hard_block=scan_decision.hard_block,
                    )
                    new_evidences.append(ev.__dict__)

                cleaned_result = scan_decision.sanitized_value if scan_decision.sanitized_value is not None else result
                span["output"] = cleaned_result
                
            tool_messages.append(ToolMessage(
                content=json.dumps(cleaned_result, default=str), tool_call_id=call_id, name=name,
            ))

        return {
            "messages": tool_messages,
            "pending_approval": pending or None,
            "clarification": clarification,
            "guard_evidences": new_evidences,
        }

    def output_guard_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        raw_answer = clean_message_content(last.content) if isinstance(last, AIMessage) else ""
        
        if not g_cfg.get("enabled", True):
            return {
                "safe_answer": raw_answer,
            }

        raw_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw_msg = msg
                break

        if raw_msg is None:
            return {
                "safe_answer": raw_answer,
            }

        version = g_cfg.get("version", "1.0.0")
        decision: GuardDecision = mask_output(
            raw_msg.content, config=config, version=version
        )
        log_event(decision, session_id=session_id, config=config)

        ev = GuardEvidence(
            stage="output",
            risk_score=decision.risk_score,
            events=[decision.event_type] if decision.event_type else [],
            rule_ids=[decision.rule_id] if decision.rule_id else [],
            latency_ms=decision.latency_ms,
            bypassed=decision.bypassed,
            hard_block=decision.hard_block,
        )

        prior_evidences = state.get("guard_evidences") or []
        ev_objs = []
        for e in prior_evidences:
            if isinstance(e, dict):
                ev_objs.append(GuardEvidence(**e))
            elif isinstance(e, GuardEvidence):
                ev_objs.append(e)
        ev_objs.append(ev)

        engine = get_engine(config)
        policy_decision = engine.evaluate(ev_objs)

        safe_content = decision.sanitized_value if decision.sanitized_value is not None else raw_msg.content
        blocked = False

        if policy_decision == PolicyDecision.BLOCK:
            safe_content = SAFE_REPLY_MESSAGE
            blocked = True

        redacted_msg = AIMessage(
            content=safe_content,
            id=raw_msg.id,
        )

        scores = [e.risk_score for e in ev_objs]
        max_score = max(scores) if scores else 0

        return {
            "messages": [redacted_msg],
            "safe_answer": safe_content,
            "guard_blocked": blocked,
            "guard_evidences": [e.__dict__ for e in ev_objs],
            "guard_risk_score": max_score,
            "guard_policy": policy_decision.value,
        }

    def route_after_input(state: AgentState) -> str:
        if state.get("guard_blocked"):
            return "output_guard"
        return "agent"

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        if not has_calls:
            return "output_guard"
        if state.get("iterations", 0) < max_iterations:
            return "tools"
        control = write_tools | clarify_tools
        wants_control = any(c["name"] in control for c in last.tool_calls)
        return "tools" if wants_control else "output_guard"

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_approval") or state.get("clarification"):
            return "output_guard"
        if state.get("iterations", 0) >= max_iterations:
            return "output_guard"
        return "agent"

    sg = StateGraph(AgentState)
    sg.add_node("input_guard", input_guard_node)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", tools_node)
    sg.add_node("output_guard", output_guard_node)
    
    sg.add_edge(START, "input_guard")
    sg.add_conditional_edges("input_guard", route_after_input, {"agent": "agent", "output_guard": "output_guard"})
    sg.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "output_guard": "output_guard"})
    sg.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "output_guard": "output_guard"})
    sg.add_edge("output_guard", END)
    return sg.compile()


def _build_async_graph(llm, tool_schemas: list[dict], registry: dict[str, AgentTool], write_tools: set[str],
                        clarify_tools: set[str], max_iterations: int, is_question: bool,
                        config: dict, agent_cfg: dict, session_id: str = ""):
    llm_auto = llm.bind_tools(tool_schemas)
    try:
        llm_required = llm.bind_tools(tool_schemas, tool_choice="required")
    except Exception:
        llm_required = llm_auto

    g_cfg = config.get("guardrails") or {}

    async def input_guard_node(state: AgentState) -> dict:
        if not g_cfg.get("enabled", True):
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        rollout_pct = g_cfg.get("rollout", {}).get("input_guard_pct", 100)
        if not should_apply_guard(rollout_pct, session_id):
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        raw_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                raw_msg = msg.content
                break

        if not raw_msg:
            return {
                "guard_blocked": False,
                "guard_evidences": [],
                "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        version = g_cfg.get("version", "1.0.0")
        decision: GuardDecision = check_input(
            raw_msg, config=config, session_id=session_id, version=version
        )
        log_event(decision, session_id=session_id, config=config)

        accum = get_accumulator(config)
        session_risk_pct = g_cfg.get("rollout", {}).get("session_risk_pct", 20)
        should_block_session = False
        cumulative_score = decision.risk_score
        
        if should_apply_guard(session_risk_pct, session_id):
            should_block_session, cumulative_score = accum.add_and_check(session_id, decision.risk_score)
            if should_block_session:
                logger.warning("Session %s blocked due to cumulative multi-turn risk", session_id)

        ev = GuardEvidence(
            stage="input",
            risk_score=decision.risk_score,
            events=[decision.event_type] if decision.event_type else [],
            rule_ids=[decision.rule_id] if decision.rule_id else [],
            latency_ms=decision.latency_ms,
            bypassed=decision.bypassed,
            hard_block=decision.hard_block or should_block_session,
        )

        engine = get_engine(config)
        policy_decision = engine.evaluate([ev])

        if policy_decision == PolicyDecision.BLOCK or should_block_session:
            blocked_msg = AIMessage(
                content="Your request contains triggers that violate our safety policies."
            )
            return {
                "messages": [blocked_msg],
                "guard_blocked": True,
                "guard_evidences": [ev.__dict__],
                "guard_risk_score": decision.risk_score,
                "guard_policy": policy_decision.value,
                "safe_answer": blocked_msg.content,
            }

        ret_dict = {
            "guard_blocked": False,
            "guard_evidences": [ev.__dict__],
            "guard_risk_score": decision.risk_score,
            "guard_policy": policy_decision.value,
        }

        if decision.sanitized_value:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    redacted_human_msg = HumanMessage(
                        content=decision.sanitized_value,
                        id=msg.id,
                    )
                    ret_dict["messages"] = [redacted_human_msg]
                    break

        return ret_dict

    async def agent_node(state: AgentState) -> dict:
        iters = state.get("iterations", 0)
        active_llm = llm_required if (iters == 0 and is_question) else llm_auto
        
        pruned_messages = _prune_messages_for_llm(state["messages"])
        response = await _ainvoke_with_retry(active_llm, pruned_messages)

        model_name, provider_name = resolve_model_provider(config, agent_cfg)
        usage.record_from_message("agent", response, prompt=pruned_messages, model=model_name, provider=provider_name)
        return {"messages": [response], "iterations": iters + 1}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        pending: list[dict] = []
        clarification: dict | None = None
        new_evidences: list[dict] = []

        for call in getattr(last, "tool_calls", None) or []:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

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

            if name == "ingest_document":
                import os
                file_path = args.get("file_path", "")
                
                # Check database for existing document matching the filename
                from backend.storage.postgres_store import PostgresStore
                db_doc = None
                try:
                    pg = PostgresStore()
                    try:
                        fname = os.path.basename(file_path)
                        cur = pg.conn.cursor()
                        cur.execute(
                            "SELECT document_id, file_path, status FROM documents WHERE filename = %s OR filename = %s OR file_path = %s ORDER BY created_at DESC LIMIT 1",
                            (file_path, fname, file_path)
                        )
                        row = cur.fetchone()
                        if row:
                            db_doc = {
                                "document_id": str(row[0]),
                                "file_path": row[1],
                                "status": row[2]
                            }
                    finally:
                        pg.close()
                except Exception as db_exc:
                    logger.warning("Failed to query documents table in tools_node: %s", db_exc)

                # Resolve file_path to database file_path if it exists on disk
                resolved_path = file_path
                if db_doc and db_doc["file_path"] and os.path.isfile(db_doc["file_path"]):
                    resolved_path = db_doc["file_path"]
                    args["file_path"] = resolved_path

                # If the document is already ready and file exists, we don't even need approval or tool execution!
                if db_doc and db_doc["status"] == "ready" and os.path.isfile(resolved_path):
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "document_id": db_doc["document_id"],
                            "status": "ready",
                            "message": f"Document {fname!r} is already ingested and ready for query."
                        }, default=str),
                        tool_call_id=call_id,
                    ))
                    continue

                # If file path doesn't exist on disk, return FileNotFoundError immediately
                if not resolved_path or not os.path.isfile(resolved_path):
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "error": f"FileNotFoundError: File {file_path!r} does not exist. "
                                     f"Please upload/attach the file in the chat input first."
                        }, default=str),
                        tool_call_id=call_id,
                    ))
                    continue

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
            with traced_tool(f"tool:{name}", input=args) as span:
                if tool is None:
                    result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        import asyncio
                        if name == "search_documents" and session_id:
                            args["session_id"] = session_id
                        result = await asyncio.to_thread(tool.run, **args)
                        if isinstance(result, dict) and result.get("ambiguity", {}).get("is_ambiguous"):
                            opts = result["ambiguity"].get("options") or []
                            if clarification is None:
                                clarification = {
                                    "question": "The document contains multiple model names. Which component's model are you referring to?",
                                    "options": opts,
                                }
                    except Exception as exc:
                        logger.warning("agent tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}
                        record_handled_error(
                            "tool_failure", str(exc), **{"tool.name": name}
                        )
                
                version = g_cfg.get("version", "1.0.0")
                from backend.guardrails.retrieval_guard import scan_tool_output
                scan_decision = scan_tool_output(
                    result, config=config, session_id=session_id, version=version
                )
                log_event(scan_decision, session_id=session_id, config=config)

                if scan_decision.risk_score > 0 or scan_decision.bypassed:
                    ev = GuardEvidence(
                        stage="retrieval",
                        risk_score=scan_decision.risk_score,
                        events=[scan_decision.event_type] if scan_decision.event_type else [],
                        rule_ids=[scan_decision.rule_id] if scan_decision.rule_id else [],
                        latency_ms=scan_decision.latency_ms,
                        bypassed=scan_decision.bypassed,
                        hard_block=scan_decision.hard_block,
                    )
                    new_evidences.append(ev.__dict__)

                cleaned_result = scan_decision.sanitized_value if scan_decision.sanitized_value is not None else result
                span["output"] = cleaned_result
                
            tool_messages.append(ToolMessage(
                content=json.dumps(cleaned_result, default=str), tool_call_id=call_id,
            ))

        return {
            "messages": tool_messages,
            "pending_approval": pending or None,
            "clarification": clarification,
            "guard_evidences": new_evidences,
        }

    async def output_guard_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        raw_answer = clean_message_content(last.content) if isinstance(last, AIMessage) else ""
        
        if not g_cfg.get("enabled", True):
            return {
                "safe_answer": raw_answer,
            }

        raw_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw_msg = msg
                break

        if raw_msg is None:
            return {
                "safe_answer": raw_answer,
            }

        version = g_cfg.get("version", "1.0.0")
        decision: GuardDecision = mask_output(
            raw_msg.content, config=config, version=version
        )
        log_event(decision, session_id=session_id, config=config)

        ev = GuardEvidence(
            stage="output",
            risk_score=decision.risk_score,
            events=[decision.event_type] if decision.event_type else [],
            rule_ids=[decision.rule_id] if decision.rule_id else [],
            latency_ms=decision.latency_ms,
            bypassed=decision.bypassed,
            hard_block=decision.hard_block,
        )

        prior_evidences = state.get("guard_evidences") or []
        ev_objs = []
        for e in prior_evidences:
            if isinstance(e, dict):
                ev_objs.append(GuardEvidence(**e))
            elif isinstance(e, GuardEvidence):
                ev_objs.append(e)
        ev_objs.append(ev)

        engine = get_engine(config)
        policy_decision = engine.evaluate(ev_objs)

        safe_content = decision.sanitized_value if decision.sanitized_value is not None else raw_msg.content
        blocked = False

        if policy_decision == PolicyDecision.BLOCK:
            safe_content = SAFE_REPLY_MESSAGE
            blocked = True

        redacted_msg = AIMessage(
            content=safe_content,
            id=raw_msg.id,
        )

        scores = [e.risk_score for e in ev_objs]
        max_score = max(scores) if scores else 0

        return {
            "messages": [redacted_msg],
            "safe_answer": safe_content,
            "guard_blocked": blocked,
            "guard_evidences": [e.__dict__ for e in ev_objs],
            "guard_risk_score": max_score,
            "guard_policy": policy_decision.value,
        }

    def route_after_input(state: AgentState) -> str:
        if state.get("guard_blocked"):
            return "output_guard"
        return "agent"

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        if not has_calls:
            return "output_guard"
        if state.get("iterations", 0) < max_iterations:
            return "tools"
        control = write_tools | clarify_tools
        wants_control = any(c["name"] in control for c in last.tool_calls)
        return "tools" if wants_control else "output_guard"

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_approval") or state.get("clarification"):
            return "output_guard"
        if state.get("iterations", 0) >= max_iterations:
            return "output_guard"
        return "agent"

    sg = StateGraph(AgentState)
    sg.add_node("input_guard", input_guard_node)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", tools_node)
    sg.add_node("output_guard", output_guard_node)
    
    sg.add_edge(START, "input_guard")
    sg.add_conditional_edges("input_guard", route_after_input, {"agent": "agent", "output_guard": "output_guard"})
    sg.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "output_guard": "output_guard"})
    sg.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "output_guard": "output_guard"})
    sg.add_edge("output_guard", END)
    return sg.compile()



# Max characters to keep in each ToolMessage payload sent back to the agent LLM.
# search_documents returns full citation JSON with snippets (~3-8 KB per call);
# after 3+ searches the raw payloads alone can exceed 30 KB, which:
#   (a) costs input tokens on every subsequent agent turn, and
#   (b) causes DeepSeek / other models to truncate their own completion, producing
#       content="" which the blank-answer fallback misreads as "nothing found".
# 2000 chars ≈ 500 tokens — enough to retain answer + top citations, trim the rest.
_TOOL_MSG_MAX_CHARS = 2000


# def _prune_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
#     """Cap large ToolMessage payloads to _TOOL_MSG_MAX_CHARS to prevent context bloat.

#     Keeps SystemMessage / HumanMessage / AIMessage intact (they're small).
#     Only trims ToolMessages that carry large JSON search results — the agent
#     only needs the answer + a citation summary, not every raw snippet field.
#     """
#     pruned = []
#     for msg in messages:
#         if isinstance(msg, ToolMessage) and len(msg.content) > _TOOL_MSG_MAX_CHARS:
#             truncated = msg.content[:_TOOL_MSG_MAX_CHARS]
#             # Try to keep valid JSON by finding the last complete top-level value
#             last_brace = max(truncated.rfind("}"), truncated.rfind("]"))
#             if last_brace > 0:
#                 truncated = truncated[: last_brace + 1]
#             truncated += "\n...[truncated for context efficiency]"
#             # ToolMessage is immutable — create a copy with the shorter content
#             msg = ToolMessage(
#                 content=truncated,
#                 tool_call_id=msg.tool_call_id,
#                 name=getattr(msg, "name", None),
#             )
#         pruned.append(msg)
#     return pruned

def _prune_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Reduce ToolMessage payload size without losing answer or citations.

    For JSON tool outputs:
      - Format search_documents output as plain text answer + citations map.
      - Strip heavy debug/metadata fields for other tool outputs.
      - Fall back to character truncation.
    """

    pruned: list[BaseMessage] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            pruned.append(msg)
            continue

        tool_name = getattr(msg, "name", None)
        if tool_name == "list_documents":
            max_chars = 15000
        elif tool_name == "get_page_context":
            max_chars = 25000
        else:
            max_chars = _TOOL_MSG_MAX_CHARS

        # Small messages don't need processing, except search_documents which we always format to plain text
        if len(msg.content) <= max_chars and tool_name != "search_documents":
            pruned.append(msg)
            continue
        new_content = msg.content

        try:
            data = json.loads(msg.content)

            if isinstance(data, dict):
                if "answer" in data:
                    ans = data.get("answer", "")
                    cits = data.get("citations") or []
                    cit_strings = []
                    for i, c in enumerate(cits, 1):
                        fname = c.get("filename") or "unknown"
                        pg_num = c.get("page")
                        doc_id = c.get("document_id")
                        loc = f" (page {pg_num})" if pg_num is not None else ""
                        did = f" [id: {doc_id}]" if doc_id else ""
                        cit_strings.append(f"[{i}] = {fname}{loc}{did}")
                    
                    content_parts = [f"Search Answer: {ans}"]
                    if cit_strings:
                        content_parts.append("Source Map:")
                        content_parts.extend(cit_strings)
                    new_content = "\n".join(content_parts)
                else:
                    # Unknown JSON format.
                    compact = dict(data)
                    for key in (
                        "chunks",
                        "retrieval_debug",
                        "retrieval_results",
                        "matches",
                        "documents",
                        "raw_results",
                        "scores",
                        "embeddings",
                        "metadata",
                        "debug",
                    ):
                        if key == "documents" and tool_name == "list_documents":
                            continue
                        compact.pop(key, None)

                    new_content = json.dumps(
                        compact,
                        ensure_ascii=False,
                        default=str,
                    )

        except Exception:
            # Not JSON -> use character truncation only if it exceeds limit.
            if len(msg.content) > max_chars:
                new_content = msg.content[:max_chars]
                last_brace = max(
                    new_content.rfind("}"),
                    new_content.rfind("]"),
                )
                if last_brace > 0:
                    new_content = new_content[: last_brace + 1]
                new_content += "\n...[truncated for context efficiency]"
            else:
                new_content = msg.content

        # Final safeguard.
        if len(new_content) > max_chars:
            new_content = (
                new_content[:max_chars]
                + "\n...[truncated for context efficiency]"
            )

        pruned.append(
            ToolMessage(
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            )
        )

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
    clean_history = [
        m for m in (conversation_history or [])
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ]
    max_history = agent_cfg.get("max_history_messages", 20)
    messages += clean_history[-max_history:]

    if len(messages) > 1:  # there IS prior history for this session
        messages.append(SystemMessage(
            "Prior conversation history is provided above. If the next user message is a short "
            "follow-up or ambiguous request (e.g., 'model name', 'what about it?'), resolve its "
            "context against the prior conversation history."
        ))
    messages.append(HumanMessage(message))


    graph = _build_graph(llm, tool_schemas, registry, write_tools, clarify_tools,
                          max_iterations, is_question, config, agent_cfg, session_id=session_id)

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
            "guard_blocked": False,
            "safe_answer": None,
            "guard_evidences": [],
            "guard_risk_score": 0,
            "guard_policy": "allow",
        })

    token_usage = sink.totals()
    calls_log = sink.get_calls_log()
    if calls_log:
        try:
            from backend.storage.postgres_store import PostgresStore
            pg = PostgresStore()
            try:
                pg.write_llm_calls(document_id=None, calls=calls_log, session_id=session_id)
            finally:
                pg.close()
        except Exception as exc:
            logger.warning("Failed to persist agent llm_calls to Postgres: %s", exc)

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
            "llm_calls": calls_log,
            "messages": final_state["messages"],
            "token_usage": token_usage,
            "trace_id": trace_info["trace_id"],
        }
    if final_state.get("pending_approval"):
        return {
            "status": "needs_approval",
            "pending": final_state["pending_approval"],
            "tool_calls": tool_calls,
            "llm_calls": calls_log,
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

        search_answers = []
        for m in final_state["messages"]:
            if isinstance(m, ToolMessage):
                try:
                    r = json.loads(m.content)
                    if isinstance(r, dict) and r.get("answer", "").strip():
                        search_answers.append(r["answer"])
                except Exception:
                    if m.content.startswith("Search Answer:"):
                        lines = m.content.splitlines()
                        ans = lines[0].replace("Search Answer:", "").strip()
                        if ans:
                            search_answers.append(ans)

        unique_answers = list(set(search_answers))
        if len(unique_answers) == 1:
            answer = unique_answers[0]
            logger.info("Recovered answer from prior tool result (fast-path fallback)")
        # else: len > 1 -> fall through to slow-path LLM synthesis below

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
                model_name, provider_name = resolve_model_provider(config, agent_cfg)
                usage.record_from_message("agent_fallback", fallback_response, prompt=messages_for_fallback, model=model_name, provider=provider_name)
            except Exception as exc:
                logger.warning("Agent fallback LLM invocation failed: %s", exc)

        if not answer.strip():
            answer = (
                "I searched the documents but couldn't produce a complete answer within "
                "the available steps. Please try rephrasing your question or naming the "
                "specific document you want to search."
            )
    # Map to final safe/redacted answer if set by guards, otherwise fallback
    final_answer = final_state.get("safe_answer")
    if final_answer is None or not final_state.get("guard_blocked"):
        final_answer = final_answer or answer

    return {
        "status": "done",
        "answer": final_answer,
        "tool_calls": tool_calls,
        "llm_calls": calls_log,
        "messages": final_state["messages"],
        "token_usage": token_usage,
        "trace_id": trace_info["trace_id"],
        "guard_risk_score": final_state.get("guard_risk_score", 0),
        "guard_policy": final_state.get("guard_policy", "allow"),
    }
