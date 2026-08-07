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

import contextlib
import contextvars
import json
import logging
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from backend.agent.intent_classifier import classify_intent
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

# ---------------------------------------------------------------------------
# Execution-trace sink — context-var based collector so every node appends
# directly without relying on LangGraph's reducer (which proved unreliable).
# Same pattern as usage.using_sink().
# ---------------------------------------------------------------------------
_trace_sink_var: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "execution_trace_sink", default=None
)

@contextlib.contextmanager
def _using_trace_sink():
    sink: list[dict] = []
    token = _trace_sink_var.set(sink)
    try:
        yield sink
    finally:
        _trace_sink_var.reset(token)

def _append_trace(item: dict) -> None:
    """Append a step dict to the active trace sink (no-op if no sink is active)."""
    sink = _trace_sink_var.get()
    if sink is not None:
        sink.append(item)

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
    # short-circuit flag: set True by tools_node when search_documents returned a
    # complete answer so route_after_tools skips the redundant agent Turn 2 LLM call
    search_shortcircuit: bool


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
    "- search_documents(query, document_scope?): Search ingested docs. "
    "Pass the user's question as `query`. Pass `document_scope` (array of doc ids or filenames) "
    "ONLY when the user explicitly quotes a real filename or document_id that you have seen in a "
    "list_documents result — NEVER invent, guess, infer, or make up a filename. If the user does not explicitly specify a file, leave document_scope null.\n"
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
    "      3. NEVER call `pd.read_excel()`, `pd.read_csv()`, or any file-reading function inside the `code` block. The file is already loaded for you. Use `df` (the active sheet) or `dfs` (dict of all sheets) directly — they are injected into your namespace before your code runs.\n"
    "      4. To list all sheets, set sheet_name='all' and use `result = list(dfs.keys())`.\n"
    "      5. To inspect columns/rows of sheet 'Vendor A', use `dfs['Vendor A'].columns.tolist()` or `dfs['Vendor A'].iloc[:5]`.\n"
    "      6. Work step-by-step: first list the sheet names, then inspect columns/rows of relevant sheets to find headers, then perform the final calculation.\n"
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
    "   CRITICAL: If the user mentions 'data sheets', 'spreadsheet', 'excel', or asks a question that clearly targets tabular data, "
    "you MUST use `excel_tool`. Do NOT call `search_documents` for these questions. RAG/Search retrieves raw text chunks which causes "
    "poor computational accuracy and UI citation issues. First resolve the Excel file name using list_documents/sql_read if needed, then "
    "run your computational/filtering code entirely inside excel_tool.\n"
    "   UI BEHAVIOR WARNING: If you can answer a question fully using `excel_tool`, DO NOT redundantly call `search_documents` "
    "to look for text matches. The UI will prioritize and show PDF text citations over Excel data. Rely purely on `excel_tool` for Excel data.\n\n"

    "## FILENAME RESTRICTIONS\n"
    "If the user query or conversation history mentions a specific file name "
    "(e.g., 'major-08.pptx'), you MUST restrict your search strictly to that document "
    "by passing its filename or UUID in the `document_scope` parameter of `search_documents`. "
    "Never search the entire corpus when a specific file is targeted. Conversely, if no explicit file is named, YOU MUST NEVER supply a document_scope.\n\n"

    "## AFTER AN INGEST\n"
    "When the user attaches a file: ingest it first. Then if they ask a question "
    "about it, search restricted ONLY to that document's id or filename.\n\n"

    "## DISAMBIGUATION\n"
    "If several documents plausibly match and the user hasn't said which, call "
    "list_documents to see the candidates, then call request_clarification with the "
    "question and the candidate filenames as options — do NOT guess or pick one silently.\n\n"

    "## MULTI-STEP\n"
    "- You MAY chain tools when a task needs it — e.g. list_documents to find a file, "
    "then search_documents scoped to it; several searches for a multi-part question; "
    "or search_documents followed by get_page_context on a fragmented result. "
    "Work step by step.\n"
    "- STRICT NO-REPEAT RULE: If a tool call (same tool name + same arguments) has already been made "
    "in this turn and returned a result, you MUST NOT call it again with the same arguments. "
    "If you are stuck, stop and give the best answer you have with what was found so far.\n"
    "- CRITICAL FALLBACK (If search_documents returns 'I could not find this...'):\n"
    "  A. If the question is PROCEDURAL (how-to, steps, replacing, installing, operating, safety):\n"
    "     - Try ONE broader search_documents with simpler keywords (e.g. 'workpiece holder' instead of 'replace workpiece holder cylinder grinder').\n"
    "     - NEVER use excel_tool for procedural/how-to questions. If it still fails, state plainly that it is not in the documents.\n"
    "  B. If the question asks for specific TABULAR DATA (part numbers, drawing numbers, quantities, prices, serials):\n"
    "     - Call list_documents() to see all ingested files.\n"
    "     - For any .xlsx/.xls/.csv files found, use excel_tool to query them.\n"
    "     - Do NOT immediately return a refusal — check the spreadsheets first!\n\n"

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

_FAST_GREETINGS = {
    # Basic Greetings
    "hi", "hello", "hey", "heyy", "hiii", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
    # Conversational Greetings
    "how are you", "how are you doing", "hows it going", "what's up", "sup",
    "hi how are you", "hello how are you", "hey how are you",
    # Thanks & Sign-offs
    "thanks", "thank you", "thx", "thanks a lot",
    "bye", "goodbye", "see ya", "have a good day",
}

_QUERY_KEYWORDS = {
    "search", "find", "what", "where", "how", "show", "spec", "data", "file", "document",
    "pdf", "excel", "sheet", "table", "calculate", "mounting", "instructions", "parameter",
    "cost", "price", "serial", "code", "drawing", "list", "ingest", "database", "sql"
}

GREETING_SYSTEM_PROMPT = (
    "You are a polite AI assistant for an enterprise document search system. "
    "Reply naturally, warmly, and concisely (in 1 short sentence) to the user's greeting. "
    "Briefly offer to help them search their ingested documents."
)

# Appended to SYSTEM_PROMPT when the intent classifier says this turn does NOT need
# the corpus (follow_up / general). Without it the ABSOLUTE MANDATE above still
# orders the model to search every time, and relaxing tool_choice alone has no
# visible effect. Placed last so it wins on recency, and it only PERMITS a direct
# answer — the model may still search if it turns out it needs to.
DIRECT_ANSWER_OVERRIDE = (
    "\n\n## THIS TURN: DIRECT ANSWER PERMITTED (overrides the ABSOLUTE MANDATE above)\n"
    "This message was classified as NOT needing the document corpus — it is either "
    "chit-chat/general knowledge, or a follow-up answerable from the conversation above.\n"
    "For THIS turn only you may answer directly, without calling a tool.\n"
    "- Follow-up: answer from the conversation above; do not re-search for something "
    "already retrieved.\n"
    "- General knowledge/reasoning: answer from your own knowledge, concisely.\n"
    "If answering actually requires the user's documents, call search_documents as "
    "normal — this permits a direct answer, it does not forbid searching."
)


def _is_greeting(text: str) -> bool:
    clean = text.strip().lower().rstrip("!.,")
    return clean in _GREETINGS


def _is_pure_fast_greeting(text: str) -> bool:
    """Check if message is a pure greeting (<=4 words, greeting term, no search keywords)."""
    clean = text.strip().lower().rstrip("!.,?").strip()
    if clean in _FAST_GREETINGS:
        return True
    words = clean.split()
    if len(words) <= 4:
        has_greeting = any(
            w in clean for w in ("hi", "hello", "hey", "morning", "evening", "thanks", "thank", "bye")
        )
        has_query_kw = any(w in clean for w in _QUERY_KEYWORDS)
        if has_greeting and not has_query_kw:
            return True
    return False


_GENERIC_DOC_TERMS = {
    # Document-related terms
    "document", "file", "pdf", "explain", "about", "this", "that", "the", "what", "is", "for", "in",
    "and", "or", "its", "our",
    # Common English stop words that appear in any document's page1 text
    "can", "you", "are", "was", "has", "had", "have", "been", "will", "not",
    "all", "any", "but", "with", "from", "use", "used", "using", "may", "also",
    "they", "their", "your", "more", "each", "both", "when", "how", "than",
    "get", "set", "per", "out", "one", "two", "new", "see",
}

_SCOPE_RESET_PHRASES = {
    "search all", "all documents", "all manuals", "across all files", "check everything",
    "entire corpus", "all files", "unscope", "not just that", "clear scope", "reset scope"
}


def _resolve_turn_document_scope(
    message: str, session_id: str, viewer_doc_id: str | None, config: dict
) -> tuple[list[str] | str | None, str | None, bool]:
    """5-Tier Priority Document Scope Resolver:
    Returns (resolved_scope, active_filename, is_ambiguous_trigger)
    - resolved_scope: list[str] | str | None for document_scope
    - active_filename: human-readable name of active file
    - is_ambiguous_trigger: True if agent MUST call request_clarification
    """
    import difflib
    from backend.storage.conversation_store import PostgresConversationStore
    from backend.storage.postgres_store import PostgresStore

    msg_lower = message.strip().lower()

    # Tier 1: Scope Clear / Reset Check
    if any(phrase in msg_lower for phrase in _SCOPE_RESET_PHRASES):
        if session_id:
            PostgresConversationStore().set_session_active_doc(session_id, None)
        return None, None, False

    # Fetch available documents with metadata
    try:
        pg = PostgresStore(config=config)
        try:
            docs = pg.list_documents_with_metadata()
        finally:
            pg.conn.close()
    except Exception as exc:
        logger.warning("_resolve_turn_document_scope DB fetch failed: %s", exc)
        return None, None, False

    # Extract non-generic tokens from user prompt (min 4 chars to avoid "can", "you", etc.)
    prompt_tokens = [
        t for t in re.findall(r"\b[a-zA-Z0-9_-]{4,}\b", msg_lower)
        if t not in _GENERIC_DOC_TERMS
    ]

    # If no meaningful tokens remain, skip Tier 2 matching entirely
    if not prompt_tokens:
        pass  # Fall through to Tier 3 (viewer) / Tier 4 (session state)


    # Tier 2: Explicit Mention (Dual-Algorithm Matching)
    matched_doc_ids = []
    doc_id_to_name = {d["document_id"]: d["filename"] for d in docs}

    if prompt_tokens:
        for doc in docs:
            doc_id = doc["document_id"]
            fname = (doc["filename"] or "").lower()
            page1 = (doc.get("page1_text") or "").lower()
            doctype = (doc.get("document_type") or "").lower()
            industry = (doc.get("industry") or "").lower()

            # Algorithm A: difflib SequenceMatcher on clean filename tokens
            fname_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", fname) if t not in _GENERIC_DOC_TERMS]
            score = 0.0
            if fname_tokens and prompt_tokens:
                ratio = difflib.SequenceMatcher(None, " ".join(prompt_tokens), " ".join(fname_tokens)).ratio()
                if ratio >= 0.70:
                    score = ratio

            # Algorithm B: Exact non-generic token containment in page1_text / metadata
            if score < 0.70:
                meta_text = f"{fname} {doctype} {industry} {page1}".lower()
                meta_token_set = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", meta_text))
                matches = sum(1 for pt in prompt_tokens if pt in meta_token_set)
                if matches > 0 and len(prompt_tokens) > 0:
                    containment_score = matches / len(prompt_tokens)
                    if containment_score >= 0.70 or matches >= 2:
                        score = 0.85

            if score >= 0.70:
                matched_doc_ids.append(doc_id)

    if len(matched_doc_ids) == 1:
        doc_id = matched_doc_ids[0]
        if session_id:
            PostgresConversationStore().set_session_active_doc(session_id, doc_id)
        return doc_id, doc_id_to_name.get(doc_id), False
    elif len(matched_doc_ids) > 1:
        # Multi-doc comparison query ("compare Operation and Maintenance manual")
        return matched_doc_ids, ", ".join([doc_id_to_name[i] for i in matched_doc_ids if i in doc_id_to_name]), False

    # Tier 3: Frontend Viewer Active Document
    if viewer_doc_id and viewer_doc_id in doc_id_to_name:
        if session_id:
            PostgresConversationStore().set_session_active_doc(session_id, viewer_doc_id)
        return viewer_doc_id, doc_id_to_name[viewer_doc_id], False

    # Tier 4: Postgres Stored Session State
    stored_doc_id = PostgresConversationStore().get_session_active_doc(session_id) if session_id else None
    if stored_doc_id and stored_doc_id in doc_id_to_name:
        return stored_doc_id, doc_id_to_name[stored_doc_id], False

    # Tier 5: Ambiguity Trigger (Deictic references without active context)
    has_deictic = any(
        term in msg_lower for term in ("this manual", "this document", "this file", "this pdf", "what is this about")
    )
    if has_deictic and len(docs) > 1 and not matched_doc_ids:
        return None, None, True

    return None, None, False


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
        import time
        t0 = time.time()
        if not g_cfg.get("enabled", True):
            dur_ms = round((time.time() - t0) * 1000, 2)
            trace_item = {
                "step": "Input Guardrail", "type": "guardrail", "status": "SKIPPED",
                "risk_score": 0, "policy": "allow", "duration_ms": dur_ms, "details": "Guardrails disabled"
            }
            logger.info("🛡️  [STEP: Input Guardrail] Disabled -> SKIPPED (%.1fms)", dur_ms)
            _append_trace(trace_item)
            return {
                "guard_blocked": False, "guard_evidences": [], "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        rollout_pct = g_cfg.get("rollout", {}).get("input_guard_pct", 100)
        if not should_apply_guard(rollout_pct, session_id):
            dur_ms = round((time.time() - t0) * 1000, 2)
            trace_item = {
                "step": "Input Guardrail", "type": "guardrail", "status": "SKIPPED",
                "risk_score": 0, "policy": "allow", "duration_ms": dur_ms, "details": "Rollout excluded"
            }
            logger.info("🛡️  [STEP: Input Guardrail] Rollout excluded -> SKIPPED (%.1fms)", dur_ms)
            _append_trace(trace_item)
            return {
                "guard_blocked": False, "guard_evidences": [], "guard_risk_score": 0,
                "guard_policy": "allow",
            }

        raw_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                raw_msg = msg.content
                break

        if not raw_msg:
            dur_ms = round((time.time() - t0) * 1000, 2)
            return {
                "guard_blocked": False, "guard_evidences": [], "guard_risk_score": 0,
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
        dur_ms = round((time.time() - t0) * 1000, 2)
        is_blocked = policy_decision == PolicyDecision.BLOCK or should_block_session

        logger.info(
            "🛡️  [STEP: Input Guardrail] Risk Score: %s | Policy: %s | Status: %s | Duration: %.1fms",
            decision.risk_score, policy_decision.value.upper(), "BLOCKED" if is_blocked else "PASS", dur_ms
        )

        trace_item = {
            "step": "Input Guardrail", "type": "guardrail",
            "status": "BLOCKED" if is_blocked else "PASS",
            "risk_score": decision.risk_score, "policy": policy_decision.value,
            "duration_ms": dur_ms, "events": [decision.event_type] if decision.event_type else []
        }

        if is_blocked:
            blocked_msg = AIMessage(
                content="Your request contains triggers that violate our safety policies."
            )
            _append_trace(trace_item)
            return {
                "messages": [blocked_msg],
                "guard_blocked": True,
                "guard_evidences": [ev.__dict__],
                "guard_risk_score": decision.risk_score,
                "guard_policy": policy_decision.value,
                "safe_answer": blocked_msg.content,
            }

        _append_trace(trace_item)
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
        import time
        t0 = time.time()
        iters = state.get("iterations", 0)
        active_llm = llm_required if (iters == 0 and is_question) else llm_auto
        
        pruned_messages = _prune_messages_for_llm(state["messages"])
        response = _invoke_with_retry(active_llm, pruned_messages)
        dur_ms = round((time.time() - t0) * 1000, 2)

        model_name, provider_name = resolve_model_provider(config, agent_cfg)
        usage.record_from_message("agent", response, prompt=pruned_messages, model=model_name, provider=provider_name)

        # Extract token usage metadata from response if available
        um = getattr(response, "usage_metadata", None) or {}
        in_tok = um.get("input_tokens", 0) or len(str(pruned_messages)) // 4
        out_tok = um.get("output_tokens", 0) or len(str(response.content)) // 4
        tot_tok = um.get("total_tokens", in_tok + out_tok)

        calls_requested = [c["name"] for c in getattr(response, "tool_calls", []) or []]
        decision_str = f"Tools: {calls_requested}" if calls_requested else "Direct Text Answer"

        logger.info(
            "🧠 [STEP: Agent LLM Call (Turn %d)] Provider: %s | Model: %s | Tokens: %d (In: %d, Out: %d) | Choice: %s | Latency: %.1fms",
            iters + 1, provider_name, model_name, tot_tok, in_tok, out_tok, decision_str, dur_ms
        )

        trace_item = {
            "step": f"Agent LLM Planner (Turn {iters + 1})",
            "type": "llm_call",
            "provider": provider_name,
            "model": model_name,
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": tot_tok,
            "duration_ms": dur_ms,
            "tool_calls_requested": calls_requested,
            "decision": decision_str
        }

        _append_trace(trace_item)
        return {"messages": [response], "iterations": iters + 1}

    def tools_node(state: AgentState) -> dict:
        import time
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        pending: list[dict] = []
        clarification: dict | None = None
        new_evidences: list[dict] = []
        step_traces: list[dict] = []
        seen_calls: set[str] = set()

        for call in getattr(last, "tool_calls", None) or []:
            t0 = time.time()
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

            call_key = f"{name}::{json.dumps(args, sort_keys=True)}"
            if call_key in seen_calls:
                dur_ms = round((time.time() - t0) * 1000, 2)
                tool_messages.append(ToolMessage(
                    content=json.dumps({"error": "Duplicate call detected — this exact tool call was already made in this turn. Use the previous result and stop retrying."}, default=str),
                    tool_call_id=call_id, name=name,
                ))
                step_traces.append({
                    "step": f"Tool: {name}", "type": "tool_execution",
                    "tool_name": name, "args": args, "duration_ms": dur_ms, "status": "duplicate_skipped"
                })
                continue
            seen_calls.add(call_key)

            if name in clarify_tools:
                if clarification is None:
                    clarification = {
                        "question": args.get("question") or "Which option?",
                        "options": args.get("options") or [],
                    }
                tool_messages.append(ToolMessage(
                    content="awaiting user selection", tool_call_id=call_id, name=name,
                ))
                dur_ms = round((time.time() - t0) * 1000, 2)
                step_traces.append({
                    "step": f"Tool: {name}", "type": "tool_execution",
                    "tool_name": name, "args": args, "duration_ms": dur_ms, "status": "clarification_requested"
                })
                continue

            if name == "ingest_document":
                import os
                file_path = args.get("file_path", "")
                
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

                resolved_path = file_path
                if db_doc and db_doc["file_path"] and os.path.isfile(db_doc["file_path"]):
                    resolved_path = db_doc["file_path"]
                    args["file_path"] = resolved_path

                if db_doc and db_doc["status"] == "ready" and os.path.isfile(resolved_path):
                    dur_ms = round((time.time() - t0) * 1000, 2)
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "document_id": db_doc["document_id"],
                            "status": "ready",
                            "message": f"Document {fname!r} is already ingested and ready for query."
                        }, default=str),
                        tool_call_id=call_id, name=name,
                    ))
                    step_traces.append({
                        "step": f"Tool: {name}", "type": "tool_execution",
                        "tool_name": name, "args": args, "duration_ms": dur_ms, "status": "already_ready"
                    })
                    continue

                if not resolved_path or not os.path.isfile(resolved_path):
                    dur_ms = round((time.time() - t0) * 1000, 2)
                    tool_messages.append(ToolMessage(
                        content=json.dumps({
                            "error": f"FileNotFoundError: File {file_path!r} does not exist. "
                                     f"Please upload/attach the file in the chat input first."
                        }, default=str),
                        tool_call_id=call_id, name=name,
                    ))
                    step_traces.append({
                        "step": f"Tool: {name}", "type": "tool_execution",
                        "tool_name": name, "args": args, "duration_ms": dur_ms, "status": "file_not_found"
                    })
                    continue

            if name in write_tools and not (
                state.get("approved_writes") and _is_approved(name, args, state.get("approved_calls"))
            ):
                dur_ms = round((time.time() - t0) * 1000, 2)
                pending.append({"id": call_id, "name": name, "args": args})
                tool_messages.append(ToolMessage(
                    content="blocked: this action writes data and needs human approval before it can run.",
                    tool_call_id=call_id, name=name,
                ))
                step_traces.append({
                    "step": f"Tool: {name}", "type": "tool_execution",
                    "tool_name": name, "args": args, "duration_ms": dur_ms, "status": "pending_user_approval"
                })
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
                
            dur_ms = round((time.time() - t0) * 1000, 2)
            args_str = json.dumps(args, default=str)
            if len(args_str) > 120:
                args_str = args_str[:120] + "..."
            
            res_str = json.dumps(cleaned_result, default=str)
            if len(res_str) > 150:
                res_summary = res_str[:150] + "..."
            else:
                res_summary = res_str

            logger.info(
                "🛠️  [STEP: Tool Executed] Tool: %s | Args: %s | Duration: %.1fms",
                name, args_str, dur_ms
            )

            step_traces.append({
                "step": f"Tool: {name}", "type": "tool_execution",
                "tool_name": name, "args": args, "duration_ms": dur_ms,
                "status": "success" if not (isinstance(cleaned_result, dict) and "error" in cleaned_result) else "error",
                "output_summary": res_summary
            })

            tool_messages.append(ToolMessage(
                content=json.dumps(cleaned_result, default=str), tool_call_id=call_id, name=name,
            ))

        for t in step_traces:
            _append_trace(t)

        # --- Search short-circuit: skip agent Turn 2 if search_documents was the
        # ONLY tool called this turn and it returned a complete non-refusal answer.
        # Saves ~6-7s and ~4000 tokens per standard question. ---
        shortcircuit = False
        all_calls = getattr(last, "tool_calls", None) or []
        if (
            len(all_calls) == 1
            and all_calls[0]["name"] == "search_documents"
            and not pending
            and not clarification
            and tool_messages
        ):
            try:
                result_data = json.loads(tool_messages[-1].content)
                search_answer = (result_data.get("answer") or "").strip()
                _REFUSAL_HINTS = (
                    "could not find this in the provided",
                    "no relevant passages found",
                    "not in the provided documents",
                )
                is_refusal = any(h in search_answer.lower() for h in _REFUSAL_HINTS)
                is_error = search_answer.lower().startswith("error:")
                if search_answer and not is_refusal and not is_error:
                    # Inject a synthetic AIMessage so output_guard_node can find it
                    tool_messages.append(AIMessage(content=search_answer))
                    shortcircuit = True
                    logger.info(
                        "⚡ [SHORT-CIRCUIT] search_documents returned complete answer — "
                        "skipping agent Turn 2 LLM call"
                    )
                    _append_trace({
                        "step": "Search Short-Circuit", "type": "routing",
                        "details": "search_documents answer injected directly, agent Turn 2 skipped"
                    })
            except Exception:
                pass  # JSON parse failed — fall through to normal agent Turn 2

        return {
            "messages": tool_messages,
            "pending_approval": pending or None,
            "clarification": clarification,
            "guard_evidences": new_evidences,
            "search_shortcircuit": shortcircuit,
        }


    def output_guard_node(state: AgentState) -> dict:
        import time
        t0 = time.time()
        last = state["messages"][-1]
        raw_answer = clean_message_content(last.content) if isinstance(last, AIMessage) else ""
        
        if not g_cfg.get("enabled", True):
            dur_ms = round((time.time() - t0) * 1000, 2)
            trace_item = {
                "step": "Output Guardrail", "type": "guardrail", "status": "SKIPPED",
                "risk_score": 0, "policy": "allow", "duration_ms": dur_ms, "details": "Guardrails disabled"
            }
            logger.info("🛡️  [STEP: Output Guardrail] Disabled -> SKIPPED (%.1fms)", dur_ms)
            _append_trace(trace_item)
            return {
                "safe_answer": raw_answer,
            }

        raw_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw_msg = msg
                break

        if raw_msg is None:
            dur_ms = round((time.time() - t0) * 1000, 2)
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
        dur_ms = round((time.time() - t0) * 1000, 2)

        logger.info(
            "🛡️  [STEP: Output Guardrail] Risk Score: %s | Policy: %s | Status: %s | Duration: %.1fms",
            decision.risk_score, policy_decision.value.upper(), "BLOCKED" if blocked else "PASS", dur_ms
        )

        trace_item = {
            "step": "Output Guardrail", "type": "guardrail",
            "status": "BLOCKED" if blocked else "PASS",
            "risk_score": decision.risk_score, "policy": policy_decision.value,
            "duration_ms": dur_ms, "sanitized": bool(decision.sanitized_value)
        }

        _append_trace(trace_item)
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
        if state.get("search_shortcircuit"):
            return "output_guard"  # skip agent Turn 2 — answer already in messages
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
        seen_calls: set[str] = set()

        for call in getattr(last, "tool_calls", None) or []:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]

            call_key = f"{name}::{json.dumps(args, sort_keys=True)}"
            if call_key in seen_calls:
                tool_messages.append(ToolMessage(
                    content=json.dumps({"error": "Duplicate call detected — this exact tool call was already made in this turn. Use the previous result and stop retrying."}, default=str),
                    tool_call_id=call_id, name=name,
                ))
                continue
            seen_calls.add(call_key)

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

        # Search short-circuit (async path — same logic as sync _build_graph)
        shortcircuit = False
        all_calls = getattr(last, "tool_calls", None) or []
        if (
            len(all_calls) == 1
            and all_calls[0]["name"] == "search_documents"
            and not pending
            and not clarification
            and tool_messages
        ):
            try:
                result_data = json.loads(tool_messages[-1].content)
                search_answer = (result_data.get("answer") or "").strip()
                _REFUSAL_HINTS = (
                    "could not find this in the provided",
                    "no relevant passages found",
                    "not in the provided documents",
                )
                is_refusal = any(h in search_answer.lower() for h in _REFUSAL_HINTS)
                is_error = search_answer.lower().startswith("error:")
                if search_answer and not is_refusal and not is_error:
                    tool_messages.append(AIMessage(content=search_answer))
                    shortcircuit = True
            except Exception:
                pass

        return {
            "messages": tool_messages,
            "pending_approval": pending or None,
            "clarification": clarification,
            "guard_evidences": new_evidences,
            "search_shortcircuit": shortcircuit,
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
        if state.get("search_shortcircuit"):
            return "output_guard"  # skip agent Turn 2 — answer already in messages
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
    active_document_id: str | None = None,
    intent_llm=None,
) -> dict:
    """`intent_llm` overrides the model used for intent classification (tests inject
    a scripted one). None -> built from config["query"]["agent"]["intent"]."""
    registry = registry if registry is not None else build_agent_registry()
    agent_cfg = (config.get("query") or {}).get("agent") or {}
    max_iterations = agent_cfg.get("max_iterations", 5)
    write_tools = set(agent_cfg.get("write_tools") or [])
    clarify_tools = set(agent_cfg.get("clarify_tools") or ["request_clarification"])

    if llm is None:
        llm = get_llm_for(config, agent_cfg)

    # 5-Tier Document Scope Resolution
    try:
        resolved_scope, active_fname, is_ambiguous_trigger = _resolve_turn_document_scope(
            message, session_id, active_document_id, config
        )
    except Exception as scope_exc:
        logger.warning("_resolve_turn_document_scope failed: %s", scope_exc)
        resolved_scope, active_fname, is_ambiguous_trigger = None, None, False

    if is_ambiguous_trigger:
        from backend.storage.postgres_store import PostgresStore
        _SEARCHABLE_EXTS = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc", ".txt", ".pptx", ".ppt"}
        pg = PostgresStore(config=config)
        try:
            docs = pg.list_documents()
            seen = set()
            options = []
            for d in docs:
                fname = d.get("filename", "")
                if not fname:
                    continue
                ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in _SEARCHABLE_EXTS:
                    continue
                if fname not in seen:
                    seen.add(fname)
                    options.append(fname)
        finally:
            pg.conn.close()

        clar_question = "Which document would you like to search?"
        logger.info("⚡ [AMBIGUITY FAST-PATH] Returning request_clarification for ambiguous prompt")
        return {
            "status": "needs_clarification",
            "question": clar_question,
            "options": options,
            "answer": clar_question,
            "tool_calls": [{"id": "clarify_1", "name": "request_clarification", "args": {"question": clar_question, "options": options}}],
            "execution_trace": [],
            "messages": [HumanMessage(message)],
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "trace_id": None,
        }

    tool_schemas = [_to_openai_tool(t) for t in registry.values()]

    # What does this turn actually need? The label decides ONE thing: whether turn 0
    # forces tool_choice="required". Replaces the old `not _is_greeting(message)`,
    # which forced a search for every non-greeting — including "what is 2+2?" and
    # "summarise that". Fails open to tools-required, so a classifier outage just
    # restores the previous behaviour. Pure greetings already returned via the
    # fast-path above and never reach here.
    intent_result = classify_intent(
        message,
        conversation_history=conversation_history,
        config=config,
        agent_cfg=agent_cfg,
        llm=intent_llm,
    )
    is_question = intent_result.requires_tools

    system_prompt_text = SYSTEM_PROMPT
    if not is_question:
        # Relax the ALWAYS-SEARCH mandate for this turn only (see the constant).
        system_prompt_text += DIRECT_ANSWER_OVERRIDE
    if is_ambiguous_trigger:
        system_prompt_text += (
            "\n\n## AMBIGUITY DISAMBIGUATION MANDATE\n"
            "- The user asks an ambiguous question ('this manual', 'what is this about') but NO document is currently "
            "active or specified, and multiple documents exist in the corpus. You are STRICTLY PROHIBITED from guessing "
            "or doing an un-scoped search. You MUST call request_clarification to ask the user which manual they mean."
        )

    messages: list[BaseMessage] = [SystemMessage(system_prompt_text)]
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


    import time
    turn_t0 = time.time()
    logger.info("\n================================================================================")
    logger.info("[TURN START] Question: %r", message[:120])
    logger.info("--------------------------------------------------------------------------------")

    # --- Fast-Path: Pure User Greeting Bypass ---
    if _is_pure_fast_greeting(message):
        logger.info("⚡ [FAST GREETING] Pure greeting detected — executing 1-line greeting LLM call")
        with _using_trace_sink() as _trace_sink, traced_request(
            "agent_chat", input=message,
            metadata={"session_id": session_id, "approved_writes": False},
        ) as trace_info, usage.using_sink() as sink:
            greeting_messages = [
                SystemMessage(GREETING_SYSTEM_PROMPT),
                HumanMessage(message),
            ]
            response = llm.invoke(greeting_messages, config={"max_tokens": 40})
            model_name, provider_name = resolve_model_provider(config, agent_cfg)
            usage.record_from_message("fast_greeting", response, prompt=greeting_messages, model=model_name, provider=provider_name)

            raw_answer = clean_message_content(response.content).strip()
            _append_trace({
                "step": "Fast Greeting LLM Call",
                "type": "llm_call",
                "provider": provider_name,
                "model": model_name,
                "duration_ms": round((time.time() - turn_t0) * 1000, 2),
                "decision": "Direct Dynamic Greeting"
            })

        token_usage = sink.totals(config=config)
        calls_log = sink.get_calls_log()
        execution_trace = list(_trace_sink)
        total_turn_sec = round(time.time() - turn_t0, 2)
        logger.info(
            "✅ [TURN COMPLETE - FAST GREETING] Latency: %.2fs | Total Tokens: %d",
            total_turn_sec, token_usage.get("total_tokens", 0)
        )
        logger.info("================================================================================\n")

        if calls_log:
            try:
                from backend.storage.postgres_store import PostgresStore
                pg = PostgresStore()
                try:
                    pg.write_llm_calls(document_id=None, calls=calls_log, session_id=session_id)
                finally:
                    pg.close()
            except Exception as db_exc:
                logger.warning("Failed to persist fast_greeting llm_calls to Postgres: %s", db_exc)

        return {
            "status": "done",
            "answer": raw_answer,
            "tool_calls": [],
            "llm_calls": calls_log,
            "execution_trace": [],
            "messages": [HumanMessage(message), AIMessage(content=raw_answer)],
            "token_usage": token_usage,
            "trace_id": trace_info.get("trace_id"),
            "guard_risk_score": 0,
            "guard_policy": "allow",
        }

    graph = _build_graph(llm, tool_schemas, registry, write_tools, clarify_tools,
                          max_iterations, is_question, config, agent_cfg, session_id=session_id)

    # ONE root span + ONE token-usage sink for the whole turn — every LLM call
    # and tool dispatch below (however deep, e.g. search_documents' internal
    # query_planner/retrieval/answerer LLM calls) nests under this single trace
    # and accumulates into this single usage total. See tracing.py / usage.py.
    with _using_trace_sink() as _trace_sink, traced_request(
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
            "search_shortcircuit": False,
        })

    token_usage = sink.totals(config=config)
    calls_log = sink.get_calls_log()
    execution_trace = list(_trace_sink)
    total_turn_sec = round(time.time() - turn_t0, 2)

    logger.info("--------------------------------------------------------------------------------")
    logger.info(
        "✅ [TURN COMPLETE] Total Latency: %.2fs | Total Tokens: %d (In: %d, Out: %d) | Steps Executed: %d",
        total_turn_sec,
        token_usage.get("total_tokens", 0),
        token_usage.get("input_tokens", 0),
        token_usage.get("output_tokens", 0),
        len(execution_trace)
    )
    logger.info("================================================================================\n")

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
            "execution_trace": execution_trace,
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
            "execution_trace": execution_trace,
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
        has_other_tools = False
        for m in final_state["messages"]:
            if isinstance(m, ToolMessage):
                if m.name != "search_documents":
                    has_other_tools = True
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
        if unique_answers:
            # Rescue the most recent unique search answer even if other tools were called
            answer = unique_answers[-1]
            logger.info("Recovered answer from prior tool result (fast-path fallback)")
        # else: no search answers -> fall through to slow-path LLM synthesis below

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
                t_fb0 = time.time()
                fallback_response = llm.invoke(messages_for_fallback)
                fb_dur_ms = round((time.time() - t_fb0) * 1000, 2)
                answer = clean_message_content(fallback_response.content)
                model_name, provider_name = resolve_model_provider(config, agent_cfg)
                usage.record_from_message("agent_fallback", fallback_response, prompt=messages_for_fallback, model=model_name, provider=provider_name)
                
                um = getattr(fallback_response, "usage_metadata", None) or {}
                in_t = um.get("input_tokens", 0)
                out_t = um.get("output_tokens", 0)
                execution_trace.append({
                    "step": "Fallback Synthesis LLM", "type": "llm_call",
                    "provider": provider_name, "model": model_name,
                    "prompt_tokens": in_t, "completion_tokens": out_t, "total_tokens": in_t + out_t,
                    "duration_ms": fb_dur_ms
                })
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
        "execution_trace": execution_trace,
        "messages": final_state["messages"],
        "token_usage": token_usage,
        "trace_id": trace_info["trace_id"],
        "guard_risk_score": final_state.get("guard_risk_score", 0),
        "guard_policy": final_state.get("guard_policy", "allow"),
        # why this turn did/didn't search — surfaced for debugging + backtesting
        "intent": intent_result.intent,
        "intent_fallback": intent_result.fallback,
    }

