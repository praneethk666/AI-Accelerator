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
    "NEVER call list_documents for content questions or short phrases like 'part no list' or 'model name' — use search_documents instead.\n"
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
    "2. Requests to list ALL ingested/uploaded files — call list_documents instead. Do NOT call this just because the user uses the word 'list' (e.g. 'parts list').\n"
    "3. Requests to ingest a new file — call ingest_document instead.\n"
    "4. Questions requiring calculation, filtering, comparisons, sorting, aggregation, data lookup, or "
    "looking up text notes/inclusions/exclusions/metadata on Excel files (e.g. 'sum column X', "
    "'which company has highest revenue', 'is scaffolding included', 'list exclusions') — call excel_tool instead. "
    "It runs real Pandas code on the actual data and is far more accurate and token-efficient than searching chunked text.\n"
    "   CRITICAL: If the user mentions 'data sheets', 'spreadsheet', 'excel', or asks a question that clearly targets tabular data, "
    "you MUST use `excel_tool`. However, for general 'parts list' queries without explicit mention of Excel, ALWAYS prioritize `search_documents` first. RAG/Search retrieves raw text chunks which causes "
    "poor computational accuracy and UI citation issues for math, but is best for general lookup. First resolve the Excel file name using list_documents/sql_read if needed, then "
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

    "## CATEGORY-BASED QUERY CLASSIFICATION & RESPONSE RULES\n"
    "Categorize the user query into one of the 3 categories below:\n\n"
    "================================================================================\n"
    "CATEGORY 1: PROCESS-BASED QUESTIONS (Interactive Guided Assistant Mode)\n"
    "================================================================================\n"
    "- Triggers: 'how to', 'steps for', 'procedure', 'setup', 'replace', 'clean', 'install', 'changeover', 'maintenance', 'operation'.\n"
    "- Workflow:\n"
    "  1. DISAMBIGUATE: If multiple manuals or sections match, list them as a numbered menu and ask the user to pick one.\n"
    "  2. OVERVIEW: Present a high-level summary of the procedure and ask:\n"
    "     'Here is the procedure for [Topic] from [Document]. We can guide you step-by-step. When you are ready, shall we start the process?'\n"
    "  3. STEP-BY-STEP: Deliver ONLY ONE STEP AT A TIME. Prompt: 'Let me know when this step is complete!'\n"
    "     Wait for user confirmation ('Done', 'Next', 'Ready') before presenting the next step.\n\n"
    "================================================================================\n"
    "CATEGORY 2: CAD / TECHNICAL DRAWING QUESTIONS (CAD & Vision Extraction Mode)\n"
    "================================================================================\n"
    "- Triggers: 'cad drawing', 'schematic', 'circuit diagram', 'sectional view', 'callout', 'dimensions', 'tolerances', 'designators'.\n"
    "- Workflow:\n"
    "  1. Target CAD/circuit drawing documents and CAD chunks (cad_route, large_format).\n"
    "  2. Output precise CAD details: callout item codes (A01, A08), exact part numbers, quantities, dimensions, tolerances, and sheet numbers.\n"
    "  3. Include exact visual citations and sheet references.\n\n"
    "================================================================================\n"
    "CATEGORY 3: DIRECT FACT QUESTIONS (Instant 1-Turn QA Mode)\n"
    "================================================================================\n"
    "- Triggers: Questions asking for specific drawing numbers, part codes, setup quantities, manufacturer/vendor names, or single specs.\n"
    "- Workflow:\n"
    "  1. Output a direct, instant answer in a SINGLE TURN with exact inline citations [Doc, p.X].\n"
    "  2. DO NOT ask for step-by-step confirmation or disambiguation unless explicitly required.\n\n"

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


def _get_filenames_for_ids(doc_ids: list[str], config: dict) -> list[str]:
    """Look up human-readable filenames for a list of document IDs."""
    if not doc_ids:
        return []
    try:
        from backend.storage.postgres_store import PostgresStore
        pg = PostgresStore(config=config)
        try:
            docs = pg.list_documents()
        finally:
            pg.conn.close()
        id_to_name = {str(d["document_id"]): d.get("filename", "") for d in docs}
        return [id_to_name[did] for did in doc_ids if did in id_to_name and id_to_name[did]]
    except Exception:
        return []


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
            logger.info("🛡️  [Input Guardrail] Skipped (disabled)")
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
            logger.info("🛡️  [Input Guardrail] Skipped (rollout excluded)")
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

        if is_blocked:
            logger.info("🛡️  [Input Guardrail] Blocked")
        else:
            logger.info("🛡️  [Input Guardrail] Passed")

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
            logger.info("🛡️  [Output Guardrail] Skipped (disabled)")
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

        if blocked:
            logger.info("🛡️  [Output Guardrail] Blocked")
        else:
            logger.info("🛡️  [Output Guardrail] Passed")

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


def _extract_json_array(text: str) -> list[str]:
    import json
    import re
    text = text.strip()
    match = re.search(r'\[\s*".*?"\s*\]', text, re.DOTALL)
    if not match:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if item]
        except Exception:
            pass
            
    # Fallback to splitting lines if JSON parsing failed but there are numbered lines
    lines = text.splitlines()
    steps = []
    for line in lines:
        cleaned = line.strip()
        match_step = re.match(r'^(?:step\s*\d+[:.]?|\d+[:.])\s*(.*)$', cleaned, re.IGNORECASE)
        if match_step:
            steps.append(match_step.group(1).strip())
        elif cleaned.startswith("-") or cleaned.startswith("*"):
            item = cleaned[1:].strip()
            if item:
                steps.append(item)
    if len(steps) >= 2:
        return steps
    return []


def _extract_json_dict(text: str) -> dict | None:
    """Extract first valid JSON dict from LLM response text."""
    import json
    import re
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group(0))
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


def _extract_checklist_steps_llm(text: str, llm) -> list[str]:
    """Use LLM to extract ALL procedural steps from document text (single call)."""
    if not text.strip():
        return []
    prompt = (
        "You are a precise technical procedure extraction assistant for industrial machinery manuals.\n"
        "Extract EVERY procedural action or instruction from the text below as a JSON list of strings.\n\n"
        "RULES:\n"
        "- Include ALL actionable steps: things the technician must DO\n"
        "  (remove, clean, insert, press, click, turn, loosen, tighten, check, attach, fix, operate, place, etc.)\n"
        "- Include numbered steps (1), (2), 1., 2., 1-1., 1-2. AND plain prose instructions\n"
        "- If a step has <IMPORTANT> or <NOTE> safety info, append it in parentheses at the end of that step\n"
        "- Do NOT include: logo text, image captions starting with 'The image shows', \n"
        "  company info, warranty text, preface, table of contents, revision history\n"
        "- Do NOT include the step number in the output string\n"
        "- Keep each step as a full, meaningful sentence that a technician can act on\n\n"
        f"DOCUMENT TEXT:\n{text}\n\n"
        "Return ONLY a valid JSON array of strings. No markdown fences, no explanation."
    )
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "").strip()
        steps = _extract_json_array(content)
        return [s.strip() for s in steps if s.strip()]
    except Exception as e:
        logger.warning("Failed to extract checklist steps via LLM: %s", e)
    return []


def _format_callout_headers(text: str) -> str:
    """Format safety callout headers so browser HTML renderers do not strip <IMPORTANT> or <NOTE> as unknown HTML tags."""
    if not text:
        return text
    text = re.sub(r'<\s*IMPORTANT\s*>', '\n\n**⚠️ IMPORTANT:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*NOTE\s*>', '\n\n**📌 NOTE:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*WARNING\s*>', '\n\n**🚨 WARNING:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*CAUTION\s*>', '\n\n**⚠️ CAUTION:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*ONE\s*POINT\s*>', '\n\n**💡 ONE POINT:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*DANGER\s*>', '\n\n**🚨 DANGER:**\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<([A-Z0-9\s]{2,20})>', r'\n\n**[\1]:**\n', text)
    return text.strip()


def _scan_pages_for_sections(blocks: list[dict], llm=None) -> list[dict]:
    """Scan document blocks page-by-page to detect sub-section headings.

    Phase 1: fast regex pass (0 LLM calls).
    Phase 2: If regex finds nothing, send each page to LLM one-by-one for confirmation.

    Returns list of {"title": "...", "page_start": N, "page_end": M, "steps": []}.
    """
    if not blocks:
        return []

    # ── Fast regex pass ──────────────────────────────────────────────────────
    sections_found: list[dict] = []
    seen_nums: set[str] = set()

    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide") or 1
        try:
            pg_int = int(pg)
        except (ValueError, TypeError):
            pg_int = 1

        raw_t = (b.get("text") or "").strip()
        if not raw_t or "..." in raw_t:
            continue

        m = re.search(r'^\s*(\d+\.\d+)\s+([A-Za-z0-9\s\-\(\)\/\,\:\.]{3,80})', raw_t)
        if m:
            sec_num = m.group(1)
            title_part = m.group(2).strip().split("\n")[0].strip()
            full_title = re.sub(r'\s+', ' ', f"{sec_num} {title_part}").strip()
            if sec_num not in seen_nums:
                seen_nums.add(sec_num)
                sections_found.append({
                    "title": full_title,
                    "sec_num": sec_num,
                    "page_start": pg_int,
                    "page_end": pg_int,
                    "steps": []
                })

    # ── LLM fallback: only when regex found nothing ───────────────────────────
    if not sections_found and llm and blocks:
        pages_by_num: dict[int, list[str]] = {}
        for b in blocks:
            ref = b.get("source_ref") or {}
            pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
            if not pg:
                continue
            try:
                pg_int = int(pg)
            except (ValueError, TypeError):
                continue
            raw_t = (b.get("text") or "").strip()
            if raw_t:
                pages_by_num.setdefault(pg_int, []).append(raw_t)

        for pg_int in sorted(pages_by_num.keys()):
            page_text = "\n".join(pages_by_num[pg_int])[:3000]
            if not page_text:
                continue
            try:
                resp = llm.invoke(
                    "Does the following page contain a new section header "
                    "(e.g. '1.1 Changing the Tailstock Position', 'Section A', 'Chapter 2')?\n"
                    "If YES, reply with ONLY the section number and title (e.g. '1.1 Changing the Tailstock Position').\n"
                    "If NO, reply with exactly: none\n\n"
                    f"PAGE TEXT:\n{page_text}"
                )
                ans = (getattr(resp, "content", "") or "").strip()
                if ans.lower() == "none" or not ans:
                    continue
                m2 = re.match(r'^(\d+\.\d+)\s+(.+)$', ans)
                sec_num = m2.group(1) if m2 else ans[:20]
                if sec_num not in seen_nums:
                    seen_nums.add(sec_num)
                    sections_found.append({
                        "title": ans[:100].strip(),
                        "sec_num": sec_num,
                        "page_start": pg_int,
                        "page_end": pg_int,
                        "steps": []
                    })
            except Exception as e:
                logger.warning("LLM page scan error on page %d: %s", pg_int, e)

    if not sections_found:
        return []

    sections_found.sort(key=lambda x: x["page_start"])

    # Compute page_end for each section
    for i, sec in enumerate(sections_found):
        if i + 1 < len(sections_found):
            sec["page_end"] = max(sec["page_start"], sections_found[i + 1]["page_start"] - 1)
        else:
            sec["page_end"] = sec["page_start"] + 10

    return sections_found


def _extract_heading_sections_fast(blocks: list[dict]) -> list[dict]:
    """Fast extraction of section titles, step estimates, and page bounds from DB blocks (0 LLM calls)."""
    if not blocks:
        return []

    sections_found: list[dict] = []
    seen_nums = set()

    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide") or 1
        try:
            pg_int = int(pg)
        except (ValueError, TypeError):
            pg_int = 1

        raw_t = (b.get("text") or "").strip()
        if not raw_t or "..." in raw_t:
            continue

        # Match headers like "1.1 Changing the Tailstock Position" or "Section 1.1 ..."
        m = re.search(r'^\s*(\d+\.\d+)\s+([A-Za-z0-9\s\-\(\)\/\,\:\.]+)', raw_t)
        if m:
            sec_num = m.group(1)
            sec_title_raw = m.group(2).strip()
            title_line = f"{sec_num} {sec_title_raw}".split("\n")[0].strip()
            title_line = re.sub(r'\s+', ' ', title_line)
            if sec_num not in seen_nums and len(title_line) < 120 and len(title_line) > 5:
                seen_nums.add(sec_num)
                sections_found.append({
                    "title": title_line,
                    "sec_num": sec_num,
                    "page_start": pg_int,
                    "page_end": pg_int,
                    "steps": []
                })

    if not sections_found:
        # Fallback: check rule-based section parser
        rule_secs = _extract_sections_with_steps(blocks)
        if rule_secs:
            for s in rule_secs:
                st_count = len(s.get("steps", []))
                sections_found.append({
                    "title": s["title"],
                    "page_start": 1,
                    "page_end": 999,
                    "step_count": st_count,
                    "steps": s.get("steps", [])
                })
        return sections_found

    sections_found.sort(key=lambda x: x["page_start"])

    # Calculate page_end and count steps in range
    for i, sec in enumerate(sections_found):
        if i + 1 < len(sections_found):
            sec["page_end"] = max(sec["page_start"], sections_found[i + 1]["page_start"] - 1)
        else:
            sec["page_end"] = sec["page_start"] + 10

        p_start = sec["page_start"]
        p_end = sec["page_end"]
        sec_blocks = []
        for b in blocks:
            ref = b.get("source_ref") or {}
            pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
            if pg:
                try:
                    p_num = int(pg)
                    if p_start <= p_num <= p_end:
                        sec_blocks.append(b)
                except (ValueError, TypeError):
                    pass

        step_matches = 0
        for sb in sec_blocks:
            t = (sb.get("text") or "").strip()
            if re.search(r'^\s*\d+\.\d+\s+[A-Za-z]', t):
                continue
            if re.search(r'^\s*(?:\(\d+\)|\b\d+[\.\-]|Step\s+\d+)', t, re.MULTILINE | re.IGNORECASE):
                step_matches += 1
        sec["step_count"] = max(step_matches, 1)

    return sections_found


def _extract_single_section_json_jit(doc_id: str, sec_title: str, page_start: int, page_end: int, config: dict, llm) -> list[str]:
    """Lazy Just-In-Time (JIT) extraction of steps for a single section using page-bounded DB blocks."""
    blocks: list[dict] = []
    if doc_id:
        try:
            from backend.storage.postgres_store import PostgresStore
            store_pg = PostgresStore(config=config)
            try:
                blocks = store_pg.get_blocks(doc_id)
            finally:
                store_pg.close()
        except Exception as e:
            logger.warning("JIT block fetch failed for doc_id=%s: %s", doc_id, e)

    sec_blocks = []
    if blocks:
        for b in blocks:
            ref = b.get("source_ref") or {}
            pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
            if pg:
                try:
                    pg_int = int(pg)
                    if page_start <= pg_int <= page_end:
                        sec_blocks.append(b)
                except (ValueError, TypeError):
                    pass

    # Do NOT fall back to arbitrary blocks — that bleeds into other sections
    if not sec_blocks:
        logger.warning("JIT: no blocks found for section '%s' pages %d-%d", sec_title, page_start, page_end)
        return []

    raw_texts = [b.get("text", "").strip() for b in sec_blocks if b.get("text", "").strip()]
    sec_text = "\n\n".join(raw_texts)[:40000]

    if not sec_text or not llm:
        return []

    prompt = (
        "You are an expert technical procedure extraction assistant.\n"
        f"Extract ONLY the procedural action steps that belong to section '{sec_title}'.\n\n"
        "STRICT BOUNDARY RULE — THIS IS CRITICAL:\n"
        f"  - ONLY extract steps that are part of section '{sec_title}'.\n"
        "  - If you see a heading like '1.2', '1.3', 'Section 2', or any new section header, STOP IMMEDIATELY.\n"
        "  - Do NOT include any steps from the next section, even if they appear in the same text block.\n\n"
        "RULES FOR EACH STEP:\n"
        "1. Extract actual action steps that the technician must perform. They may be numbered (e.g. (1), (2)...) OR unnumbered sequential sentences.\n"
        "2. If steps are numbered, each numbered item is ONE step — never merge them. If unnumbered, extract each distinct action sentence as a separate step.\n"
        "3. YOU MUST RETAIN ALL NOTES AND CAUTIONS: Never drop or delete 'NOTE', '📌 NOTE', 'IMPORTANT', or 'CAUTION' text! Instead, append the entire note text to the END of the preceding action step. NEVER extract them as their own separate step.\n"
        "4. Do NOT extract section titles, introductory paragraphs, or torque specification tables as steps.\n"
        "5. Return ONLY a valid JSON array of step strings, nothing else:\n"
        "   [\"(1) Action 1...\", \"Unnumbered action 2...<NOTE> Watch out for X\"]\n\n"
        f"TEXT FOR SECTION '{sec_title}' (Pages {page_start}-{page_end}):\n{sec_text}"
    )

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "").strip()
        raw_steps = _extract_json_array(content)
        clean_steps = []
        for st in raw_steps:
            st_clean = st.strip()
            if st_clean:
                clean_steps.append(_format_callout_headers(st_clean))
        return clean_steps
    except Exception as e:
        logger.warning("JIT section extraction failed for '%s': %s", sec_title, e)
        return []



def _extract_sections_with_steps_agentic(blocks: list[dict], llm=None, target_topic: str = "") -> list[dict]:
    """Agentic (LLM-driven) extraction of procedural sections and steps.

    Uses LLM language intelligence to parse document blocks into structured sections
    and steps, handling any PDF formatting variation automatically.
    Merges loose notes into parent steps and ensures full sentence completion.
    """
    rule_secs = _extract_sections_with_steps(blocks)
    rule_step_count = sum(len(s["steps"]) for s in rule_secs) if rule_secs else 0

    if llm and blocks:
        try:
            SKIP_SUBS = (
                "HOW TO READ", "Handing Over", "Cost-free Repairs", "Revision History",
                "Published by", "brand identity", "company's logo",
            )
            clean_blocks = []
            for b in blocks:
                raw = b.get("text", "").strip()
                if not raw:
                    continue
                if any(s in raw for s in SKIP_SUBS):
                    continue
                if raw.lower().startswith(("the image", "the diagram", "the figure", "the photo", "[logo]")) and len(raw) > 300:
                    continue
                clean_blocks.append(raw)

            if clean_blocks:
                doc_text = "\n\n".join(clean_blocks)[:250000]
                topic_instruction = ""
                if target_topic and len(target_topic.strip()) > 3:
                    topic_instruction = f"\nTARGET PROCEDURE / TOPIC: {target_topic[:300]}\nFocus your extraction specifically on the section and steps related to this procedure.\n"

                prompt = (
                    "You are an expert technical procedure extraction assistant.\n"
                    "Analyze the following technical manual text and extract all procedural sections along with their complete step-by-step instructions.\n"
                    f"{topic_instruction}\n"
                    "STRICT RULES FOR ACTION STEPS:\n"
                    "1. Extract ONLY actual procedural ACTION steps (things the technician must DO).\n"
                    "2. INCLUDE ALL EXPLANATORY SENTENCES: Include all trailing explanation sentences belonging to a step. Do NOT cut off a step at the first period.\n"
                    "3. RETAIN AND ATTACH ALL NOTES (DO NOT DELETE THEM!): You MUST preserve all 'NOTE', '📌 NOTE', 'CAUTION', and 'IMPORTANT' blocks. Do NOT delete them. Do NOT extract them as standalone steps. You MUST append them directly to the END of the preceding action step they belong to.\n"
                    "4. Group steps into their respective sections (e.g. '1.1 Changing the Tailstock Position' or '1.2 Setting the Tailstock Spindle Position').\n"
                    "5. PRESERVE EVERY INDIVIDUAL NUMBERED STEP: Extract EVERY numbered step (1), (2), (3)... as an individual action item. Do NOT combine multiple distinct numbered steps together into a single item.\n"
                    "6. Return ONLY valid JSON in this exact structure:\n"
                    "{\n"
                    '  "sections": [\n'
                    '    {\n'
                    '      "title": "1.1 Section Title",\n'
                    '      "steps": [\n'
                    '        "Action step 1 full sentence...",\n'
                    '        "Action step 2 full sentence..."\n'
                    '      ]\n'
                    '    }\n'
                    '  ]\n'
                    "}\n\n"
                    f"DOCUMENT TEXT:\n{doc_text}"
                )
                import time
                response = None
                for attempt in range(3):
                    try:
                        response = llm.invoke(prompt)
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if ("rate limit" in err_str or "429" in err_str or "tpm" in err_str) and attempt < 2:
                            logger.warning("Rate limit hit during agentic section extraction (attempt %d/3), waiting 2s: %s", attempt + 1, e)
                            time.sleep(2.0)
                        else:
                            raise e

                content = getattr(response, "content", "").strip() if response else ""
                parsed = _extract_json_dict(content)

                if parsed and isinstance(parsed.get("sections"), list):
                    valid_secs = []
                    for s in parsed["sections"]:
                        if isinstance(s, dict) and s.get("title") and isinstance(s.get("steps"), list):
                            raw_steps = [str(st).strip() for st in s["steps"] if str(st).strip()]
                            clean_steps: list[str] = []
                            for st in raw_steps:
                                st_clean = st.strip()
                                if not st_clean:
                                    continue
                                # Merge loose standalone <IMPORTANT> or <NOTE> blocks into previous step with proper callout formatting
                                if (st_clean.startswith("<") or st_clean.lower().startswith(("note", "important", "caution", "warning", "one point"))) and clean_steps:
                                    if not st_clean.startswith("<"):
                                        parts = st_clean.split(maxsplit=1)
                                        hdr = parts[0].upper()
                                        rest = parts[1] if len(parts) > 1 else ""
                                        st_clean = f"<{hdr}>\n{rest}".strip()
                                    clean_steps[-1] = f"{clean_steps[-1]}\n\n{st_clean}"
                                else:
                                    clean_steps.append(st_clean)
                            if len(clean_steps) >= 2:
                                formatted_steps = [_format_callout_headers(st) for st in clean_steps]
                                valid_secs.append({
                                    "title": str(s["title"]).strip(),
                                    "steps": formatted_steps
                                })
                    llm_step_count = sum(len(s["steps"]) for s in valid_secs)
                    
                    if valid_secs:
                        logger.info("🤖 [Agentic Extractor] Extracted %d sections with %d total steps via LLM", len(valid_secs), llm_step_count)
                        return valid_secs
                    elif valid_secs:
                        return valid_secs
        except Exception as e:
            logger.warning("Agentic section extraction encountered error (%s), using fallback", e)

    return rule_secs


def _get_blocks_for_page(blocks: list[dict], page_no: int) -> list[dict]:
    """Return all database text blocks for a specific page number."""
    res = []
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or 0
        if pg == page_no:
            res.append(b)
    return res


def _detect_procedure_boundaries(blocks: list[dict]) -> tuple[int, int]:
    """Detect start_page and end_page boundary for procedure in a PDF manual dynamically."""
    pages = []
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or 0
        if pg > 0:
            pages.append(pg)

    if not pages:
        return (1, 1)

    min_p, max_p = min(pages), max(pages)

    # Find first page containing procedural steps (1), 1., Step 1, or explicit section headers
    start_p = None
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or 0
        if pg > 0:
            text = b.get("text", "").strip()
            if (text.startswith("(1)") or text.startswith("1.") or text.startswith("1-1") or
                text.lower().startswith("step 1") or text.startswith("[1]") or text.startswith("1.1")):
                start_p = pg
                break

    if not start_p:
        start_p = min_p

    # End page is max page before Revision History / Published by boilerplate (only if near end of doc)
    end_p = max_p
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or 0
        text = b.get("text", "").strip()
        if ("Revision History" in text or "Published by" in text) and pg >= (max_p - 2) and pg > start_p:
            end_p = min(end_p, pg - 1)

    return (start_p, max(start_p, end_p))


def _extract_next_step_from_page(
    page_blocks: list[dict],
    page_no: int,
    current_step_num: int,
    last_step_text: str | None,
    llm=None
) -> dict:
    """Extract the next actionable instruction from a single page's blocks.

    Returns dict:
    {"step_num": N, "text": "...", "page": page_no, "is_exhausted": False}
    or {"is_exhausted": True}
    """
    SKIP_SUBS = (
        "HOW TO READ", "Handing Over", "Cost-free Repairs", "Revision History",
        "Published by", "brand identity", "company's logo",
    )
    clean_texts = []
    for b in page_blocks:
        raw = b.get("text", "").strip()
        if not raw:
            continue
        if any(s in raw for s in SKIP_SUBS):
            continue
        if raw.lower().startswith(("the image", "the diagram", "the figure", "the photo", "[logo]")) and len(raw) > 300:
            continue
        clean_texts.append(raw)

    if not clean_texts:
        return {"is_exhausted": True}

    page_text = "\n\n".join(clean_texts)

    # Check for explicit parenthesized step (N) matching current_step_num
    import re
    paren_re = re.compile(r'^\((\d+)\)\s*(.+)$', re.DOTALL)
    for t in clean_texts:
        m = paren_re.match(t)
        if m:
            n = int(m.group(1))
            if n == current_step_num:
                return {
                    "step_num": current_step_num,
                    "text": m.group(2).strip(),
                    "page": page_no,
                    "is_exhausted": False
                }

    # Ask LLM for next actionable instruction on page_no
    if llm:
        try:
            prompt = (
                "You are a technical procedure execution assistant.\n"
                f"Analyze the text from Page {page_no} of the manual below.\n"
                f"Find the NEXT immediate actionable instruction (Step {current_step_num}) following this previous step: '{last_step_text or 'Start of procedure'}'.\n\n"
                "RULES:\n"
                "1. Return ONLY the single next actionable step instruction.\n"
                "2. Do NOT invent instructions or summarize multiple steps.\n"
                "3. If Page {page_no} has no more steps, return JSON: {\"is_exhausted\": true}.\n"
                "4. Return valid JSON in this format: {\"step_text\": \"Actionable instruction...\", \"is_exhausted\": false}\n\n"
                f"PAGE {page_no} TEXT:\n{page_text}"
            )
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", "").strip()
            parsed = _extract_json_dict(content)
            if parsed:
                if parsed.get("is_exhausted"):
                    return {"is_exhausted": True}
                if parsed.get("step_text"):
                    return {
                        "step_num": current_step_num,
                        "text": str(parsed["step_text"]).strip(),
                        "page": page_no,
                        "is_exhausted": False
                    }
        except Exception as e:
            logger.warning("LLM next step extraction failed on page %d: %s", page_no, e)

    # Rule fallback: first clean text sentence on page
    for t in clean_texts:
        if len(t) < 300 and t[0].isupper() and not any(t.lower().startswith(p) for p in ("note", "important", "caution", "warning", "figure")):
            return {
                "step_num": current_step_num,
                "text": t,
                "page": page_no,
                "is_exhausted": False
            }

    return {"is_exhausted": True}


def _extract_sections_with_steps(blocks: list[dict]) -> list[dict]:
    """Extract procedural sections and their steps from document blocks.

    Approach (user's idea — fully deterministic):
    1. Scan all blocks to find section headers (N.M Title) and (N) parenthesized steps.
    2. Under each section header, collect every (1)...(N) block as steps.
    3. For sections with NO (N) markers, collect short plain-sentence blocks.
    4. Result is always exact — no verb matching, no heuristics.
    """
    import re

    fw = str.maketrans('０１２３４５６７８９', '0123456789')

    # Normalise a raw block text
    def norm(raw: str) -> str:
        return raw.replace('（', '(').replace('）', ')').translate(fw)

    # Regex for section header line "1. Title", "1.1 Title", or "1.4 Setting..."
    SEC_RE = re.compile(
        r'^(\d+(?:\.\d+)?)\.?\s+([A-Z][A-Za-z0-9\s\-\(\)\/\.\,\:\;\&\=\+]{2,80})$'
    )
    # Regex for bold header "**Title**"
    BOLD_RE = re.compile(
        r'^\*\*([A-Z][A-Za-z0-9\s\-\(\)\/\.\,\:\;\&\=\+]{2,80})\*\*$'
    )
    # Regex for step matching: "(1)", "1.", "1-1.", "Step 1:", "[1]"
    PAREN_RE = re.compile(r'^(?:\((\d+)\)|\b(\d+)\.|\b(\d+-\d+)\.|Step\s+(\d+)[:\.]?|\[(\d+)\])\s*(.+)$', re.IGNORECASE | re.DOTALL)

    def _parse_step(raw_text: str) -> tuple[int, str] | None:
        m = PAREN_RE.match(raw_text)
        if m:
            n_str = m.group(1) or m.group(2) or m.group(4) or m.group(5)
            n_val = int(n_str) if n_str and n_str.isdigit() else 1
            body = m.group(6).strip()
            return n_val, body
        return None

    # Blocks to always skip
    SKIP_SUB = ("HOW TO READ", "Handing Over", "Cost-free Repairs", "Revision History",
                "Published by", "brand identity", "company's logo")
    SKIP_STR = ("the image", "the diagram", "the figure", "the photo", "this diagram",
                "this image", "this object", "the object", "the logo", "[logo]", "[figure]")

    # ── Pass 0: Build title→section_number from ALL blocks (TOC is a markdown table) ──
    title_to_num: dict[str, str] = {}
    for b in blocks:
        raw = b.get("text", "").strip()
        if not raw:
            continue
        text = norm(raw)
        for line in text.splitlines():
            line = line.strip().strip('|').strip()  # remove table pipes
            # Remove trailing page numbers like "...1-4" or ".....1"
            line = re.sub(r'[\s\.]+\d+\s*$', '', line).strip()
            m = SEC_RE.match(line)
            if m:
                num = m.group(1)
                title = m.group(2).strip()
                if title and len(title) > 3:
                    title_to_num[title.lower()] = num

    # ── Pass 1: Scan blocks in order ─────────────────────────────────────────
    # Assign every block to a section bucket
    # bucket: list of (sec_title, sec_num, [block_texts])
    buckets: list[dict] = []   # {title, num, blocks: [str]}
    cur_title: str | None = None
    cur_num:   str | None = None

    for b in blocks:
        raw = b.get("text", "").strip()
        if not raw:
            continue
        text = norm(raw)
        text_lo = text.lower()

        # Skip boilerplate
        if any(s in text for s in SKIP_SUB):
            continue
        if any(text_lo.startswith(s) for s in SKIP_STR):
            continue
        if len(text) > 800 and text.count('\n') > 6:
            continue  # long vision AI paragraphs

        # --- Detect section header ---
        first_line = text.splitlines()[0].strip() if '\n' in text else text
        first_line_clean = first_line.strip('* |').strip()

        # Numbered header like "1. Cleaning up..." or "1.1 Replacing..."
        if len(first_line_clean) < 120:
            sec_m = SEC_RE.match(first_line_clean)
            if sec_m:
                cur_num   = sec_m.group(1)
                cur_title = f"{cur_num} {sec_m.group(2).strip()}"
                buckets.append({"title": cur_title, "num": cur_num, "blocks": []})
                continue

        # Bold header like "**Setting the Phase Indexing Encoder**"
        if len(first_line) < 120 and not any(text_lo.startswith(s) for s in SKIP_STR):
            bold_m = BOLD_RE.match(first_line)
            if bold_m:
                bold_title = bold_m.group(1).strip()
                mapped = title_to_num.get(bold_title.lower())
                if not mapped and cur_num:
                    # Infer next section number as fallback
                    mn = re.match(r'^(\d+)\.(\d+)$', cur_num)
                    if mn:
                        mapped = f"{mn.group(1)}.{int(mn.group(2))+1}"
                    else:
                        mapped = f"{cur_num}.1"
                if mapped:
                    cur_num   = mapped
                    cur_title = f"{mapped} {bold_title}"
                    buckets.append({"title": cur_title, "num": cur_num, "blocks": []})
                    continue

        # Skip markdown table blocks (but not in Pass 0)
        if text.count('|') >= 4:
            continue

        # Fallback: if no section header was found yet and we encounter step (1), auto-create Section 1
        if cur_title is None and PAREN_RE.match(text):
            cur_num   = "1"
            cur_title = "1. Procedure"
            buckets.append({"title": cur_title, "num": cur_num, "blocks": []})

        # Assign this block to current section bucket
        if cur_title is not None and buckets:
            buckets[-1]["blocks"].append(text)

    # ── Pass 2: For each bucket, extract steps using (1)...(N) anchoring ──────
    # YOUR IDEA: under every section, find where (1) appears, then collect from there
    sections: list[dict] = []
    seen_nums: set[str] = set()

    for bucket in buckets:
        sec_title = bucket["title"]
        sec_num   = bucket["num"]
        blks      = bucket["blocks"]

        if sec_num in seen_nums or not blks:
            continue

        steps: list[str] = []

        # Check if this section uses numbered steps
        has_paren_steps = any(_parse_step(t) is not None for t in blks)

        if has_paren_steps:
            found_start = False
            for t in blks:
                parsed = _parse_step(t)
                if parsed:
                    n, body = parsed
                    if n == 1 and not found_start:
                        found_start = True
                        steps = []  # reset — start fresh from initial (1)
                    if found_start or n > 1 or not steps:
                        found_start = True
                        steps.append(t.strip())
                elif found_start and steps:
                    t_clean = t.strip()
                    # Preserve <IMPORTANT> / <NOTE> callout blocks with header formatting under parent step
                    if any(t_clean.lower().startswith(p) for p in ("<important>", "<note>", "<caution>", "<warning>", "<one point>", "important", "note", "caution", "warning", "one point")):
                        if not t_clean.startswith("<"):
                            parts = t_clean.split(maxsplit=1)
                            hdr = parts[0].upper()
                            rest = parts[1] if len(parts) > 1 else ""
                            t_clean = f"<{hdr}>\n{rest}".strip()
                        steps[-1] = f"{steps[-1]}\n\n{t_clean}"
                    elif not any(t_clean.lower().startswith(p) for p in ("figure", "table", "photo", "the image", "the diagram", "[logo]", "[figure]")):
                        steps[-1] = f"{steps[-1]} {t_clean}"
        else:
            # No (N) markers → collect plain procedural sentences
            SKIP_PREFIXES = ("note", "important", "caution", "warning", "figure",
                             "table", "photo", "example", "the image", "the diagram")
            for t in blks:
                if any(t.lower().startswith(p) for p in SKIP_PREFIXES):
                    continue
                if len(t) > 300 or t.count('\n') > 3:
                    continue
                if t and t[0].isupper():
                    steps.append(t)

        if steps:
            seen_nums.add(sec_num)
            formatted_steps = [_format_callout_headers(st) for st in steps]
            sections.append({"title": sec_title, "steps": formatted_steps})

    # Sort by section number
    def _key(s: dict) -> tuple:
        m = re.match(r'^(\d+)\.(\d+)', s["title"])
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    sorted_sections = sorted(sections, key=_key)

    # Pass 3: Merge continuous procedures that were falsely split by bold sub-headers
    merged_sections = []
    for sec in sorted_sections:
        if not merged_sections:
            merged_sections.append(sec)
            continue
            
        prev_sec = merged_sections[-1]
        
        if prev_sec["steps"] and sec["steps"]:
            last_parsed = _parse_step(prev_sec["steps"][-1])
            first_parsed = _parse_step(sec["steps"][0])
            
            if last_parsed and first_parsed:
                last_num = last_parsed[0]
                first_num = first_parsed[0]
                
                if first_num == last_num + 1:
                    if "Procedure" in prev_sec["title"] and "Procedure" not in sec["title"]:
                        prev_sec["title"] = sec["title"]
                        
                    prev_sec["steps"].extend(sec["steps"])
                    continue
                    
        merged_sections.append(sec)
        
    return merged_sections



def _group_steps_into_sections(steps: list[str], blocks: list[dict] | None = None) -> list[dict]:
    """Group flat steps into distinct sections using DB blocks or by splitting flat steps lists."""
    import re
    if blocks:
        sec_from_blocks = _extract_sections_with_steps_agentic(blocks)
        if len(sec_from_blocks) > 1:
            return sec_from_blocks

    if not steps or len(steps) <= 5:
        return []

    sections = []
    current_title = "Section 1.1"
    current_steps = []
    sec_counter = 1

    for step in steps:
        m_prefix = re.match(r'^\[([A-Za-z0-9\.\s\-]+)\s*-\s*Step\s*\d+\s*of\s*\d+\]\s*(.+)$', step)
        if m_prefix:
            sec_title = m_prefix.group(1).strip()
            step_content = m_prefix.group(2).strip()
            if current_steps and current_title != sec_title:
                sections.append({"title": current_title, "steps": current_steps})
                current_steps = []
            current_title = sec_title
            current_steps.append(step_content)
            continue

        is_reset = bool(re.match(r'^\(1\)\s', step) or re.match(r'^1\.\s', step) or re.match(r'^1-1\.\s', step))
        if is_reset and len(current_steps) >= 2:
            sections.append({"title": current_title, "steps": current_steps})
            sec_counter += 1
            current_title = f"Section 1.{sec_counter}"
            current_steps = []

        current_steps.append(step)

    if current_steps:
        sections.append({"title": current_title, "steps": current_steps})

    # If steps were flat (>15 steps) and not split by reset headers, split into 6 chunks
    if len(sections) == 1 and len(steps) > 15:
        chunk_size = max(5, len(steps) // 6)
        sections = []
        for i in range(0, len(steps), chunk_size):
            chunk = steps[i:i + chunk_size]
            sec_num = (i // chunk_size) + 1
            sections.append({"title": f"Section 1.{sec_num}", "steps": chunk})

    return sections if len(sections) > 1 else []


def _extract_numbered_steps_from_blocks(blocks: list[dict]) -> list[str]:
    """Extract numbered steps from document blocks with section awareness."""
    sec_data = _extract_sections_with_steps(blocks)
    if sec_data:
        formatted_steps = []
        multi_section = len(sec_data) > 1
        for sec in sec_data:
            sec_title = sec["title"]
            sec_steps = sec["steps"]
            sec_count = len(sec_steps)
            for idx, step_text in enumerate(sec_steps, start=1):
                if multi_section:
                    prefix = f"**[{sec_title} - Step {idx} of {sec_count}]** "
                else:
                    prefix = ""
                formatted_steps.append(f"{prefix}{step_text}")
        return formatted_steps

    # --- Pattern 2: Section-dot numbers N-N. (e.g. 1-1., 2-3., 5-14.)
    steps_list: list[tuple[tuple[int, int], str]] = []
    seen_keys: set[tuple[int, int]] = set()
    for b in blocks:
        text = b.get("text", "").strip()
        if not text:
            continue
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = re.match(r'^(\d+)-(\d+)\.\s*(.+)$', line_str)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps_list.append((key, m.group(3).strip()))
    if len(steps_list) >= 2:
        steps_list.sort(key=lambda x: x[0])
        return [s for _, s in steps_list]

    # --- Pattern 3: Standard dot numbers 1., 2., ...
    steps_map = {}
    current_num = None
    for b in blocks:
        text = b.get("text", "").strip()
        if not text:
            continue
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = re.match(r'^(\d+)\.\s*(.+)$', line_str)
            if m:
                num = int(m.group(1))
                content = m.group(2).strip()
                steps_map[num] = content
                current_num = num
            elif current_num is not None:
                if (
                    not re.match(r'^\d+\.', line_str)
                    and not re.match(r'^[A-Z\s]{8,}$', line_str)
                    and len(line_str) > 3
                ):
                    steps_map[current_num] = steps_map[current_num] + " " + line_str
    if len(steps_map) >= 2:
        return [steps_map[k].strip() for k in sorted(steps_map.keys())]

    return []


def _collect_procedural_steps_from_blocks(blocks: list[dict]) -> list[str]:
    """Deterministically collect all actionable steps from vision-processed PDF blocks.

    Designed for manuals where each sentence is a separate DB block and the original
    step numbers (1)-(38) were not preserved by the vision AI.

    Strategy:
    1. Group blocks by page number (from source_ref.page)
    2. Find the FIRST page that has >= 3 actionable blocks (real procedure start)
    3. Skip preface/cover/warranty pages and the last revision-history page
    4. Within procedural pages: track NOTE/IMPORTANT context windows to skip note bodies
    5. Skip image captions, logos, intro text, section headers
    """
    import re
    from collections import defaultdict

    # === Skip filters ===

    # Exact match safety-label blocks (just the label, no content)
    SKIP_EXACT = {"<IMPORTANT>", "<NOTE>", "<ONE POINT>", "<CAUTION>", "<WARNING>", "<DANGER>"}

    # Block starts that identify image/diagram AI descriptions
    CAPTION_PREFIXES = (
        "The image shows", "The image displays", "The image is",
        "The diagram shows", "The diagram is", "The caption",
        "* **Logo", "[logo]", "[figure]",
        "that describes the steps",   # partial image caption artifact (block 17)
        "** The image shows",          # blocks starting with ** marker + image
    )

    # Substrings that identify non-procedural boilerplate pages
    BOILERPLATE_SUBSTRINGS = (
        # Preface / warranty text
        "Published by", "All rights Reserved", "Reproduction of this manual",
        "Cost-free Repairs", "Chargeable Repairs", "term of the guarantee",
        "Revision History", "Description of revision",
        "No part of this manual", "All specifications and designs",
        "subject to change without notice",
        "Store this operation manual",
        "Read the operation manual and become fully familiar",
    )

    # Note body prefixes - generic heuristics for secondary text
    NOTE_BODY_PREFIXES = (
        "Note:", "Important:", "Caution:", "Warning:", "Danger:",
        "<IMPORTANT>", "<NOTE>", "<CAUTION>", "<WARNING>", "<DANGER>"
    )

    # Section heading patterns (page section titles, not steps)
    SECTION_HEADER_RE = re.compile(
        r'^\d+\.\s+[A-Z][A-Za-z ]{5,}$|'          # "1. Cleaning up the Wheel..."
        r'^\*\*\[?[A-Z][^*]{2,}\]?\*\*$|'          # **[Tools to be prepared]**
        r'^(Preface|Contents|HOW TO READ|About the Manual|Guarantee|Overview|Table of Contents)'
    )

    INLINE_MARKER_RE = re.compile(r'^\(Operation manual|^When a protector such as')

    # Imperative verb regex — used for first_proc_page detection AND per-block check
    IMPERATIVE_RE = re.compile(
        r'\b(remove|loosen|clean|insert|press|click|turn|tighten|check|'
        r'attach|fix|mount|operate|place|release|rotate|confirm|apply|'
        r'grind|screw|lift|close|open|start|stop|shut|display|input|'
        r'put|scrape|use|extract|push|pull)\b',
        re.IGNORECASE
    )
    # Separate, more strict regex for first-procedure-page detection
    STRONG_IMPERATIVE_RE = re.compile(
        r'\b(remove|loosen|clean up|insert|press the|click the|turn the|'
        r'tighten|attach|fix the|mount the|operate|rotate|apply oil|grind|'
        r'loosen and remove|put an M6|screw the bolts|fix with)\b',
        re.IGNORECASE
    )

    # === Pass 1: group blocks by page ===
    pages: dict[int, list[str]] = defaultdict(list)
    for b in blocks:
        ref = b.get("source_ref") or {}
        page = ref.get("page") or ref.get("sheet") or 0
        text = b.get("text", "").strip()
        if text:
            pages[page].append(text)

    sorted_pages = sorted(pages.keys())
    if not sorted_pages:
        return []

    last_page = sorted_pages[-1]

    # === Pass 2: find first REAL procedure page ===
    # Require a page to have at least 3 unique strong-imperative blocks
    # to avoid triggering on preface pages that just mention "check" once
    first_proc_page = last_page  # default: no procedure found
    for pg in sorted_pages:
        count = 0
        for t in pages[pg]:
            # Skip known preface content even in this scan
            if any(sub in t for sub in ("Refer to the chapter", "follow the directions",
                                        "supplement to Chapter", "Please follow the procedures below")):
                continue
            if STRONG_IMPERATIVE_RE.search(t):
                count += 1
        if count >= 2:
            first_proc_page = pg
            break

    # === Pass 3: collect steps from procedural pages ===
    steps: list[str] = []
    seen: set[str] = set()

    for pg in sorted_pages:
        if pg < first_proc_page:
            continue
        # Skip revision-history last page
        if pg == last_page:
            page_text = " ".join(pages[pg])
            if "Revision" in page_text or "revision" in page_text:
                continue

        for text in pages[pg]:
            if not text or len(text) < 15:
                continue

            # Skip exact safety label blocks
            if text in SKIP_EXACT:
                continue

            # Skip image/diagram caption blocks
            skip = False
            for prefix in CAPTION_PREFIXES:
                if text.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            # Skip boilerplate text
            for sub in BOILERPLATE_SUBSTRINGS:
                if sub in text:
                    skip = True
                    break
            if skip:
                continue

            # Skip section headers
            if SECTION_HEADER_RE.match(text):
                continue

            # Skip inline markers
            if INLINE_MARKER_RE.match(text):
                continue

            # Skip known NOTE/IMPORTANT body blocks
            for prefix in NOTE_BODY_PREFIXES:
                if text.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            # Skip blocks that are mostly bullet-list image descriptions
            bullet_lines = [l for l in text.splitlines() if l.strip().startswith(("*", "-"))]
            if len(bullet_lines) > 3 and len(text) > 200:
                continue

            # Skip long multi-paragraph blocks (image analysis artifacts > 500 chars with > 4 newlines)
            if len(text) > 500 and text.count("\n") > 4:
                continue

            # Skip blocks starting with ** (markdown image captions from vision AI)
            if text.startswith("**"):
                continue

            # Require actionable content: must contain an imperative verb
            has_imperative = bool(re.search(
                r'\b(remove|loosen|clean|insert|press|click|turn|tighten|check|'
                r'attach|fix|mount|operate|place|release|rotate|confirm|apply|'
                r'grind|screw|put|scrape|use|extract|push|shut|display|input)\b',
                text, re.IGNORECASE
            ))
            if not has_imperative:
                # Allow blocks with explicit parenthesized step numbers even without imperative
                if not re.match(r'^\(\d+\)', text):
                    continue

            # Strip leading step number prefix if present
            clean = re.sub(r'^\(\d+\)\s*', '', text)
            clean = re.sub(r'^\d+-\d+\.\s*', '', clean)
            clean = re.sub(r'^\d+\.\s+', '', clean)
            clean = clean.strip()

            if not clean:
                continue

            # Deduplicate by first 60 chars
            key = clean[:60].lower()
            if key not in seen:
                seen.add(key)
                steps.append(clean)

    return steps



def _find_exact_step_page(step_text: str, blocks: list[dict], fallback_page: int = 1) -> int:
    """Find the exact PDF page number in DB blocks where step_text originates."""
    if not step_text or not blocks:
        return fallback_page

    import re

    # ── Section Page Range Detection ──
    # Build list of (section_number, start_page) from DB blocks (skipping TOC lines)
    sec_pages: list[tuple[str, int]] = []
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
        if not pg:
            continue
        try:
            pg_int = int(pg)
        except (ValueError, TypeError):
            continue
        raw_t = (b.get("text") or "").strip()
        if not raw_t:
            continue

        # Skip Table of Contents (TOC) blocks
        if "...." in raw_t or "..." in raw_t or re.search(r'[\.\s]{3,}\d+[-\s]*\d*$', raw_t):
            continue

        m = re.search(r'^\s*(\d+\.\d+)\b', raw_t)
        if m:
            s_num = m.group(1)
            if not any(sp[0] == s_num for sp in sec_pages):
                sec_pages.append((s_num, pg_int))

    sec_pages.sort(key=lambda x: x[1])

    min_page = fallback_page
    max_page = 9999

    sec_num_match = re.search(r'^\s*\*?\s*\[\s*(\d+\.\d+)', step_text)
    if sec_num_match:
        target_sec = sec_num_match.group(1)
        for i, (s_num, s_pg) in enumerate(sec_pages):
            if s_num == target_sec:
                min_page = s_pg
                if i + 1 < len(sec_pages):
                    max_page = sec_pages[i + 1][1] - 1
                break
        fallback_page = min_page

    # Strip any leading section bracket prefix [1.1 Title] to access the actual step marker
    clean_for_step_num = re.sub(r'^\*?\s*\[.*?\]\s*\*?\s*', '', step_text.strip())
    step_num_match = re.search(
        r'^(?:\((\d+)\)|\b(\d+)\.|\b(\d+-\d+)\.|Step\s+(\d+)[:\.]?|\[(\d+)\])',
        clean_for_step_num,
        re.IGNORECASE
    )
    step_num_str = None
    if step_num_match:
        step_num_str = (step_num_match.group(1) or step_num_match.group(2) or
                        step_num_match.group(4) or step_num_match.group(5))

    def _norm(s: str) -> str:
        s = re.sub(r'^\*+\s*\[.*?\]\s*\*+\s*', '', s)
        s = re.sub(r'^\[.*?\]\s*', '', s)
        s = re.sub(r'^\*+|\*+$', '', s)
        s = re.sub(r'^(?:\(\d+\)|\b\d+[\.\-:]|Step\s+\d+[:\.]?|\[\d+\])\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'<\s*[A-Z_]+\s*>', ' ', s)
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        return ' '.join(s.lower().split())

    clean_step = _norm(step_text)
    if not clean_step or len(clean_step) < 3:
        return fallback_page

    step_snippet = clean_step[:60] if len(clean_step) >= 60 else clean_step

    # Partition blocks: forward (min_page <= page <= max_page) vs prior/other pages
    forward_blocks = []
    prior_blocks = []
    for b in blocks:
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
        if not pg:
            continue
        try:
            pg_int = int(pg)
        except (ValueError, TypeError):
            continue

        if min_page <= pg_int <= max_page:
            forward_blocks.append((pg_int, b))
        else:
            prior_blocks.append((pg_int, b))

    # Pass 1: Step number + snippet match in forward_blocks
    if step_num_str:
        step_patterns = [
            f"({step_num_str})",
            f"{step_num_str}.",
            f"{step_num_str}-",
            f"step {step_num_str}",
            f"[{step_num_str}]"
        ]
        for pg_int, b in forward_blocks:
            raw_b = (b.get("text") or "").strip()
            if not raw_b:
                continue
            raw_b_lo = raw_b.lower()
            if any(p in raw_b_lo for p in step_patterns) or re.search(r'\b' + re.escape(step_num_str) + r'\b', raw_b):
                norm_b = _norm(raw_b)
                if (step_snippet and step_snippet in norm_b) or (len(norm_b) >= 12 and norm_b in clean_step):
                    return pg_int

    # Pass 2: Snippet match in forward_blocks
    for pg_int, b in forward_blocks:
        raw_b = (b.get("text") or "").strip()
        if not raw_b:
            continue
        norm_b = _norm(raw_b)
        if (step_snippet and step_snippet in norm_b) or (len(norm_b) >= 12 and norm_b in clean_step):
            return pg_int

    # Pass 3: Keyword overlap scoring in forward_blocks
    STOP_WORDS = {"the", "that", "this", "with", "from", "have", "before", "after", "then", "when", "your", "into", "over", "step", "page", "section", "figure", "table", "note", "important", "must", "shall", "should"}
    step_words = set(w for w in clean_step.split() if len(w) >= 3 and w not in STOP_WORDS)

    if step_words and forward_blocks:
        page_scores: dict[int, int] = {}
        for pg_int, b in forward_blocks:
            norm_b = _norm(b.get("text") or "")
            if not norm_b:
                continue
            b_words = set(w for w in norm_b.split() if len(w) >= 3 and w not in STOP_WORDS)
            matches = len(step_words.intersection(b_words))
            if matches > 0:
                page_scores[pg_int] = page_scores.get(pg_int, 0) + matches

        if page_scores:
            best_pg, best_score = max(page_scores.items(), key=lambda item: item[1])
            if best_score >= 2 or (len(step_words) <= 3 and best_score >= 1):
                return best_pg

    # Pass 4: Check prior_blocks if forward_blocks returned no match
    if step_num_str:
        step_patterns = [
            f"({step_num_str})",
            f"{step_num_str}.",
            f"{step_num_str}-",
            f"step {step_num_str}",
            f"[{step_num_str}]"
        ]
        for pg_int, b in prior_blocks:
            raw_b = (b.get("text") or "").strip()
            if not raw_b:
                continue
            raw_b_lo = raw_b.lower()
            if any(p in raw_b_lo for p in step_patterns) or re.search(r'\b' + re.escape(step_num_str) + r'\b', raw_b):
                norm_b = _norm(raw_b)
                if (step_snippet and step_snippet in norm_b) or (len(norm_b) >= 12 and norm_b in clean_step):
                    return pg_int

    for pg_int, b in prior_blocks:
        raw_b = (b.get("text") or "").strip()
        if not raw_b:
            continue
        norm_b = _norm(raw_b)
        if (step_snippet and step_snippet in norm_b) or (len(norm_b) >= 12 and norm_b in clean_step):
            return pg_int

    return fallback_page


def _select_best_sections_agentic(sec_data: list[dict], target_topic: str, llm=None) -> list[dict]:
    """Agentically filter extracted document sections down to target_topic, or retain all sections for full procedures."""
    if not sec_data:
        return []
    if len(sec_data) <= 1 or not target_topic or len(target_topic.strip()) < 3:
        return sec_data

    # If target topic asks for overview, full procedure, changeover, or general manual steps, retain ALL sections
    topic_clean = target_topic.lower()
    if any(w in topic_clean for w in ("overview", "full", "complete", "all", "entire", "setup", "changeover", "workhead", "work spindle", "procedure", "checklist", "manual", "guide")):
        return sec_data

    import re
    topic_words = set(w for w in re.findall(r'\b\w{4,}\b', topic_clean) if w not in (
        "step", "steps", "procedure", "how", "with", "from", "that", "this", "have", "been",
        "using", "your", "what", "which", "process", "checklist", "manual", "guide", "here"
    ))

    if not topic_words:
        return sec_data

    # Find match score across all sections
    scores = []
    for sec in sec_data:
        title = sec.get("title", "").lower()
        steps_text = " ".join(sec.get("steps", [])).lower()
        sec_full = f"{title} {steps_text}"
        title_matches = sum(1 for w in topic_words if w in title)
        full_matches = sum(1 for w in topic_words if w in sec_full)
        scores.append(title_matches * 3 + full_matches)

    max_score = max(scores) if scores else 0

    matched_secs = []
    for sec, score in zip(sec_data, scores):
        if max_score > 0 and score >= max(1, max_score * 0.3):
            matched_secs.append(sec)

    # If matching preserved most sections or user wants complete coverage, return all sections
    if matched_secs and len(matched_secs) >= max(2, int(len(sec_data) * 0.5)):
        return sec_data
    if matched_secs:
        logger.info("🤖 [Agentic Section Selection] Matched target topic '%s' to %d section(s): %s",
                    target_topic[:50], len(matched_secs), [s['title'] for s in matched_secs])
        return matched_secs

    return sec_data


def _extract_page_steps_llm(text: str, page_num: int, llm) -> list[str]:
    """Extract procedural steps from a single page's text using LLM."""
    if not text.strip():
        return []

    logger.info("Sending Page %d to LLM for step extraction...", page_num)

    prompt = (
        "You are a precise technical procedure extraction assistant for industrial machinery manuals.\n"
        "Extract ONLY the procedural action steps present on this specific page.\n\n"
        "STRICT RULES:\n"
        "1. PRESERVE STEP NUMBERS: If steps are numbered (e.g. (1), (2), (3)... or 1., 2., 3...), ALWAYS keep the step number prefix (e.g. '(14)') at the start of each step string!\n"
        "2. ONE NUMBER = ONE STEP: Each numbered item (e.g. '(14) Operate the machine...') must remain a SINGLE step. Do NOT split a single numbered step into multiple steps.\n"
        "3. STRICT NOTE PLACEMENT: Notes (<IMPORTANT>, <NOTE>, <CAUTION>) MUST be attached to the step they physically follow in the source text! For example, if an <IMPORTANT> note appears physically after step (2) and before step (3), you MUST append it to the end of the step (2) string. Do NOT attach it to step (1)! Format notes with double newlines and bold text like this: '\\n\\n**IMPORTANT:** Note text here.' (Do NOT use < > brackets in the bold header).\n"
        "4. DO NOT HALLUCINATE OR INVENT STEPS: Do NOT include title headers, preface warnings, or unnumbered introductory statements like 'Read the operation manual' unless it is an explicit action step.\n"
        "5. Return ONLY a valid JSON array of strings, with no markdown fences or explanation:\n"
        "   [\"(1) First step text...\", \"(2) Second step text...\\n\\n**IMPORTANT:**\\nNote text here.\"]\n\n"
        f"PAGE {page_num} TEXT:\n{text}\n"
    )

    try:
        response = llm.invoke(prompt)
        content = clean_message_content(response.content)
        import json
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(s).strip() for s in data if str(s).strip()]
    except Exception as e:
        logger.warning("LLM page %d step extraction failed: %s", page_num, e)

    return []


def _detect_procedure_start_page_llm(blocks: list[dict], procedure_title: str, llm) -> int:
    """Use a single LLM call to classify which page the procedure Step 1 begins on.

    Reads a condensed preview of each page from pages 1 to 20, and asks the LLM
    to identify the exact start page where actual action steps begin.
    """
    if not blocks:
        return 1

    SKIP_SUBS = (
        "HOW TO READ", "Handing Over", "Cost-free Repairs", "Revision History",
        "Published by", "brand identity", "company's logo", "Table of Contents"
    )

    page_map: dict[int, list[str]] = {}
    for b in blocks:
        raw = (b.get("text") or "").strip()
        if not raw or any(s in raw for s in SKIP_SUBS):
            continue
        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
        try:
            pg_num = int(pg) if pg is not None else None
        except (ValueError, TypeError):
            pg_num = None
        if pg_num is not None:
            if pg_num not in page_map:
                page_map[pg_num] = []
            page_map[pg_num].append(raw)

    if not page_map:
        return 1

    sorted_pages = sorted(page_map.keys())

    if llm:
        candidate_pages = [p for p in sorted_pages if p <= 20]

        page_snippets = []
        for p in candidate_pages:
            full_p_text = " ".join(page_map[p])
            snippet = full_p_text[:250].replace("\n", " ")
            page_snippets.append(f"Page {p}: {snippet}")

        previews_text = "\n".join(page_snippets)

        prompt = (
            "You are an industrial technical manual classifier.\n"
            f"Goal: Find the page where the procedure for '{procedure_title}' starts its actual action steps.\n\n"
            "Here are snippets from the first pages:\n"
            f"{previews_text}\n\n"
            "RULES:\n"
            "1. Identify the first page where actual action steps (e.g. '(1)', '1.', 'Remove...', 'Loosen...') begin.\n"
            "2. Do NOT select the cover/title page or introduction pages.\n"
            "3. Return ONLY the integer page number (e.g. 6). No words, no explanation."
        )

        try:
            resp = llm.invoke(prompt)
            content = clean_message_content(resp.content).strip()
            m = re.search(r'\b\d+\b', content)
            if m:
                detected_page = int(m.group(0))
                if detected_page in page_map:
                    logger.info("[Start Page Classifier] LLM identified procedure start page: %d (for '%s')", detected_page, procedure_title)
                    return detected_page
        except Exception as e:
            logger.warning("[Start Page Classifier] LLM start page detection failed: %s", e)

    # Fallback heuristic: find first page with (1) or 1.
    for p in sorted_pages:
        p_text = "\n".join(page_map[p])
        if re.search(r'^\s*(?:\(1\)|1\.)\s+[A-Z]', p_text, re.MULTILINE) or "(1)" in p_text:
            logger.info("[Start Page Classifier] Heuristic fallback identified start page: %d", p)
            return p

    return 1


def _extract_all_steps_page_by_page_agentic(blocks: list[dict], llm, start_page: int = 1) -> list[str]:
    """Iterate through document blocks page-by-page and use LLM to extract all steps.

    Args:
        blocks: All document blocks from Postgres.
        llm: LLM client.
        start_page: First page to read (derived from Qdrant retrieval min-page or LLM classifier).
                    Pages before this are skipped — avoids cover/preface hallucinations.
    """
    if not blocks or not llm:
        return []

    SKIP_SUBS = (
        "HOW TO READ", "Handing Over", "Cost-free Repairs", "Revision History",
        "Published by", "brand identity", "company's logo", "Table of Contents"
    )

    page_map: dict[int, list[str]] = {}
    fallback_texts: list[str] = []

    for b in blocks:
        raw = b.get("text", "").strip()
        if not raw:
            continue
        if any(s in raw for s in SKIP_SUBS):
            continue

        ref = b.get("source_ref") or {}
        pg = ref.get("page") or ref.get("sheet") or ref.get("slide")
        try:
            pg_num = int(pg) if pg is not None else None
        except (ValueError, TypeError):
            pg_num = None

        if pg_num is not None:
            if pg_num not in page_map:
                page_map[pg_num] = []
            page_map[pg_num].append(raw)
        else:
            fallback_texts.append(raw)

    all_extracted_steps: list[str] = []

    if page_map:
        sorted_pages = sorted(page_map.keys())
        consecutive_empty = 0
        found_any_steps = False
        for p in sorted_pages:
            # Skip pages before start_page (preface, cover, TOC)
            if p < start_page:
                logger.info("Skipping Page %d (before procedure start page %d)", p, start_page)
                continue
            p_text = "\n\n".join(page_map[p])
            if len(p_text.strip()) < 20:
                continue
            steps = _extract_page_steps_llm(p_text, p, llm)
            if steps:
                all_extracted_steps.extend(steps)
                found_any_steps = True
                consecutive_empty = 0
            else:
                if found_any_steps:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("Page %d: 2 consecutive empty pages after steps found — stopping early.", p)
                        break
    elif fallback_texts:
        full_text = "\n\n".join(fallback_texts)
        chunks = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]
        for idx, chunk in enumerate(chunks, 1):
            steps = _extract_page_steps_llm(chunk, idx, llm)
            if steps:
                all_extracted_steps.extend(steps)

    # Deduplicate and sort by step number if steps have numbers (1..N)
    PAREN_NUM_RE = re.compile(r'^(?:\((\d+)\)|\b(\d+)\.|\b(\d+-\d+)\.|Step\s+(\d+)[:\.]?|\[(\d+)\])', re.IGNORECASE)
    seen_nums: set[int] = set()
    deduped_steps: list[str] = []

    for s in all_extracted_steps:
        m = PAREN_NUM_RE.match(s)
        if m:
            n_str = m.group(1) or m.group(2) or m.group(4) or m.group(5)
            if n_str and n_str.isdigit():
                n_val = int(n_str)
                if n_val in seen_nums:
                    continue
                seen_nums.add(n_val)
        deduped_steps.append(s)

    if seen_nums:
        def _sort_key(s: str) -> int:
            m = PAREN_NUM_RE.match(s)
            if m:
                n_str = m.group(1) or m.group(2) or m.group(4) or m.group(5)
                if n_str and n_str.isdigit():
                    return int(n_str)
            return 9999
        deduped_steps.sort(key=_sort_key)

    return deduped_steps


def _extract_steps_from_text_or_blocks(document_id: str, final_answer: str, config: dict, llm, start_page: int = 1) -> list[str]:
    """Extract procedural steps using an agentic multi-tier strategy.

    Args:
        start_page: First page to read from. Can be overridden by LLM start page classifier.

    Tier 1: Page-by-Page Agentic LLM extraction from DB blocks (with LLM start page classification)
    Tier 2: Rule-based section & step extraction from DB blocks
    Tier 3: LLM extraction from text
    """
    blocks: list[dict] = []
    if document_id:
        from backend.storage.postgres_store import PostgresStore
        store = PostgresStore(config=config)
        try:
            blocks = store.get_blocks(document_id)
        except Exception as e:
            logger.warning("Error loading blocks for step extraction: %s", e)
        finally:
            store.close()

    # Tier 1: Page-by-Page Agentic LLM Extraction (with Option B LLM Start Page Classifier)
    if blocks and llm:
        try:
            classified_start = _detect_procedure_start_page_llm(blocks, final_answer, llm)
            actual_start_page = max(start_page, classified_start) if start_page > 1 else classified_start
            logger.info("[Step Extraction] Using procedure start page: %d (classified=%d, requested=%d)", actual_start_page, classified_start, start_page)
            page_steps = _extract_all_steps_page_by_page_agentic(blocks, llm, start_page=actual_start_page)
            if page_steps and len(page_steps) >= 2:
                formatted = [_format_callout_headers(s) for s in page_steps]
                logger.info("[Step Extraction] Tier 1 (Page-by-Page Agentic LLM): Extracted %d steps from page %d+", len(formatted), actual_start_page)
                return formatted
        except Exception as e:
            logger.warning("Page-by-page agentic step extraction failed: %s", e)

    # Tier 2: Rule-based section & step extraction from DB blocks
    if blocks:
        sec_data = _extract_sections_with_steps(blocks)
        if sec_data:
            selected_secs = _select_best_sections_agentic(sec_data, final_answer, llm=llm) if final_answer else sec_data
            all_steps: list[str] = []
            for sec in selected_secs:
                prefix = f"[{sec['title']}] " if len(selected_secs) > 1 else ""
                for s in sec["steps"]:
                    all_steps.append(f"{prefix}{s}")
            if len(all_steps) >= 2:
                formatted = [_format_callout_headers(s) for s in all_steps]
                logger.info("[Step Extraction] Tier 2 (Rule-based Sections): Extracted %d steps", len(formatted))
                return formatted

    # Tier 2: LLM checklist extraction from text (answer or blocks text)
    text_source = final_answer if (final_answer and len(final_answer.strip()) > 30) else ""
    if not text_source and blocks:
        text_source = "\n".join(b.get("text", "") for b in blocks[:50] if b.get("text"))

    if text_source and llm:
        try:
            ans_steps = _extract_checklist_steps_llm(text_source, llm)
            if ans_steps and len(ans_steps) >= 2:
                formatted = [_format_callout_headers(s) for s in ans_steps]
                logger.info("[Step Extraction] Tier 2 (LLM Checklist): Extracted %d steps", len(formatted))
                return formatted
        except Exception as e:
            logger.warning("Tier 2 LLM step extraction failed: %s", e)

    # Tier 3: Agentic section & step extraction for un-structured blocks
    if blocks and llm:
        sec_data = _extract_sections_with_steps_agentic(blocks, llm=llm, target_topic=final_answer)
        if sec_data:
            selected_secs = _select_best_sections_agentic(sec_data, final_answer, llm=llm)
            all_steps: list[str] = []
            for sec in selected_secs:
                prefix = f"[{sec['title']}] " if len(selected_secs) > 1 else ""
                for s in sec["steps"]:
                    all_steps.append(f"{prefix}{s}")
            if len(all_steps) >= 2:
                formatted = [_format_callout_headers(s) for s in all_steps]
                logger.info("[Step Extraction] Tier 3 (Agentic Sections): Extracted %d steps", len(formatted))
                return formatted

    # Tier 3: LLM extraction from text (last resort)
    if blocks:
        SKIP_PREFIXES = ("The image ", "The diagram ", "[logo]", "[figure]", "* **Logo")
        SKIP_SUBS = ("Published by", "Revision History", "Cost-free", "Guarantee", "HOW TO READ", "Preface")
        lines = []
        for b in blocks:
            t = b.get("text", "").strip()
            if not t or len(t) < 30:
                continue
            if any(t.startswith(p) for p in SKIP_PREFIXES):
                continue
            if any(s in t for s in SKIP_SUBS):
                continue
            lines.append(t)
        steps_source = "\n".join(lines)
    else:
        steps_source = final_answer

    logger.info("[Step Extraction] Tier 3 (LLM) on %d chars of text.", len(steps_source))
    return _extract_checklist_steps_llm(steps_source, llm)


def _find_associated_cad_diagram(text: str, config: dict) -> dict | None:
    import os
    from backend.storage.postgres_store import PostgresStore
    store = PostgresStore(config=config)
    try:
        docs = store.list_documents()
        cad_docs = [
            d for d in docs
            if d.get("file_type") == "pdf" and d.get("document_type") in ("cad_drawing", "circuit_diagram")
        ]
        if not cad_docs:
            return None
        
        import re
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        
        best_doc = None
        best_score = 0
        for doc in cad_docs:
            fname = doc.get("filename", "")
            fname_stem = os.path.splitext(fname)[0]
            fname_words = set(re.findall(r'\b\w{4,}\b', fname_stem.lower()))
            intersection = words.intersection(fname_words)
            score = len(intersection)
            if score > best_score:
                best_score = score
                best_doc = doc
                
        if best_doc and best_score > 0:
            return {
                "document_id": str(best_doc["document_id"]),
                "filename": best_doc["filename"],
                "page": 1,
                "fileType": "pdf"
            }
    except Exception as e:
        logger.warning("Error finding associated CAD diagram: %s", e)
    finally:
        store.close()
    return None


def _clean_title(filename: str) -> str:
    import os
    name = os.path.splitext(filename)[0]
    import re
    name = re.sub(r'^\d{8}_[A-Za-z0-9]+_\d{2}_(?:[A-Z0-9]{5,10}_)?', '', name)
    name = re.sub(r'^\d+_\d+_\d+_(?:[A-Z0-9]{5,10}_)?', '', name)
    name = re.sub(r'^[A-Za-z0-9]+_\d+_(?:[A-Z0-9]{5,10}_)?', '', name)
    name = name.replace('_', ' ').replace('-', ' ').strip()
    return name.title()


def _find_selected_disambiguation(message: str, options: list[dict]) -> dict | None:
    msg = message.strip().lower()
    for opt in options:
        idx_str = str(opt["index"])
        title = opt["title"].lower()
        if msg == idx_str or msg.startswith(idx_str) or title in msg:
            return opt
    return None


def _get_document_content_from_db(document_id: str, config: dict) -> str:
    from backend.storage.postgres_store import PostgresStore
    store = PostgresStore(config=config)
    try:
        blocks = store.get_blocks(document_id)
        if not blocks:
            return ""
        texts = [b.get("text", "") for b in blocks[:2000] if b.get("text")]
        return "\n".join(texts)
    except Exception as e:
        logger.warning("Error fetching doc content: %s", e)
    finally:
        store.close()
    return ""


def _is_new_query(message: str) -> bool:
    msg = message.strip().lower()
    confirmation_words = {
        "yes", "ok", "okay", "start", "sure", "proceed", "yep", "y", "done", 
        "completed", "complete", "next", "stop", "cancel", "exit", "show cad diagram", 
        "no, proceed with step 1", "no, thanks", "no", "1", "2", "3", "4", "5",
        "🚀 start guided process", "✅ step complete - next"
    }
    if msg in confirmation_words:
        return False
        
    import re
    if re.match(r'^\d+(\.|$)', msg):
        return False
        
    words = msg.split()
    if len(words) > 5:
        return True
    if not any(w in confirmation_words for w in words):
        return True
    return False


def _extract_all_citations(messages: list[BaseMessage]) -> list[dict]:
    tool_names = {}
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            tool_names[call["id"]] = call["name"]
            
    all_citations = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_name = getattr(m, "name", None) or tool_names.get(m.tool_call_id)
            if tool_name == "search_documents":
                try:
                    content = m.content.strip()
                    if not content:
                        continue
                    res = json.loads(content)
                    if isinstance(res, str):
                        res = json.loads(res)
                    citations = res.get("citations") or []
                    all_citations.extend(citations)
                except Exception:
                    pass
    return all_citations


def _get_primary_citation(messages: list[BaseMessage]) -> tuple[str | None, int | None]:
    citations = _extract_all_citations(messages)
    if not citations:
        return None, None

    # Find the user's query text from messages
    user_query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            user_query = m.content.strip().lower()
            break

    # Key industrial component keywords
    keywords = ["tailstock", "work holder", "workpiece", "wheel mounting", "coolant", "taper", "spindle", "chute"]
    matched_kws = [kw for kw in keywords if kw in user_query]

    if matched_kws and len(citations) > 1:
        for cit in citations:
            fname_lower = (cit.get("filename") or "").lower()
            if any(kw in fname_lower for kw in matched_kws):
                return str(cit.get("document_id")), cit.get("page")

    primary = citations[0]
    return str(primary.get("document_id")), primary.get("page")


def _get_page_content_from_db(document_id: str, page: int | str, config: dict) -> str:
    import os
    from backend.storage.postgres_store import PostgresStore
    store = PostgresStore(config=config)
    try:
        blocks = store.get_blocks(document_id)
        if not blocks:
            return ""
        
        try:
            page_no = int(page)
            is_numeric = True
        except (TypeError, ValueError):
            page_no = str(page).strip()
            is_numeric = False
            
        if is_numeric:
            page_blocks = [
                b for b in blocks
                if isinstance(b.get("source_ref"), dict) and (
                    b["source_ref"].get("page") == page_no or
                    b["source_ref"].get("slide") == page_no
                )
            ]
        else:
            page_blocks = [
                b for b in blocks
                if isinstance(b.get("source_ref"), dict) and b["source_ref"].get("sheet") == page_no
            ]
            
        texts = [b.get("text", "") for b in page_blocks if b.get("text")]
        return "\n".join(texts)
    except Exception as e:
        logger.warning("Error fetching page content from DB: %s", e)
    finally:
        store.close()
    return ""


PROCE_CLASSIFIER_PROMPT = (
    "You are a strict procedure classification judge for industrial machinery and technical manuals.\n"
    "Your task is to determine whether the user is asking to execute/follow an actionable, "
    "step-by-step physical/machine operational procedure (e.g. maintenance, cleaning, setup, assembly, "
    "disassembly, part replacement, calibration, wiring adjustments) based on their query and the retrieved manual excerpts.\n\n"
    "CLASSIFICATION RULES:\n"
    "1. Output 'YES' if the retrieved content contains sequential operational steps and the user query "
    "   wants to perform or follow this procedure.\n"
    "2. Output 'NO' if the user query is asking for informational explanation (e.g. explaining what an alarm/error code means, "
    "   component definitions, safety specs, dimensions, troubleshooting theory) even if numbered points appear in the text.\n\n"
    "Output strictly 'YES' or 'NO' with no extra punctuation or commentary."
)


def _has_procedural_step_markers(text: str) -> bool:
    """Fast structural pre-check: does the text contain sequential step markers or procedural action keywords?"""
    if not text:
        return False
    import re
    patterns = (
        r'\bstep\s*\d+',                         # "Step 1", "Step 2"
        r'\(\s*1\s*\).*?\(\s*2\s*\)',            # "(1)...(2)"
        r'^\s*1\.\s+.*?\n\s*2\.\s+',             # "1. ...\n 2. ..."
        r'\b1\.\s+.*?\b2\.\s+',                  # "1. ... 2. ..."
        r'\b1\.\d+\b.*?\b1\.\d+\b',              # "1.1 ... 1.2"
        r'\b(procedure|checklist|how to|steps to|instructions)\b',
        r'\b(clean|changeover|replace|install|assemble|disassemble|dismantle|calibrate|lubricate|maintenance|setup)\b',
    )
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def _classify_procedure_intent_llm(query: str, snippet: str, llm) -> bool:
    """Use a micro-LLM call (gpt-4o-mini, max_tokens=10) to dynamically determine if the query and chunks constitute an actionable procedure."""
    if not query or not snippet or not llm:
        return False
    try:
        classifier_messages = [
            SystemMessage(PROCE_CLASSIFIER_PROMPT),
            HumanMessage(f"USER QUERY:\n{query}\n\nRETRIEVED MANUAL EXCERPT:\n{snippet[:2000]}")
        ]
        try:
            resp = llm.invoke(classifier_messages, config={"max_tokens": 10, "temperature": 0.0})
        except TypeError:
            resp = llm.invoke(classifier_messages)
        decision = getattr(resp, "content", "").strip().upper()
        logger.info("🧠 [PROC CLASSIFIER] Query: %r | Decision: %r", query[:60], decision)
        return decision.startswith("YES")
    except Exception as e:
        logger.warning("Procedure classifier micro-LLM call failed: %s", e)
        return False


def _is_new_query(msg: str) -> bool:
    """Check if incoming user message is a new question vs a button click / continuation in guided assistant mode."""
    msg_lo = msg.strip().lower()
    # Numbered menu selection (e.g. "1", "1.", "1. Title", "2")
    import re
    if re.match(r'^\d+(\.|\s|$)', msg_lo):
        return False
    # Button clicks and step completion choices in guided process UI
    if any(tok in msg_lo for tok in (
        "start", "guided", "yes", "no", "next", "complete", "stop", "cancel",
        "proceed", "show cad", "option", "thanks", "done", "step complete", "continue"
    )):
        return False
    # If text is a new query (starts with question words or contains > 3 words)
    if any(msg_lo.startswith(w) for w in ("how", "what", "where", "why", "when", "can", "tell", "give")):
        return True
    return len(msg.split()) > 3


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
) -> dict:
    registry = registry if registry is not None else build_agent_registry()
    agent_cfg = (config.get("query") or {}).get("agent") or {}
    max_iterations = agent_cfg.get("max_iterations", 5)
    write_tools = set(agent_cfg.get("write_tools") or [])
    clarify_tools = set(agent_cfg.get("clarify_tools") or ["request_clarification"])

    if llm is None:
        llm = get_llm_for(config, agent_cfg)

    # --- Interactive Guided Assistant Mode Handler ---
    context_docs: list[str] = []  # Session-scoped document context list
    if session_id:
        try:
            from backend.storage.conversation_store import get_conversation_store
            store = get_conversation_store()

            # === Register active_document_id into session context (cumulative, never cleared) ===
            if active_document_id:
                try:
                    store.add_context_doc(session_id, active_document_id)
                except Exception:
                    pass

            state = store.get_interactive_state(session_id) or {}
            mode = state.get("mode")

            # Auto-register doc_id from active state or cached procedure
            _state_doc = (state.get("selected_option") or {}).get("document_id") or state.get("document_id")
            if _state_doc:
                try:
                    store.add_context_doc(session_id, _state_doc)
                except Exception:
                    pass

            _cached_proc = store.get_procedure_cache(session_id) if not mode else None
            if _cached_proc and _cached_proc.get("document_id"):
                try:
                    store.add_context_doc(session_id, _cached_proc.get("document_id"))
                except Exception:
                    pass

            # Load current session context doc list for use later in the pre-search interceptor
            try:
                context_docs = store.get_context_docs(session_id) or []
            except Exception:
                context_docs = []

            for _did in (_state_doc, _cached_proc.get("document_id") if _cached_proc else None, active_document_id):
                if _did and _did not in context_docs:
                    context_docs.append(_did)

            # ── Context Search Confirmation Handler ──────────────────────────────
            # Handles the case where a scoped search returned no answer and we asked
            # the user: "Search globally?" The user's reply (Yes/No) lands here.
            if mode == "context_search":
                _msg_lower = message.strip().lower()
                _orig_q = state.get("original_message", message)
                _yes_words = ("yes", "search globally", "search global", "✅", "sure", "ok", "proceed")
                _no_words  = ("no", "that's fine", "stop", "❌", "nope", "don't")
                if any(w in _msg_lower for w in _yes_words):
                    # User approved global search: clear context_search state, override message
                    store.set_interactive_state(session_id, None)
                    message = _orig_q
                    context_docs = []  # skip the context pre-search interceptor
                    # fall through to normal main LLM graph
                elif any(w in _msg_lower for w in _no_words):
                    # User declined — return polite refusal
                    store.set_interactive_state(session_id, None)
                    _no_ans = "Understood! I'll keep my answers limited to your current context. Feel free to ask anything else about those documents."
                    return {
                        "status": "done",
                        "answer": _no_ans,
                        "tool_calls": [],
                        "execution_trace": [],
                        "messages": [HumanMessage(message), AIMessage(content=_no_ans)],
                        "token_usage": {"total_tokens": 30, "input_tokens": 10, "output_tokens": 20},
                        "trace_id": None,
                    }
            # ── End Context Search Confirmation Handler ──────────────────────────

            # ── Procedure Context Agent ────────────────────────────────────────
            # When there is a cached procedure but no active guided session,
            # route the message through a dedicated secondary LLM agent that has
            # the full procedure context in its system prompt. It decides what to
            # do — resume, explain a section, summarise, answer a procedure Q, or
            # signal ACTION:SKIP to fall through to the main search agent.
            if not mode:
                try:
                    cache = store.get_procedure_cache(session_id)
                    if cache:
                        cached_sections = cache.get("sections", [])
                        cached_title    = cache.get("title", "Procedure")
                        cached_fname    = cache.get("filename", "")
                        cached_doc_id   = cache.get("document_id")
                        stopped_sec     = cache.get("stopped_at_sec", 0)
                        stopped_step    = cache.get("stopped_at_step", 0)

                        # ── Build procedure context block for system prompt ──
                        proc_lines = []
                        for i, sec in enumerate(cached_sections):
                            s_title = sec.get("title", f"Section {i+1}")
                            s_steps = sec.get("steps", [])
                            if s_steps:
                                proc_lines.append(f"\n{s_title} ({len(s_steps)} steps):")
                                for si, st in enumerate(s_steps):
                                    proc_lines.append(f"  Step {si+1}: {st.replace(chr(10), ' ').strip()}")
                            else:
                                proc_lines.append(f"\n{s_title} (steps not yet loaded)")
                        proc_context = "\n".join(proc_lines)

                        stopped_sec_title = (
                            cached_sections[stopped_sec].get("title", f"Section {stopped_sec+1}")
                            if stopped_sec < len(cached_sections) else "Unknown"
                        )
                        stopped_sec_total = len(
                            cached_sections[stopped_sec].get("steps", [])
                            if stopped_sec < len(cached_sections) else []
                        )

                        proc_agent_system = (
                            "You are a Procedure Context Agent for a technical manual assistant.\n"
                            "The user has been working on a guided procedure in this chat session.\n\n"
                            f"PROCEDURE: {cached_title}\n"
                            f"SOURCE: {cached_fname}\n\n"
                            "SECTIONS AND STEPS:\n"
                            f"{proc_context}\n\n"
                            f"CURRENT PROGRESS: Stopped at '{stopped_sec_title}', "
                            f"Step {stopped_step} of {stopped_sec_total}.\n\n"
                            "YOUR JOB: Respond ONLY to questions or messages related to this procedure.\n\n"
                            "RESPONSE RULES — follow exactly:\n"
                            "1. If the user wants to resume / continue / proceed / go back to the procedure, OR says '▶️ Resume Procedure': "
                            "   reply with exactly the token: ACTION:RESUME\n"
                            "2. If the user asks about a specific section (e.g. 'explain 1.2', 'steps in section 1.3', "
                            "   'what is 1.4 about'): reply with exactly: ACTION:EXPLAIN:<section_number> "
                            "   (e.g. ACTION:EXPLAIN:1.2)\n"
                            "3. If the user asks to see all sections, an overview, or a list of what's in the procedure: "
                            "   reply with exactly: ACTION:LIST\n"
                            "4. If the user asks a general question about the procedure (e.g. what tools are needed, "
                            "   what does a specific step mean, safety concerns) — answer it directly and concisely "
                            "   using the steps listed above. Do not output an ACTION token.\n"
                            "5. If the user's message has NOTHING to do with this procedure (e.g. a completely "
                            "   different topic, a new search question, a greeting): "
                            "   reply with exactly the token: ACTION:SKIP\n\n"
                            "CRITICAL: Output ONLY one of the ACTION tokens OR a direct prose answer. "
                            "Never mix an ACTION token with prose."
                        )

                        # ── Build message list: system + last 10 history turns + current ──
                        clean_hist = [
                            m for m in (conversation_history or [])
                            if isinstance(m, (HumanMessage, AIMessage))
                            and not getattr(m, "tool_calls", None)
                        ]
                        proc_agent_messages: list = [SystemMessage(proc_agent_system)]
                        proc_agent_messages += clean_hist[-10:]
                        proc_agent_messages.append(HumanMessage(message))

                        logger.info("[PROC AGENT] Invoking procedure context agent for session %s", session_id)
                        proc_resp = llm.invoke(proc_agent_messages)
                        proc_decision = getattr(proc_resp, "content", "").strip()
                        logger.info("[PROC AGENT] Decision: %r", proc_decision[:80])

                        # ── ACTION:SKIP → fall through to manual context search ──
                        if proc_decision.startswith("ACTION:SKIP"):
                            logger.info("[PROC AGENT] Falling through to context manual search")
                            if cached_doc_id and cached_doc_id not in context_docs:
                                context_docs.append(cached_doc_id)
                            # fall through to context pre-search interceptor

                        # ── ACTION:RESUME → restore state from cache ──
                        elif proc_decision.startswith("ACTION:RESUME"):
                            cur_sec = cached_sections[stopped_sec] if stopped_sec < len(cached_sections) else {}
                            cur_steps = cur_sec.get("steps", [])
                            sec_title = cur_sec.get("title", f"Section {stopped_sec + 1}")
                            total_steps = len(cur_steps)

                            # JIT-extract steps if this section hasn't been loaded yet
                            if not cur_steps and cached_doc_id:
                                p_start = cur_sec.get("page_start", 1)
                                p_end   = cur_sec.get("page_end", 999)
                                cur_steps = _extract_single_section_json_jit(
                                    cached_doc_id, sec_title, p_start, p_end, config, llm
                                )
                                cached_sections[stopped_sec]["steps"] = cur_steps
                                total_steps = len(cur_steps)

                            # Rollover if current section is fully completed
                            if stopped_step >= total_steps and total_steps > 0:
                                stopped_sec += 1
                                stopped_step = 0
                                if stopped_sec < len(cached_sections):
                                    cur_sec = cached_sections[stopped_sec]
                                    sec_title = cur_sec.get("title", f"Section {stopped_sec + 1}")
                                    cur_steps = cur_sec.get("steps", [])
                                    if not cur_steps and cached_doc_id:
                                        p_start = cur_sec.get("page_start", 1)
                                        p_end   = cur_sec.get("page_end", 999)
                                        cur_steps = _extract_single_section_json_jit(
                                            cached_doc_id, sec_title, p_start, p_end, config, llm
                                        )
                                        cached_sections[stopped_sec]["steps"] = cur_steps
                                    total_steps = len(cur_steps)
                                else:
                                    # Procedure is fully complete
                                    store.set_interactive_state(session_id, None)
                                    store.set_procedure_cache(session_id, None)
                                    answer = f"🎉 **{cached_title}** is completely finished!\n\nLet me know if you need help with anything else."
                                    return {
                                        "status": "done",
                                        "answer": answer,
                                        "tool_calls": [],
                                        "execution_trace": [],
                                        "messages": [HumanMessage(message), AIMessage(content=answer)],
                                        "token_usage": {"total_tokens": 50, "input_tokens": 20, "output_tokens": 30},
                                        "trace_id": None,
                                    }

                            # Restore full interactive session state
                            resume_state = {
                                "mode": "guided_assistant",
                                "stage": "active",
                                "title": cached_title,
                                "document_id": cached_doc_id,
                                "sections": cached_sections,
                                "selected_option": {
                                    "document_id": cached_doc_id,
                                    "filename": cached_fname,
                                    "title": cached_title,
                                },
                                "current_sec_idx":  stopped_sec,
                                "current_step_idx": stopped_step,
                                "current_idx":      stopped_step,
                                "steps":            cur_steps,
                                "start_page":       cur_sec.get("page_start", 1),
                                "current_page":     cur_sec.get("page_start", 1),
                            }
                            store.set_interactive_state(session_id, resume_state)

                            step_text = cur_steps[stopped_step] if stopped_step < total_steps else ""
                            step_page = cur_sec.get("page_start", 1)
                            import json as _json
                            step_tc = [{
                                "name": "get_page_context",
                                "args": {"document_id": cached_doc_id, "page": step_page},
                                "result": _json.dumps({"document_id": cached_doc_id, "filename": cached_fname, "page": step_page})
                            }] if cached_doc_id else []

                            answer = (
                                f"▶️ **Resuming: {cached_title}**\n\n"
                                f"**[{sec_title}] Step {stopped_step + 1} of {total_steps}:** {step_text}\n\n"
                                f"**Source:** Page {step_page}\n\n"
                                "Let me know when this step is complete."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "tool_calls": step_tc,
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": {"total_tokens": 80, "input_tokens": 50, "output_tokens": 30},
                                "trace_id": None,
                            }

                        # ── ACTION:EXPLAIN:<num> → show section steps from cache ──
                        elif proc_decision.startswith("ACTION:EXPLAIN:"):
                            sec_num = proc_decision.split("ACTION:EXPLAIN:", 1)[-1].strip()
                            matched = next(
                                (s for s in cached_sections if s.get("title", "").startswith(sec_num)),
                                None
                            )
                            if matched:
                                sec_title  = matched.get("title", sec_num)
                                sec_steps  = matched.get("steps", [])
                                # JIT-extract if not loaded
                                if not sec_steps and cached_doc_id:
                                    p_start = matched.get("page_start", 1)
                                    p_end   = matched.get("page_end", 999)
                                    sec_steps = _extract_single_section_json_jit(
                                        cached_doc_id, sec_title, p_start, p_end, config, llm
                                    )
                                    matched["steps"] = sec_steps

                                step_lines = []
                                for idx, st in enumerate(sec_steps):
                                    st_flat = st.replace("\n", " ").strip()
                                    step_lines.append(f"**{idx+1}.** {st_flat}")
                                steps_text = "\n\n".join(step_lines) if step_lines else "_No steps extracted yet._"
                                answer = (
                                    f"### {sec_title} ({len(sec_steps)} Steps)\n\n"
                                    f"{steps_text}\n\n"
                                    f"*(From: {cached_fname})*"
                                )
                                return {
                                    "status": "done",
                                    "answer": answer,
                                    "tool_calls": [],
                                    "execution_trace": [],
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": {"total_tokens": 60, "input_tokens": 20, "output_tokens": 40},
                                    "trace_id": None,
                                }
                            # section not found — fall through to main agent

                        # ── ACTION:LIST → show all sections overview ──
                        elif proc_decision.startswith("ACTION:LIST"):
                            sec_lines = []
                            for i, sec in enumerate(cached_sections):
                                s_title  = sec.get("title", f"Section {i+1}")
                                s_steps  = sec.get("steps", [])
                                n_label  = f"{len(s_steps)} steps" if s_steps else "not yet loaded"
                                status   = "✅" if i < stopped_sec else ("🔄" if i == stopped_sec else "⏳")
                                sec_lines.append(f"{status} **{s_title}** — {n_label}")
                            answer = (
                                f"## {cached_title}\n\n"
                                + "\n\n".join(sec_lines)
                                + f"\n\n*(Source: {cached_fname})*"
                            )
                            return {
                                "status": "done",
                                "answer": answer,
                                "tool_calls": [],
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": {"total_tokens": 50, "input_tokens": 15, "output_tokens": 35},
                                "trace_id": None,
                            }

                        # ── Direct prose answer from LLM ──
                        elif proc_decision and not proc_decision.startswith("ACTION:"):
                            return {
                                "status": "done",
                                "answer": proc_decision,
                                "tool_calls": [],
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=proc_decision)],
                                "token_usage": {"total_tokens": 80, "input_tokens": 40, "output_tokens": 40},
                                "trace_id": None,
                            }

                except Exception as proc_agent_err:
                    logger.warning("[PROC AGENT] Failed, falling through to main agent: %s", proc_agent_err)
            # ── End Procedure Context Agent ────────────────────────────────────



            if mode == "guided_assistant" and (state.get("stage") in ("active", "disambiguation", "overview", "section_disambiguation", "procedure_offer", "trouble_global_search_pending") or not _is_new_query(message)):
                with _using_trace_sink() as _trace_sink, usage.using_sink() as sink:
                    def _get_tu(ans_str: str) -> dict:
                        tu = sink.totals(config=config)
                        # No longer generating fake tokens. If LLM was not used, it returns 0.
                        if tu.get("total_tokens", 0) == 0:
                            tu = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
                        return tu

                    msg_lower = message.strip().lower()
                    doc_id = (state.get("selected_option") or {}).get("document_id") or state.get("document_id")
                    if not doc_id and conversation_history:
                        doc_id, _ = _get_primary_citation(conversation_history)

                    existing_steps = state.get("steps") or []
                    blocks = None
                    if doc_id:
                        try:
                            from backend.storage.postgres_store import PostgresStore
                            store_pg = PostgresStore(config=config)
                            try:
                                blocks = store_pg.get_blocks(doc_id)
                            finally:
                                store_pg.close()
                        except Exception as e:
                            logger.warning("PostgresStore block fetch skipped in interactive handler: %s", e)
                            blocks = None

                    stage = state.get("stage")

                    # ── Phase 1: First offer → directly to section discovery ──────────────
                    if stage == "procedure_offer":
                        msg_clean = message.strip().lower()
                        is_continue = any(w in msg_clean for w in ("continue", "yes", "start", "proceed", "yep", "y", "ok", "sure"))
                        if is_continue:
                            doc_id = state.get("document_id") or (state.get("selected_option") or {}).get("document_id")
                            title = state.get("title", "Procedure")

                            # Scan all blocks for sub-section headings (fast regex, LLM fallback only)
                            sections: list[dict] = []
                            all_blocks_for_scan: list[dict] = []
                            if doc_id:
                                try:
                                    from backend.storage.postgres_store import PostgresStore
                                    store_pg2 = PostgresStore(config=config)
                                    try:
                                        all_blocks_for_scan = store_pg2.get_blocks(doc_id)
                                    finally:
                                        store_pg2.close()
                                except Exception as e:
                                    logger.warning("Block fetch for section discovery failed: %s", e)

                            if all_blocks_for_scan:
                                sections = _scan_pages_for_sections(all_blocks_for_scan, llm=llm)

                            has_sections = len(sections) > 1

                            if has_sections:
                                # Multi-section: show section names, transition to overview for JIT extraction
                                state["stage"] = "overview"
                                state["sections"] = sections
                                state["current_sec_idx"] = 0
                                state["current_step_idx"] = 0
                                state["current_idx"] = 0
                                state["steps"] = []
                                store.set_interactive_state(session_id, state)

                                sec_list_lines = "\n".join(
                                    f"{i+1}. **{sec['title']}**"
                                    for i, sec in enumerate(sections)
                                )
                                first_sec_title = sections[0]["title"]
                                answer = (
                                    f"I found **{len(sections)} sections** in this procedure:\n\n"
                                    f"{sec_list_lines}\n\n"
                                    f"Shall we proceed with **{first_sec_title}**?"
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Shall we proceed with Section 1?",
                                    "options": ["Proceed to Section 1", "Stop"],
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                            else:
                                # Flat procedure (no sub-sections): page-by-page LLM extraction → active
                                proc_start_page = state.get("retrieval_min_page", 1)
                                p_end = 999
                                flat_steps: list[str] = []
                                if doc_id:
                                    flat_steps = _extract_steps_from_text_or_blocks(doc_id, title, config, llm, start_page=proc_start_page)

                                step1 = flat_steps[0] if flat_steps else "No steps found."
                                cur_page = _find_exact_step_page(step1, all_blocks_for_scan, fallback_page=proc_start_page) if step1 and all_blocks_for_scan else proc_start_page
                                p_start = cur_page

                                flat_section = [{"title": title, "page_start": p_start, "page_end": p_end, "steps": flat_steps}]
                                state["stage"] = "active"
                                state["sections"] = flat_section
                                state["current_sec_idx"] = 0
                                state["current_step_idx"] = 0
                                state["current_idx"] = 0
                                state["steps"] = flat_steps
                                state["current_page"] = p_start
                                state["start_page"] = p_start
                                store.set_interactive_state(session_id, state)

                                fname_val2 = (state.get("selected_option") or {}).get("filename", "")
                                step_tc = []
                                if doc_id:
                                    import json as _json2
                                    step_tc = [{
                                        "name": "get_page_context",
                                        "args": {"document_id": doc_id, "page": p_start},
                                        "result": _json2.dumps({"document_id": doc_id, "filename": fname_val2, "page": p_start})
                                    }]

                                answer = (
                                    f"⚠️ **SAFETY MANDATE:** Ensure the main power is TURNED OFF before proceeding.\n\n"
                                    f"**[{title}] Step 1 of {len(flat_steps)}:** {step1}\n\n"
                                    f"**Source:** Page {p_start}\n\n"
                                    "Let me know when Step 1 is complete."
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Is this step completed?",
                                    "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                    "tool_calls": step_tc,
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                        else:
                            store.set_interactive_state(session_id, None)
                            answer = "No problem! Let me know if you need anything else."
                            return {
                                "status": "done",
                                "answer": answer,
                                "tool_calls": [],
                                "execution_trace": list(_trace_sink),
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }

                    # ── Existing stages ──────────────────────────────────────────────────
                    elif stage == "disambiguation":
                        opts = state.get("disambiguation_options", [])
                        sel = _find_selected_disambiguation(message, opts)
                        if sel:
                            doc_id = sel["document_id"]
                            fname = sel["filename"]
                            title = sel["title"]
                            
                            # Load document content to generate overview and extract steps
                            doc_text = _get_document_content_from_db(doc_id, config)
                            
                            overview_prompt = (
                                "You are a technical document assistant. Write a short, high-level overview of the "
                                f"procedure for '{title}' described in the text below.\n"
                                "Summarize the main sections or objective in 2-3 sentences.\n\n"
                                f"Text:\n{doc_text[:8000]}\n"
                            )
                            overview_resp = llm.invoke(overview_prompt)
                            overview = getattr(overview_resp, "content", "").strip()
                            
                            steps = _extract_steps_from_text_or_blocks(doc_id, "", config, llm, start_page=state.get("retrieval_min_page", 1))
                            sections = [{"title": title, "steps": steps}] if steps else []
                            if blocks:
                                try:
                                    sec_extracted = _extract_sections_with_steps(blocks)
                                    if sec_extracted:
                                        sections = sec_extracted
                                except Exception:
                                    pass

                            total_steps = sum(len(sec.get("steps", [])) for sec in sections) if sections else len(steps)
                            first_sec_steps = sections[0].get("steps", []) if sections else steps
                            
                            if len(steps) >= 2 or total_steps >= 2:
                                state["stage"] = "overview"
                                state["selected_option"] = sel
                                state["sections"] = sections
                                state["current_sec_idx"] = 0
                                state["current_step_idx"] = 0
                                state["steps"] = first_sec_steps
                                state["current_idx"] = 0
                                state["title"] = title
                                store.set_interactive_state(session_id, state)
                                
                                sec_list_str = "\n".join(
                                    f"- **{sec['title']}** ({len(sec['steps'])} step{'s' if len(sec['steps']) > 1 else ''})"
                                    for sec in sections
                                )
                                first_sec_title = sections[0]["title"] if sections else title

                                answer = (
                                    f"### Overview of {title}\n"
                                    f"{overview}\n\n"
                                    f"This procedure consists of {len(sections)} section{'s' if len(sections) > 1 else ''} ({total_steps} total steps):\n\n"
                                    f"{sec_list_str}\n\n"
                                    f"We will guide you section-by-section, starting with **Section {first_sec_title}**. Shall we start?"
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "When you are ready, shall we start the process?",
                                    "options": ["🚀 Start Guided Process", "No, thanks"],
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                            else:
                                store.set_interactive_state(session_id, None)
                                answer = f"I could not extract structured steps for {title}. Let me know if you need anything else!"
                                return {
                                    "status": "done",
                                    "answer": answer,
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                        else:
                            store.set_interactive_state(session_id, None)
                            answer = "Okay! Let me know if you need help with anything else."
                            return {
                                "status": "done",
                                "answer": answer,
                                "tool_calls": [],
                                "execution_trace": list(_trace_sink),
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        
                    elif stage == "section_disambiguation":
                        sections = state.get("sections", [])
                        msg_clean = message.strip().lower()
                        selected_sec = None
                        for idx, sec in enumerate(sections):
                            opt_str = str(idx + 1)
                            title_lower = sec["title"].lower()
                            sec_num = sec["title"].split()[0].lower()
                            if msg_clean == opt_str or opt_str in msg_clean or title_lower in msg_clean or sec_num in msg_clean:
                                selected_sec = sec
                                break
                        if not selected_sec and sections:
                            for idx, sec in enumerate(sections):
                                if str(idx + 1) in msg_clean:
                                    selected_sec = sec
                                    break

                        if selected_sec:
                            steps = selected_sec["steps"]
                            sec_title = selected_sec["title"]
                            total_steps = len(steps)
                            step1 = steps[0]

                            sel = state.get("selected_option") or {}
                            doc_id = sel.get("document_id") or state.get("document_id")
                            fname_val = sel.get("filename", "")

                            cur_page = _find_exact_step_page(step1, blocks, fallback_page=state.get("start_page", 1))

                            state["stage"] = "active"
                            state["steps"] = steps
                            state["current_idx"] = 0
                            state["title"] = sec_title
                            state["start_page"] = cur_page
                            state["current_page"] = cur_page
                            store.set_interactive_state(session_id, state)

                            step_tc = []
                            if doc_id:
                                import json as _json
                                step_tc = [{
                                    "name": "get_page_context",
                                    "args": {"document_id": doc_id, "page": cur_page},
                                    "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": cur_page})
                                }]

                            answer = (
                                f"⚠️ **SAFETY MANDATE:** Ensure the main power is TURNED OFF, wheel spindle is stopped, and lockout/tagout is applied before opening machine covers.\n\n"
                                f"**[{sec_title}] Step 1 of {total_steps}:** {step1}\n\n"
                                f"**Source:** Page {cur_page}\n\n"
                                f"Let me know when Step 1 is complete or if you have any questions."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "Stop checklist"],
                                "tool_calls": step_tc,
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        else:
                            store.set_interactive_state(session_id, None)
                            answer = "Guided process cancelled. Let me know if you need anything else!"
                            return {
                                "status": "done",
                                "answer": answer,
                                "tool_calls": [],
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }

                    elif stage == "overview":
                        msg_clean = message.strip().lower()
                        is_start = any(w in msg_clean for w in ("start", "yes", "ready", "yep", "y", "go", "guided", "proceed", "section", "ok", "sure", "continue"))
                        if is_start:
                            sel = state.get("selected_option", {})
                            doc_id = sel.get("document_id") or state.get("document_id")
                            
                            # Lock procedure boundary: start_page to end_page
                            if blocks:
                                start_page, end_page = _detect_procedure_boundaries(blocks)
                                state["start_page"] = start_page
                                state["end_page"] = end_page
                                state["current_page"] = start_page

                            sections = state.get("sections", [])
                            if not sections:
                                steps = state.get("steps", [])
                                if doc_id and len(steps) < 2:
                                    proc_ctx = state.get("title") or state.get("query") or ""
                                    fresh_steps = _extract_steps_from_text_or_blocks(doc_id, proc_ctx, config, llm, start_page=state.get("retrieval_min_page", 1))
                                    if fresh_steps and len(fresh_steps) >= 2:
                                        steps = fresh_steps
                                sections = [{"title": state.get("title", "Procedure"), "steps": steps}]
                                state["sections"] = sections

                            current_sec_idx = 0
                            current_sec = sections[0] if sections else {"title": state.get("title", "Procedure"), "page_start": 1, "page_end": 999, "steps": []}
                            sec_title = current_sec.get("title", "Procedure")
                            
                            # JIT Section JSON extraction if section steps are empty
                            steps = current_sec.get("steps", [])
                            if not steps and doc_id:
                                p_start = current_sec.get("page_start", 1)
                                p_end = current_sec.get("page_end", 999)
                                steps = _extract_single_section_json_jit(doc_id, sec_title, p_start, p_end, config, llm)
                                if not steps:
                                    steps = state.get("steps", [])
                                current_sec["steps"] = steps

                            total_sec_steps = len(steps)
                            step1 = steps[0] if steps else ""
                            title = state.get("title", "Procedure")

                            cur_page = current_sec.get("page_start") or _find_exact_step_page(step1, blocks, fallback_page=state.get("start_page", 1))

                            cad_info = _find_associated_cad_diagram(title, config)
                            state["cad_info"] = cad_info
                            state["stage"] = "active"
                            state["current_sec_idx"] = 0
                            state["current_step_idx"] = 0
                            state["current_idx"] = 0
                            state["steps"] = steps
                            state["current_page"] = cur_page
                            store.set_interactive_state(session_id, state)
                            
                            fname_val = (state.get("selected_option") or {}).get("filename", "")
                            step_tc = []
                            if doc_id:
                                import json as _json
                                step_tc = [{
                                    "name": "get_page_context",
                                    "args": {"document_id": doc_id, "page": cur_page},
                                    "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": cur_page})
                                }]

                            answer = (
                                f"⚠️ **SAFETY MANDATE:** Ensure the main power is TURNED OFF, wheel spindle is stopped, and lockout/tagout is applied before opening machine covers.\n\n"
                                f"**[{sec_title}]**\n"
                                f"**Step 1 of {total_sec_steps}:** {step1}\n\n"
                                f"**Source:** Page {cur_page}\n\n"
                                "Let me know when Step 1 is complete or if you would like to view the full summary for this section."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "tool_calls": step_tc,
                                "execution_trace": list(_trace_sink),
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        else:
                            store.set_interactive_state(session_id, None)
                            answer = "Guided process cancelled. Let me know if you need anything else!"
                            return {
                                "status": "done",
                                "answer": answer,
                                "tool_calls": [],
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                            
                    elif stage == "trouble_global_search_pending":
                        msg_clean = message.strip().lower()
                        _yes_words = ("yes", "search globally", "search global", "✅", "sure", "ok", "proceed", "yep", "y")
                        _no_words = ("no", "that's fine", "stop", "❌", "nope", "don't")
                        if any(w in msg_clean for w in _yes_words):
                            state["stage"] = "active"
                            store.set_interactive_state(session_id, state)
                            
                            from backend.retrieval.search_documents import SearchDocumentsTool as _SDT
                            _global_result = _SDT().run(
                                query=state.get("trouble_query", message),
                                document_scope=[],
                                session_id=session_id
                            )
                            _global_answer = (_global_result.get("answer") or "").strip()
                            if not _global_answer:
                                _global_answer = "I could not find relevant information across the manuals."
                            
                            current_step_idx = state.get("current_step_idx", state.get("current_idx", 0))
                            sections = state.get("sections", [])
                            current_sec_idx = state.get("current_sec_idx", 0)
                            total_sec_steps = len(sections[current_sec_idx].get("steps", [])) if current_sec_idx < len(sections) else 0
                                
                            answer = (
                                f"{_global_answer}\n\n"
                                f"*(Still on **Step {current_step_idx + 1} of {total_sec_steps}**)*\n"
                                "Let me know when you have completed this step or if you need more help."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "tool_calls": [],
                                "execution_trace": list(_trace_sink),
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        else:
                            state["stage"] = "active"
                            store.set_interactive_state(session_id, state)
                            current_step_idx = state.get("current_step_idx", state.get("current_idx", 0))
                            sections = state.get("sections", [])
                            current_sec_idx = state.get("current_sec_idx", 0)
                            total_sec_steps = len(sections[current_sec_idx].get("steps", [])) if current_sec_idx < len(sections) else 0
                            
                            answer = (
                                f"Understood, keeping answers restricted to the current manual.\n\n"
                                f"*(Still on **Step {current_step_idx + 1} of {total_sec_steps}**)*\n"
                                "Let me know when you have completed this step or if you need more help."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "tool_calls": [],
                                "execution_trace": list(_trace_sink),
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                            
                    elif stage == "active":
                        msg_clean = message.strip().lower()
                        sections = state.get("sections", [])
                        current_sec_idx = state.get("current_sec_idx", 0)
                        current_step_idx = state.get("current_step_idx", state.get("current_idx", 0))
                        
                        if not sections:
                            steps = state.get("steps", [])
                            sections = [{"title": state.get("title", "Procedure"), "steps": steps}]

                        if current_sec_idx >= len(sections):
                            current_sec_idx = len(sections) - 1

                        current_sec = sections[current_sec_idx]
                        sec_title = current_sec.get("title", "Procedure")
                        steps = current_sec.get("steps", [])
                        total_sec_steps = len(steps)
                        cad_info = state.get("cad_info")
                        title = state.get("title", "Procedure")
                        sel = state.get("selected_option") or {}
                        doc_id = sel.get("document_id") or state.get("document_id")
                        fname_val = sel.get("filename", "")

                        # Handle Option B: Full Section Summary Request
                        if "summary" in msg_clean or "full section" in msg_clean:
                            step_lines = []
                            for idx, st in enumerate(steps):
                                # Replace internal newlines with a space so each step stays on one block
                                st_flat = st.replace("\n", " ").strip()
                                step_lines.append(f"**{idx+1}.** {st_flat}")
                            sec_steps_str = "\n\n".join(step_lines)
                            answer = (
                                f"### Full Summary: {sec_title} ({total_sec_steps} Steps)\n\n"
                                f"{sec_steps_str}\n\n"
                                f"You can proceed step-by-step or continue to the next section when ready."
                            )
                            cur_page = current_sec.get("page_start", state.get("current_page", 1))
                            step_tc = []
                            if doc_id:
                                import json as _json
                                step_tc = [{
                                    "name": "get_page_context",
                                    "args": {"document_id": doc_id, "page": cur_page},
                                    "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": cur_page})
                                }]
                            
                            # Fast-forward progress so that stopping now treats all steps as viewed
                            if total_sec_steps > 0:
                                state["current_step_idx"] = total_sec_steps - 1
                                state["current_idx"] = total_sec_steps - 1
                                store.set_interactive_state(session_id, state)

                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "How would you like to proceed?",
                                "options": ["✅ Step Complete - Next", "➡️ Proceed to Next Section", "Stop checklist"],
                                "tool_calls": step_tc,
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }

                        if "next section" in msg_clean or "proceed to next" in msg_clean:
                            next_sec_idx = current_sec_idx + 1
                            if next_sec_idx < len(sections):
                                next_sec = sections[next_sec_idx]
                                next_steps = next_sec.get("steps", [])
                                next_sec_title = next_sec.get("title", f"Section {next_sec_idx + 1}")

                                # JIT extract steps if not yet loaded for this section
                                if not next_steps and doc_id:
                                    p_start = next_sec.get("page_start", 1)
                                    p_end = next_sec.get("page_end", 999)
                                    next_steps = _extract_single_section_json_jit(doc_id, next_sec_title, p_start, p_end, config, llm)
                                    next_sec["steps"] = next_steps
                                    sections[next_sec_idx] = next_sec
                                    state["sections"] = sections

                                step1 = next_steps[0] if next_steps else ""
                                step_page = next_sec.get("page_start") or _find_exact_step_page(step1, blocks, fallback_page=state.get("current_page", 1))

                                state["current_sec_idx"] = next_sec_idx
                                state["current_step_idx"] = 0
                                state["current_idx"] = 0
                                state["steps"] = next_steps
                                state["current_page"] = step_page
                                store.set_interactive_state(session_id, state)

                                step_tc = []
                                if doc_id:
                                    import json as _json
                                    step_tc = [{
                                        "name": "get_page_context",
                                        "args": {"document_id": doc_id, "page": step_page},
                                        "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": step_page})
                                    }]

                                answer = (
                                    f"➡️ **Starting Section {next_sec_title}** ({len(next_steps)} steps).\n\n"
                                    f"⚠️ **SAFETY MANDATE:** Ensure the main power is TURNED OFF before proceeding.\n\n"
                                    f"**[{next_sec_title}] Step 1 of {len(next_steps)}:** {step1}\n\n"
                                    f"**Source:** Page {step_page}\n\n"
                                    "Let me know when Step 1 is complete."
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Is this step completed?",
                                    "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                    "tool_calls": step_tc,
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }


                        if message == "Show CAD Diagram":
                            step_text = steps[current_step_idx] if current_step_idx < total_sec_steps else ""
                            cur_page = _find_exact_step_page(step_text, blocks, fallback_page=state.get("current_page") or state.get("start_page", 1))
                            state["current_page"] = cur_page
                            store.set_interactive_state(session_id, state)
                            step_tc = []
                            if doc_id:
                                import json as _json
                                step_tc = [{
                                    "name": "get_page_context",
                                    "args": {"document_id": doc_id, "page": cur_page},
                                    "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": cur_page})
                                }]
                            answer = (
                                f"Showing CAD diagram `{cad_info.get('filename')}`.\n\n"
                                f"**[{sec_title}]**\n"
                                f"**Step {current_step_idx + 1} of {total_sec_steps}:** {step_text}\n\n"
                                f"**Source:** Page {cur_page}\n\n"
                                "Let me know when this step is complete."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "cad_diagrams": [cad_info],
                                "tool_calls": step_tc,
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        
                        if message == "No, proceed with Step 1":
                            step_text = steps[current_step_idx] if current_step_idx < total_sec_steps else ""
                            cur_page = _find_exact_step_page(step_text, blocks, fallback_page=state.get("current_page") or state.get("start_page", 1))
                            state["current_page"] = cur_page
                            store.set_interactive_state(session_id, state)
                            step_tc = []
                            if doc_id:
                                import json as _json
                                step_tc = [{
                                    "name": "get_page_context",
                                    "args": {"document_id": doc_id, "page": cur_page},
                                    "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": cur_page})
                                }]
                            answer = (
                                f"**[{sec_title}]**\n"
                                f"**Step {current_step_idx + 1} of {total_sec_steps}:** {step_text}\n\n"
                                f"**Source:** Page {cur_page}\n\n"
                                "Let me know when this step is complete."
                            )
                            return {
                                "status": "needs_clarification",
                                "answer": answer,
                                "question": "Is this step completed?",
                                "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                "tool_calls": step_tc,
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }
                        
                        if any(w in msg_clean for w in ("stop", "cancel", "exit")):
                            try:
                                all_sections = state.get("sections", [])
                                cur_sec_idx = state.get("current_sec_idx", 0)
                                cur_step_idx = state.get("current_step_idx", state.get("current_idx", 0))
                                proc_title = state.get("title") or (state.get("selected_option") or {}).get("title", "Procedure")

                                # ── Save procedure context to cache for follow-up questions ──
                                try:
                                    cache_payload = {
                                        "title": proc_title,
                                        "document_id": state.get("document_id") or (state.get("selected_option") or {}).get("document_id"),
                                        "filename": (state.get("selected_option") or {}).get("filename", ""),
                                        "sections": all_sections,
                                        "stopped_at_sec": cur_sec_idx,
                                        "stopped_at_step": cur_step_idx + 1,  # store 1-based count of completed steps
                                    }
                                    store.set_procedure_cache(session_id, cache_payload)
                                except Exception as ce:
                                    logger.warning("Failed to save procedure cache: %s", ce)

                                summary_lines = [f"## 📋 Progress Summary — {proc_title}\n"]

                                for i, sec in enumerate(all_sections):
                                    s_title = sec.get("title", f"Section {i+1}")
                                    s_steps = sec.get("steps", [])
                                    n_steps = len(s_steps)

                                    if i < cur_sec_idx:
                                        summary_lines.append(f"✅ **{s_title}** — All {n_steps} steps completed")
                                    elif i == cur_sec_idx:
                                        done_count = cur_step_idx + 1  # +1: current shown step counts as completed
                                        if done_count == n_steps and n_steps > 0:
                                            summary_lines.append(f"✅ **{s_title}** — All {n_steps} steps completed *(stopped here)*")
                                        else:
                                            summary_lines.append(
                                                f"🔄 **{s_title}** — {done_count} of {n_steps if n_steps else '?'} steps completed *(stopped here)*"
                                            )
                                        if s_steps and done_count > 0:
                                            for si in range(min(done_count, len(s_steps))):
                                                st_flat = s_steps[si].replace("\n", " ").strip()[:120]
                                                summary_lines.append(f"  - ~~{st_flat}~~")
                                    else:
                                        summary_lines.append(f"⏳ **{s_title}** — not started")

                                is_100_percent_done = False
                                if cur_sec_idx >= len(all_sections) - 1:
                                    last_sec = all_sections[-1]
                                    if (cur_step_idx + 1) >= len(last_sec.get("steps", [])):
                                        is_100_percent_done = True
                                
                                if is_100_percent_done:
                                    prompt = (
                                        "You are a helpful and celebratory technical assistant.\n"
                                        f"The user has just successfully completed 100% of the technical procedure: '{proc_title}'.\n"
                                        "Please generate a short, congratulatory final message (2-3 sentences) letting them know they have "
                                        "completely finished all sections and steps, and asking if they need help with anything else. DO NOT use placeholders like [Your Name]."
                                    )
                                    try:
                                        answer = llm.invoke(prompt).content.strip()
                                    except Exception:
                                        answer = f"🎉 **Procedure Complete!** You have finished all steps for '{proc_title}'. Let me know if you need anything else."
                                    options = []
                                    store.set_interactive_state(session_id, None)
                                    store.set_procedure_cache(session_id, None)
                                    status_val = "done"
                                else:
                                    summary_lines.append("\n\nLet me know if you'd like to resume, or ask me anything about a specific section.")
                                    answer = "\n\n".join(summary_lines)
                                    options = ["▶️ Resume Procedure"]
                                    store.set_interactive_state(session_id, None)
                                    status_val = "needs_clarification"
                            except Exception:
                                answer = "Guided process stopped. Let me know if you need anything else!"
                                options = []
                                status_val = "done"
                                store.set_interactive_state(session_id, None)

                            return {
                                "status": status_val,
                                "answer": answer,
                                "options": options,
                                "tool_calls": [],
                                "execution_trace": [],
                                "messages": [HumanMessage(message), AIMessage(content=answer)],
                                "token_usage": _get_tu(answer),
                                "trace_id": None,
                            }

                            
                        is_question = any(msg_clean.startswith(w) for w in ("how", "what", "which", "why", "where", "when", "can", "tell", "explain", "is", "are", "do", "does", "h=")) or "?" in message
                        is_done = not is_question and (
                            any(w in msg_clean for w in ("✅", "step complete", "next step", "next", "proceed", "done with this step", "step done", "completed step"))
                            or msg_clean in ("yes", "done", "ok", "next", "completed")
                        )
                        if is_done:
                            next_step_idx = current_step_idx + 1
                            if next_step_idx < total_sec_steps:
                                next_step = steps[next_step_idx]
                                last_page = state.get("current_page") or state.get("start_page", 1)
                                step_page = _find_exact_step_page(next_step, blocks, fallback_page=last_page)
                                state["current_step_idx"] = next_step_idx
                                state["current_idx"] = next_step_idx
                                state["current_page"] = step_page
                                store.set_interactive_state(session_id, state)
                                step_tc = []
                                if doc_id:
                                    import json as _json
                                    step_tc = [{
                                        "name": "get_page_context",
                                        "args": {"document_id": doc_id, "page": step_page},
                                        "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": step_page})
                                    }]

                                answer = (
                                    f"**[{sec_title}]**\n"
                                    f"**Step {next_step_idx + 1} of {total_sec_steps}:** {next_step}\n\n"
                                    f"**Source:** Page {step_page}\n\n"
                                    "Let me know when this step is complete."
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Is this step completed?",
                                    "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                    "tool_calls": step_tc,
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                            else:
                                # Section is complete! Check if there is a next section
                                next_sec_idx = current_sec_idx + 1
                                if next_sec_idx < len(sections):
                                    next_sec = sections[next_sec_idx]
                                    next_sec_title = next_sec.get("title", f"Section {next_sec_idx + 1}")
                                    next_steps = next_sec.get("steps", [])
                                    if not next_steps and doc_id:
                                        p_start = next_sec.get("page_start", 1)
                                        p_end = next_sec.get("page_end", 999)
                                        next_steps = _extract_single_section_json_jit(doc_id, next_sec_title, p_start, p_end, config, llm)
                                        next_sec["steps"] = next_steps

                                    step1 = next_steps[0] if next_steps else ""
                                    step_page = next_sec.get("page_start") or _find_exact_step_page(step1, blocks, fallback_page=state.get("current_page", 1))

                                    state["current_sec_idx"] = next_sec_idx
                                    state["current_step_idx"] = 0
                                    state["current_idx"] = 0
                                    state["steps"] = next_steps
                                    state["current_page"] = step_page
                                    store.set_interactive_state(session_id, state)

                                    step_tc = []
                                    if doc_id:
                                        import json as _json
                                        step_tc = [{
                                            "name": "get_page_context",
                                            "args": {"document_id": doc_id, "page": step_page},
                                            "result": _json.dumps({"document_id": doc_id, "filename": fname_val, "page": step_page})
                                        }]

                                    answer = (
                                        f"🎉 **{sec_title} Complete!**\n\n"
                                        f"Next up: **Section {next_sec_title}** ({len(next_steps)} steps).\n\n"
                                        f"**[{next_sec_title}]**\n"
                                        f"**Step 1 of {len(next_steps)}:** {step1}\n\n"
                                        f"**Source:** Page {step_page}\n\n"
                                        "Let me know when Step 1 is complete."
                                    )
                                    return {
                                        "status": "needs_clarification",
                                        "answer": answer,
                                        "question": "Is this step completed?",
                                        "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                        "tool_calls": step_tc,
                                        "execution_trace": list(_trace_sink),
                                        "messages": [HumanMessage(message), AIMessage(content=answer)],
                                        "token_usage": _get_tu(answer),
                                        "trace_id": None,
                                    }
                                else:
                                    # All sections complete!
                                    store.set_interactive_state(session_id, None)
                                    
                                    all_steps_flat = []
                                    for sec in sections:
                                        all_steps_flat.extend(sec.get("steps", []))

                                    summary_prompt = (
                                        "You are a helpful and celebratory technical assistant.\n"
                                        f"The user has just successfully completed 100% of the technical procedure: '{title}'.\n"
                                        "Please generate a short, congratulatory final message (2-3 sentences) letting them know they have "
                                        "completely finished all sections and steps, and asking if they need help with anything else. DO NOT use placeholders like [Your Name]."
                                    )
                                    try:
                                        summary_resp = llm.invoke(summary_prompt)
                                        answer = getattr(summary_resp, "content", "").strip()
                                    except Exception:
                                        answer = f"🎉 **Guided process complete!** All {len(sections)} sections and {len(all_steps_flat)} steps of '{title}' were completed successfully. Let me know if you need anything else."
                                    
                                    store.set_procedure_cache(session_id, None)
                                    return {
                                        "status": "done",
                                        "answer": answer,
                                        "tool_calls": [],
                                        "execution_trace": list(_trace_sink),
                                        "messages": [HumanMessage(message), AIMessage(content=answer)],
                                        "token_usage": _get_tu(answer),
                                        "trace_id": None,
                                    }
                        else:
                            # === PROCEDURE AGENT: Live Structured JSON & Context Memory Handler ===
                            sel = state.get("selected_option") or {}
                            doc_id = sel.get("document_id") or state.get("document_id")
                            fname_val = sel.get("filename") or state.get("filename") or ""

                            # 1. Build Structured Procedure JSON & Progress Memory Context
                            proc_lines = []
                            for i, sec in enumerate(sections):
                                s_title = sec.get("title", f"Section {i+1}")
                                s_steps = sec.get("steps", [])
                                if s_steps:
                                    proc_lines.append(f"\n{s_title} ({len(s_steps)} steps):")
                                    for si, st in enumerate(s_steps):
                                        step_tag = ""
                                        if i < current_sec_idx or (i == current_sec_idx and si < current_step_idx):
                                            step_tag = " [COMPLETED]"
                                        elif i == current_sec_idx and si == current_step_idx:
                                            step_tag = " [CURRENT ACTIVE STEP]"
                                        else:
                                            step_tag = " [PENDING]"
                                        proc_lines.append(f"  Step {si+1}: {st.replace(chr(10), ' ').strip()}{step_tag}")
                                else:
                                    proc_lines.append(f"\n{s_title} (steps pending)")
                            proc_context = "\n".join(proc_lines)

                            current_sec_name = sections[current_sec_idx].get("title", f"Section {current_sec_idx+1}") if current_sec_idx < len(sections) else "Unknown"
                            pending_sections = [s.get("title", f"Section {si+1}") for si, s in enumerate(sections) if si > current_sec_idx]
                            pending_sec_str = ", ".join(pending_sections) if pending_sections else "None (this is the final section)"

                            proc_agent_system = (
                                "You are a Procedure Context Agent assisting a technician during an active machine procedure.\n"
                                "You have direct access to the live structured procedure JSON and context memory below.\n\n"
                                f"PROCEDURE: {title}\n"
                                f"SOURCE: {fname_val}\n\n"
                                "SECTIONS AND STEPS:\n"
                                f"{proc_context}\n\n"
                                "CURRENT PROGRESS & MEMORY STATE:\n"
                                f"- Active Section: '{current_sec_name}' (Section {current_sec_idx + 1} of {len(sections)})\n"
                                f"- Current Step in Progress: Step {current_step_idx + 1} of {total_sec_steps}\n"
                                f"- Steps Completed in this section: {current_step_idx} step(s)\n"
                                f"- Steps Remaining in this section: {max(0, total_sec_steps - current_step_idx - 1)} step(s)\n"
                                f"- Pending Future Sections: {pending_sec_str}\n\n"
                                "RESPONSE RULES — follow strictly:\n"
                                "1. If the user asks about progress, status, completed steps, remaining steps, or pending sections "
                                "   (e.g. 'how many steps are completed?', 'what have we done so far?', 'how many steps left?', 'which section are we on?'):\n"
                                "   Answer clearly and concisely directly using the progress and memory state above. Do not output an ACTION token.\n"
                                "2. If the user asks about specific steps, tool requirements, safety warnings, or clarification on procedure instructions:\n"
                                "   Answer clearly and accurately using the structured steps above. Do not output an ACTION token.\n"
                                "3. If the user asks a question about general machine specifications, components, troubleshooting, or topics NOT covered in this procedure:\n"
                                "   Output strictly the token: ACTION:SEARCH\n\n"
                                "CRITICAL: Output ONLY the direct answer OR the exact token 'ACTION:SEARCH'. Never mix ACTION:SEARCH with text."
                            )

                            proc_agent_messages = [
                                SystemMessage(proc_agent_system),
                                HumanMessage(message)
                            ]

                            proc_ans = ""
                            try:
                                if llm:
                                    proc_resp = llm.invoke(proc_agent_messages)
                                    proc_ans = getattr(proc_resp, "content", "").strip()
                            except Exception as e:
                                logger.warning("Active procedure agent LLM failed: %s", e)
                                proc_ans = "ACTION:SEARCH"

                            # If procedure agent answered from memory / structured JSON
                            if proc_ans and not proc_ans.startswith("ACTION:SEARCH"):
                                answer = (
                                    f"{proc_ans}\n\n"
                                    f"*(Still on **Step {current_step_idx + 1} of {total_sec_steps}**)*\n"
                                    "Let me know when you have completed this step or if you need more help."
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Is this step completed?",
                                    "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }

                            # 2. Fallback: Current Manual PDF Scoped Search
                            from backend.retrieval.search_documents import SearchDocumentsTool as _SDT
                            _scoped_answer = ""
                            try:
                                if doc_id:
                                    _scoped_result = _SDT().run(
                                        query=message,
                                        document_scope=[doc_id],
                                        session_id=session_id
                                    )
                                    _scoped_answer = (_scoped_result.get("answer") or "").strip()
                            except Exception as e:
                                logger.warning(f"Stuck Technician SDT failed: {e}")
                                _scoped_answer = ""

                            _refusal_phrases = (
                                "could not find", "cannot find", "no information", "not found",
                                "does not contain", "not available", "no mention", "unable to find",
                                "nothing relevant", "not provided", "i don't have"
                            )
                            _is_refusal = (
                                not _scoped_answer
                                or len(_scoped_answer) < 40
                                or any(ph in _scoped_answer.lower() for ph in _refusal_phrases)
                            )

                            if _is_refusal:
                                state["stage"] = "trouble_global_search_pending"
                                state["trouble_query"] = message
                                store.set_interactive_state(session_id, state)
                                _manual_title = _clean_title(fname_val) if fname_val else "your current manual"
                                answer = (
                                    f"I could not find information about this in **{_manual_title}**. "
                                    "Would you like me to search all manuals globally?"
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Can I search globally?",
                                    "options": ["✅ Yes, search globally", "❌ No, that's fine"],
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }
                            else:
                                trouble_ans = _scoped_answer
                                answer = (
                                    f"{trouble_ans}\n\n"
                                    f"*(Still on **Step {current_step_idx + 1} of {total_sec_steps}**)*\n"
                                    "Let me know when you have completed this step or if you need more help."
                                )
                                return {
                                    "status": "needs_clarification",
                                    "answer": answer,
                                    "question": "Is this step completed?",
                                    "options": ["✅ Step Complete - Next", "📋 View Full Section Summary", "Stop checklist"],
                                    "tool_calls": [],
                                    "execution_trace": list(_trace_sink),
                                    "messages": [HumanMessage(message), AIMessage(content=answer)],
                                    "token_usage": _get_tu(answer),
                                    "trace_id": None,
                                }

            else:
                if mode:
                    logger.info("⚡ [GUIDED STATE] New query detected. Clearing active state.")
                    store.set_interactive_state(session_id, None)
        except Exception as state_exc:
            logger.warning("Failed to process guided assistant state: %s", state_exc)

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
    is_question = not _is_greeting(message)

    # === CHAT CONTEXT AGENT: Session-Scoped Pre-Search Interceptor =============
    # When the session has context docs (PDFs opened / procedures run in this chat)
    # and the 5-tier scope resolver did NOT already pin a specific document,
    # try answering from only those docs first.  If found → return immediately.
    # If not → ask the user (with buttons) whether to expand to global search.
    if context_docs and not resolved_scope and is_question and session_id:
        try:
            from backend.retrieval.search_documents import SearchDocumentsTool as _SDT
            _scoped_result = _SDT().run(
                query=message,
                document_scope=context_docs,
                session_id=session_id,
            )
            _scoped_answer = (_scoped_result.get("answer") or "").strip()

            # Refusal detection: LLM answerer emits one of these when it can't find the answer
            _refusal_phrases = (
                "could not find", "cannot find", "no information", "not found",
                "does not contain", "not available", "no mention", "unable to find",
                "not provided", "i don't have",
            )
            _is_refusal = (
                not _scoped_answer
                or len(_scoped_answer) < 40
                or any(ph in _scoped_answer.lower() for ph in _refusal_phrases)
            )

            if not _is_refusal:
                # Good answer found in context docs — return it directly (fast path)
                logger.info("⚡ [CONTEXT AGENT] Answered from session context (%d docs)", len(context_docs))
                _ctx_sources = _scoped_result.get("sources") or []
                _ctx_tc = [
                    {"name": "search_documents",
                     "args": {"query": message, "document_scope": context_docs},
                     "result": ""}
                ]
                return {
                    "status": "done",
                    "answer": _scoped_answer,
                    "tool_calls": _ctx_tc,
                    "execution_trace": [],
                    "messages": [HumanMessage(message), AIMessage(content=_scoped_answer)],
                    "token_usage": _scoped_result.get("token_usage") or {"total_tokens": 100, "input_tokens": 80, "output_tokens": 20},
                    "trace_id": None,
                }
            else:
                # No answer in context — ask user for permission to search globally
                logger.info("⚡ [CONTEXT AGENT] No answer in context (%d docs), asking user for global search", len(context_docs))
                _ctx_fnames = _get_filenames_for_ids(context_docs, config)
                _clean_names = [_clean_title(f) for f in _ctx_fnames if f]
                _ctx_label = ", ".join(_clean_names[:3]) or "your current manual"
                if len(_clean_names) > 3:
                    _ctx_label += f" and {len(_clean_names) - 3} more"

                _cq = (
                    f"I couldn't find information about this in **{_ctx_label}**. "
                    "Would you like me to search all manuals globally?"
                )

                # Persist state so we can re-run the original question on user approval
                try:
                    from backend.storage.conversation_store import get_conversation_store as _gcs
                    _gcs().set_interactive_state(session_id, {
                        "mode": "context_search",
                        "stage": "global_search_pending",
                        "original_message": message,
                    })
                except Exception:
                    pass

                return {
                    "status": "needs_clarification",
                    "answer": _cq,
                    "question": _cq,
                    "options": ["\u2705 Yes, search globally", "\u274c No, that's fine"],
                    "tool_calls": [],
                    "execution_trace": [],
                    "messages": [HumanMessage(message), AIMessage(content=_cq)],
                    "token_usage": {"total_tokens": 50, "input_tokens": 30, "output_tokens": 20},
                    "trace_id": None,
                }
        except Exception as _ctx_exc:
            logger.warning("[CONTEXT AGENT] Error in scoped pre-search, falling through: %s", _ctx_exc)
    # === END CHAT CONTEXT AGENT ================================================


    system_prompt_text = SYSTEM_PROMPT
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
        if not answer.strip() and unique_answers:
            # Rescue the most recent unique search answer if LLM produced no text
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

    # ── Evaluate if search results contain an actionable procedure ──
    tool_snippets = []
    for m in final_state.get("messages", []):
        if isinstance(m, ToolMessage) and m.content:
            try:
                td = json.loads(m.content)
                if isinstance(td, dict):
                    if td.get("answer"):
                        tool_snippets.append(td["answer"])
                    for src in (td.get("sources") or td.get("passages") or []):
                        if isinstance(src, dict) and src.get("text"):
                            tool_snippets.append(src["text"])
            except Exception:
                tool_snippets.append(m.content[:500])

    combined_context = (message or "") + "\n" + (final_answer or "") + "\n" + "\n".join(tool_snippets)

    # Fast structural marker pre-check + Dynamic Micro-LLM Intent Classification
    has_step_markers = _has_procedural_step_markers(combined_context)
    is_procedural = False
    if has_step_markers and llm:
        is_procedural = _classify_procedure_intent_llm(message, combined_context, llm)

    # Check if search results contained multiple distinct files for a procedural query
    if session_id and is_procedural and not final_state.get("guard_blocked"):
        try:
            # Find search results in tool messages
            unique_files = [] # list of (doc_id, filename)
            citations = _extract_all_citations(final_state["messages"])
            for c in citations:
                doc_id = str(c.get("document_id"))
                fname = c.get("filename")
                if doc_id and fname and (doc_id, fname) not in unique_files:
                    unique_files.append((doc_id, fname))
            
            if len(unique_files) >= 2:
                from backend.storage.conversation_store import get_conversation_store
                store = get_conversation_store()
                
                state = {
                    "mode": "guided_assistant",
                    "stage": "disambiguation",
                    "disambiguation_options": [
                        {"index": idx + 1, "document_id": doc_id, "filename": fname, "title": _clean_title(fname)}
                        for idx, (doc_id, fname) in enumerate(unique_files)
                    ]
                }
                store.set_interactive_state(session_id, state)
                
                menu_lines = [f"I found cleaning/procedural instructions across {len(unique_files)} different manuals:"]
                for opt in state["disambiguation_options"]:
                    menu_lines.append(f"{opt['index']}. {opt['title']} (Document: {opt['filename']})")
                menu_lines.append("\nWhich area/manual would you like to proceed with?")
                
                options = [f"{opt['index']}. {opt['title']}" for opt in state["disambiguation_options"]]
                
                offer_answer = "\n".join(menu_lines)
                return {
                    "status": "needs_clarification",
                    "answer": offer_answer,
                    "question": "Which area/manual would you like to proceed with?",
                    "options": options,
                    "tool_calls": tool_calls,
                    "llm_calls": calls_log,
                    "execution_trace": execution_trace,
                    "messages": final_state["messages"],
                    "token_usage": token_usage,
                    "trace_id": trace_info["trace_id"],
                    "guard_risk_score": final_state.get("guard_risk_score", 0),
                    "guard_policy": final_state.get("guard_policy", "allow"),
                }
        except Exception as e:
            logger.warning("Failed to construct disambiguation menu: %s", e)

    # ── Single-file procedural offering — Phase 1: lightweight first offer (0 DB reads, 0 LLM calls) ──
    if session_id and final_answer and not final_state.get("guard_blocked") and final_state.get("status") != "needs_clarification" and is_procedural:
        try:
            from backend.storage.conversation_store import get_conversation_store
            store = get_conversation_store()

            doc_id, page_no = _get_primary_citation(final_state["messages"])
            logger.info("[GUIDED PHASE1] doc_id=%s page=%s", doc_id, page_no)

            if doc_id:
                fname = "Document"
                citations = _extract_all_citations(final_state["messages"])
                if citations:
                    fname = citations[0].get("filename") or fname

                title = _clean_title(fname)
                # Extract human-readable manual code from filename (e.g. "G0793V10")
                manual_code = re.sub(r'[_\-\.](pdf|docx?|xlsx?|pptx?)$', '', fname, flags=re.IGNORECASE)
                manual_code = re.sub(r'[\._]', ' ', manual_code).strip()

                # Derive procedure start page from citations returned by the search tool.
                # _extract_all_citations already has correct doc_id strings and page numbers.
                citation_pages = [
                    c.get("page")
                    for c in citations
                    if str(c.get("document_id", "")) == str(doc_id)
                    and c.get("page") is not None
                ]
                retrieval_min_page = min((int(p) for p in citation_pages if str(p).isdigit()), default=1)
                logger.info("[GUIDED PHASE1] retrieval_min_page=%d (from %d citation pages for doc_id=%s)", retrieval_min_page, len(citation_pages), doc_id)

                state = {
                    "mode": "guided_assistant",
                    "stage": "procedure_offer",
                    "document_id": doc_id,
                    "filename": fname,
                    "title": title,
                    "query": message,
                    "start_page": page_no or 1,
                    "retrieval_min_page": retrieval_min_page,
                    "selected_option": {"document_id": doc_id, "filename": fname, "title": title}
                }
                store.set_interactive_state(session_id, state)

                offer_answer = (
                    f"I found a procedure about **{title}** from **{manual_code}**.\n\n"
                    "Would you like to start the guided procedure?"
                )
                import json as _json_offer
                offer_tc = [{
                    "name": "get_page_context",
                    "args": {"document_id": doc_id, "page": 1},
                    "result": _json_offer.dumps({"document_id": doc_id, "filename": fname, "page": 1})
                }]
                return {
                    "status": "needs_clarification",
                    "answer": offer_answer,
                    "question": "Would you like to start the guided procedure?",
                    "options": ["Continue", "No, thanks"],
                    "tool_calls": offer_tc,
                    "llm_calls": calls_log,
                    "execution_trace": execution_trace,
                    "messages": [HumanMessage(message), AIMessage(content=offer_answer)],
                    "token_usage": token_usage,
                    "trace_id": trace_info["trace_id"],
                    "guard_risk_score": final_state.get("guard_risk_score", 0),
                    "guard_policy": final_state.get("guard_policy", "allow"),
                }
        except Exception as e:
            logger.warning("Failed to offer interactive checklist: %s", e)


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
    }

