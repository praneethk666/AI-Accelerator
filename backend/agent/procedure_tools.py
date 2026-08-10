"""Agent-callable tools: start and drive a guided, one-step-at-a-time procedure
walkthrough (backend/agent/step_parser.py's parsed step graph), state persisted
in Postgres (backend/storage/conversation_store.py::get/set_session_active_procedure)
so it survives across the executor's restart-based turns.

Neither tool goes through write_tools (not a corpus mutation) or clarify_tools
(must not pause for human approval every single step -- that would break the
one-step-at-a-time flow entirely). Normal 2-turn tool-call flow, same as any
other agent tool.

The agent sets AdvanceProcedureStepTool's `action` argument directly from its own
reading of the user's reply ("done" -> next, "what does that mean" -> repeat,
"go back" -> back, a branch condition met -> goto) -- deliberately no separate
classifier call for this, matching the project's explicit "very agentic" design
choice: the tool-calling model makes this judgment itself, same trust level as
any other tool argument it sets.
"""
from __future__ import annotations

import os
from typing import Any


def _load_default_config() -> dict:
    from backend.core.config import load_config
    return load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))


def _walkthrough_enabled(config: dict) -> bool:
    cfg = (config.get("query") or {}).get("agent", {}).get("procedure_walkthrough") or {}
    return bool(cfg.get("enabled", False))


def _max_steps(config: dict) -> int:
    cfg = (config.get("query") or {}).get("agent", {}).get("procedure_walkthrough") or {}
    return int(cfg.get("max_steps", 50) or 50)


def _find_previous_step(steps: dict, current: str) -> str | None:
    """Steps only carry a forward `next` pointer -- "back" is computed by
    scanning for whichever step's `next` equals the current one, rather than
    keeping a separate history stack in the state blob."""
    for step_id, step in steps.items():
        if step.get("next") == current:
            return step_id
    return None


def _step_response(procedure: dict, step_id: str) -> dict[str, Any]:
    step = procedure["steps"][step_id]
    resp: dict[str, Any] = {
        "document_id": procedure["document_id"],
        "section_title": procedure["section_title"],
        "step_id": step_id,
        "step_text": step["text"],
        "page": step.get("page"),
        "has_next": step.get("next") is not None,
        "status": procedure["status"],
    }
    if step.get("branches"):
        resp["branches"] = step["branches"]
    return resp


class StartProcedureWalkthroughTool:
    """Locate a section (by title hint and/or page) and parse it into an
    ordered step graph, then start a guided walkthrough of it. Conforms to the
    AgentTool protocol in backend/agent_tools.py.
    """

    name = "start_procedure_walkthrough"
    description = (
        "Start a guided, ONE-STEP-AT-A-TIME walkthrough of a numbered procedure "
        "in a manual (e.g. a maintenance/changeover task) -- presents step 1, "
        "then waits for the user's reply before advancing. Use this instead of "
        "dumping the whole procedure as one answer when the user wants to "
        "actually DO the steps, not just read them. Locate the right section "
        "first (browse_document_outline or search_documents), then pass its "
        "document_id and either section_title (a substring of the exact section "
        "heading, e.g. 'Replacing the Workpiece Holder') or the page it starts "
        "on. Not every section is a numbered step list -- if this returns an "
        "error, present the section as normal prose instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document containing the procedure."},
            "section_title": {
                "type": "string",
                "description": "Substring of the exact section heading, e.g. 'Replacing the Workpiece Holder'.",
            },
            "page": {"type": "integer", "description": "The page the section starts on, if known."},
        },
        "required": ["document_id"],
    }

    def run(self, document_id: str = "", section_title: str | None = None,
            page: int | None = None, session_id: str | None = None) -> dict[str, Any]:
        if not document_id:
            return {"error": "document_id is required"}
        if not section_title and page is None:
            return {"error": "pass section_title and/or page to locate the section"}
        if not session_id:
            return {"error": "no active session -- cannot start a stateful walkthrough"}

        config = _load_default_config()
        if not _walkthrough_enabled(config):
            return {"error": "guided procedure walkthroughs are disabled for this "
                             "deployment (query.agent.procedure_walkthrough.enabled: "
                             "false). Present the procedure as normal prose instead."}

        from backend.storage.postgres_store import PostgresStore
        pg = PostgresStore()
        try:
            blocks = pg.get_blocks(document_id)
        finally:
            pg.close()
        if not blocks:
            return {"error": f"no extracted content found for document {document_id!r}"}

        from backend.agent.step_parser import parse_procedure_from_blocks
        parsed = parse_procedure_from_blocks(
            blocks, section_hint=section_title, start_page=page, max_steps=_max_steps(config))
        if parsed is None:
            return {"error": "could not locate a numbered step procedure there -- "
                             "present the section as normal prose instead "
                             "(via search_documents or get_page_context)."}

        import datetime
        procedure = {
            "document_id": document_id,
            "section_title": parsed["section_title"],
            "page_range": parsed["page_range"],
            "steps": parsed["steps"],
            "current_step": parsed["first_step"],
            "status": "in_progress",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        from backend.storage.conversation_store import PostgresConversationStore
        PostgresConversationStore().set_session_active_procedure(session_id, procedure)

        return _step_response(procedure, procedure["current_step"])

    __call__ = run


class AdvanceProcedureStepTool:
    """Advance, repeat, back up, jump, or abort the session's active guided
    procedure walkthrough. Conforms to the AgentTool protocol in
    backend/agent_tools.py.
    """

    name = "advance_procedure_step"
    description = (
        "Advance the user's active guided procedure walkthrough (started via "
        "start_procedure_walkthrough). Choose `action` yourself from the user's "
        "reply: 'next' when they confirm the current step is done ('done', "
        "'ok', 'next'); 'repeat' when they need it re-explained ('what does "
        "that mean', 'I don't understand', 'say that again') -- this does NOT "
        "advance; 'back' to return to the previous step; 'goto' with a specific "
        "step_id when the current step's branches indicate that's the right "
        "next step for their situation; 'abort' when they want to stop the "
        "walkthrough entirely. Fails with a clear error if no walkthrough is "
        "active for this session."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["next", "repeat", "back", "goto", "abort"],
                "description": "What to do, chosen from the user's actual reply.",
            },
            "step_id": {
                "type": "string",
                "description": "Required for action='goto' -- the target step id "
                "(e.g. from the current step's `branches`).",
            },
        },
        "required": ["action"],
    }

    def run(self, action: str = "", step_id: str | None = None,
            session_id: str | None = None) -> dict[str, Any]:
        if not session_id:
            return {"error": "no active session"}
        if action not in ("next", "repeat", "back", "goto", "abort"):
            return {"error": f"unknown action {action!r} -- must be one of "
                             "next/repeat/back/goto/abort"}
        if not _walkthrough_enabled(_load_default_config()):
            return {"error": "guided procedure walkthroughs are disabled for this deployment "
                             "(query.agent.procedure_walkthrough.enabled: false)."}

        from backend.storage.conversation_store import PostgresConversationStore
        store = PostgresConversationStore()
        procedure = store.get_session_active_procedure(session_id)
        if not procedure or procedure.get("status") != "in_progress":
            return {"error": "no active procedure walkthrough for this session -- "
                             "call start_procedure_walkthrough first."}

        if action == "abort":
            store.set_session_active_procedure(session_id, None)
            return {"status": "aborted", "section_title": procedure["section_title"]}

        current = procedure["current_step"]
        steps = procedure["steps"]

        if action == "repeat":
            return _step_response(procedure, current)

        if action == "back":
            prev = _find_previous_step(steps, current)
            if prev is None:
                return {"error": "already at the first step -- nothing to go back to"}
            procedure["current_step"] = prev
            store.set_session_active_procedure(session_id, procedure)
            return _step_response(procedure, prev)

        if action == "goto":
            if not step_id:
                return {"error": "step_id is required for action='goto'"}
            if step_id not in steps:
                return {"error": f"no such step {step_id!r} in this procedure"}
            procedure["current_step"] = step_id
            store.set_session_active_procedure(session_id, procedure)
            return _step_response(procedure, step_id)

        # action == "next"
        nxt = steps[current].get("next")
        if nxt is None:
            procedure["status"] = "completed"
            store.set_session_active_procedure(session_id, None)
            return {
                "status": "completed",
                "section_title": procedure["section_title"],
                "note": "That was the last step -- the procedure is complete.",
            }
        procedure["current_step"] = nxt
        store.set_session_active_procedure(session_id, procedure)
        return _step_response(procedure, nxt)

    __call__ = run
