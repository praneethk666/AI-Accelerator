"""Headless tests for the agent-executor loop — mocked LLM, fake tools, no network,
no LLM keys, no DB. Covers: picking + running a read tool, blocking a write tool
until approved then re-running it, and the iteration cap terminating a runaway loop.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.agent.executor import run_agent


class _ScriptedLLM:
    """Fake chat model: bind_tools() is a no-op returning self; invoke() pops the
    next canned response off a script, in order, and records what it was called with."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.invocations: list[list] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return self._responses.pop(0)


class _FakeSearchTool:
    name = "search_documents"
    description = "Search the ingested documents and answer a question."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": "42", "citations": []}


class _FakeIngestTool:
    name = "ingest_document"
    description = "Ingest a document file through the full pipeline."
    input_schema = {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"document_id": "doc-1", "status": "ready"}


_CONFIG = {"query": {"agent": {"max_iterations": 5, "write_tools": ["ingest_document"]}}}


def _tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_agent_picks_and_runs_a_read_tool_then_answers():
    # Real feature found live during the 4-Aug merge (backend/agent/executor.py's
    # "search short-circuit"): when search_documents is the ONLY tool called this
    # turn and it returns a complete non-refusal answer, that answer is used
    # DIRECTLY -- Turn 2 (a second LLM call to just repeat/rephrase it) is
    # skipped entirely (saves ~6-7s/~4000 tokens per standard question, since
    # search_documents already does its own retrieval+answer synthesis
    # internally). The second scripted LLM response below is deliberately never
    # consumed -- it exists only to prove the loop does NOT reach it.
    search = _FakeSearchTool()
    llm = _ScriptedLLM([
        _tool_call_message("search_documents", {"query": "what is the warranty period?"}),
        AIMessage(content="The warranty period is 42 months."),
    ])

    result = run_agent(
        "what is the warranty period?",
        config=_CONFIG,
        registry={"search_documents": search},
        llm=llm,
    )

    assert result["status"] == "done"
    assert result["answer"] == "42"  # short-circuited straight from the tool's own answer
    assert len(llm.invocations) == 1  # Turn 2 was skipped
    assert search.calls == [{"query": "what is the warranty period?"}]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "search_documents"
    assert "42" in result["tool_calls"][0]["result"]


def test_search_short_circuit_does_not_fire_on_a_refusal_answer():
    # The short-circuit explicitly excludes refusal answers (see executor.py's
    # _REFUSAL_HINTS) -- a "could not find this" answer must still go through a
    # real Turn 2 LLM call, not be handed back verbatim as if it were final.
    search = _FakeSearchTool()
    search.run = lambda **kw: {"answer": "I could not find this in the provided documents.", "citations": []}
    llm = _ScriptedLLM([
        _tool_call_message("search_documents", {"query": "what is the warranty period?"}),
        AIMessage(content="I don't have that information in the indexed documents."),
    ])

    result = run_agent(
        "what is the warranty period?",
        config=_CONFIG,
        registry={"search_documents": search},
        llm=llm,
    )

    assert len(llm.invocations) == 2  # Turn 2 DID run -- no short-circuit
    assert result["answer"] == "I don't have that information in the indexed documents."


def test_write_tool_needs_approval_then_runs_once_approved(tmp_path):
    ingest = _FakeIngestTool()
    report_file = tmp_path / "report.pdf"
    report_file.write_text("dummy")
    report_path = str(report_file).replace("\\", "/")

    # First ask: the model wants to ingest — must be blocked, not executed.
    llm_first_ask = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": report_path}),
    ])
    result = run_agent(
        f"please ingest {report_path}",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm_first_ask,
    )

    assert result["status"] == "needs_approval"
    assert result["answer"] is None
    assert result["pending"] == [{"id": "call_1", "name": "ingest_document", "args": {"file_path": report_path}}]
    assert ingest.calls == []  # blocked — must not have run

    # User approves; caller re-invokes with approved_writes=True AND echoes the
    # approved call back (approval is bound to the exact name+args shown).
    llm_approved = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": report_path}),
        AIMessage(content="Ingested — document_id doc-1, status ready."),
    ])
    result = run_agent(
        f"please ingest {report_path}",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm_approved,
        approved_writes=True,
        approved_calls=[{"name": "ingest_document", "args": {"file_path": report_path}}],
    )

    assert result["status"] == "done"
    assert ingest.calls == [{"file_path": report_path}]
    assert "doc-1" in result["answer"]


def test_approval_is_bound_to_approved_args(tmp_path):
    """Approving one file must NOT authorize a different one the model re-proposes."""
    ingest = _FakeIngestTool()
    report_file = tmp_path / "report.pdf"
    report_file.write_text("dummy")
    report_path = str(report_file).replace("\\", "/")

    evil_file = tmp_path / "EVIL.pdf"
    evil_file.write_text("evil")
    evil_path = str(evil_file).replace("\\", "/")

    llm = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": evil_path}),
    ])
    result = run_agent(
        f"please ingest {report_path}",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm,
        approved_writes=True,
        approved_calls=[{"name": "ingest_document", "args": {"file_path": report_path}}],
    )
    assert result["status"] == "needs_approval"   # args mismatch => re-blocked
    assert ingest.calls == []                     # the un-approved file did NOT run


def test_request_clarification_pauses_for_user_choice():
    """The agent calling request_clarification pauses the loop with a machine-readable
    question + options instead of guessing."""
    llm = _ScriptedLLM([
        _tool_call_message("request_clarification",
                           {"question": "Which document?", "options": ["a.pdf", "b.pdf"]}),
    ])
    result = run_agent(
        "what is the torque spec?",
        config=_CONFIG,
        registry={},          # clarify is intercepted before dispatch; no tool needed
        llm=llm,
    )
    assert result["status"] == "needs_clarification"
    assert result["question"] == "Which document?"
    assert result["options"] == ["a.pdf", "b.pdf"]


class _FakeGenericTool:
    """A tool name with NO special-casing anywhere in executor.py's tools_node
    (unlike "ingest_document", which has its own hardcoded DB/filesystem
    pre-flight checks that bypass the registered tool's .run() entirely when
    the path doesn't exist -- confirmed live while writing this test, a bad
    choice for exercising generic tool-dispatch/cap behavior) and NOT
    "search_documents" (which triggers the short-circuit below)."""
    name = "generic_test_tool"
    description = "A generic tool with no special-casing in tools_node."
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"result": "ok"}  # deliberately no "answer" key -- not search_documents-shaped


def test_iteration_cap_terminates_a_runaway_tool_calling_loop():
    # Real interaction found live during the 4-Aug merge: executor.py's "search
    # short-circuit" (see test_agent_picks_and_runs_a_read_tool_then_answers)
    # fires whenever a turn's ONLY tool call is search_documents with a valid
    # answer -- which, with the OLD version of this test (search_documents
    # scripted 10x, always answering "42"), fired on iteration 1 itself,
    # never actually exercising the iteration cap at all. Using a genuinely
    # generic tool here so the cap-enforcement path this test exists to
    # protect is the one actually running.
    other = _FakeGenericTool()
    llm = _ScriptedLLM([_tool_call_message("generic_test_tool", {"x": "y"}) for _ in range(10)])

    result = run_agent(
        "loop forever",
        config={"query": {"agent": {"max_iterations": 3, "write_tools": []}}},
        registry={"generic_test_tool": other},
        llm=llm,
    )

    # Traced live: 3 real agent-loop turns run (respecting max_iterations=3),
    # but only 2 tool calls actually dispatch -- the 3rd turn's tool_calls
    # never reach tools_node because route_after_agent, once iterations hits
    # the cap, routes straight to output_guard for any non-write/clarify tool.
    # A 4th LLM call then happens OUTSIDE the graph entirely: run_agent's own
    # post-graph fallback, since no clean single-answer recovery was available
    # (generic_test_tool's result has no "answer" key, unlike search_documents'
    # fast-path -- see test_iteration_cap_recovers_answer_from_search_tool_history).
    assert len(llm.invocations) == 4
    assert len(other.calls) == 2
    assert result["status"] == "done"  # cap hit outside the tools node -> no pending_approval


def test_iteration_cap_recovers_answer_from_search_tool_history():
    # The cap-hit fast path (run_agent's post-graph fallback, ~line 1905): if
    # the loop ends with no final AIMessage, scan tool history for a single,
    # unique search_documents answer and use it directly rather than pay for
    # an extra LLM synthesis call. To reach the cap at all (3 real agent
    # turns) without the EARLIER search short-circuit intercepting turn 1
    # first, each turn calls search_documents TWICE (breaks the
    # short-circuit's `len(all_calls) == 1` gate) while keeping every tool
    # name literally "search_documents" (the fast path's `not
    # has_other_tools` check requires this) — both calls return the same
    # answer, so exactly one unique answer survives to the cap-hit fallback.
    def _two_search_calls(call_id: str) -> AIMessage:
        return AIMessage(content="", tool_calls=[
            {"name": "search_documents", "args": {"query": "x"}, "id": call_id + "_a", "type": "tool_call"},
            {"name": "search_documents", "args": {"query": "y"}, "id": call_id + "_b", "type": "tool_call"},
        ])

    search = _FakeSearchTool()
    llm = _ScriptedLLM([_two_search_calls(f"call_{i}") for i in range(10)])
    result = run_agent(
        "loop forever",
        config={"query": {"agent": {"max_iterations": 3, "write_tools": []}}},
        registry={"search_documents": search},
        llm=llm,
    )
    assert len(llm.invocations) == 3
    assert result["answer"] == "42"
    assert result["status"] == "done"


def test_prune_messages_for_llm_reconstructs_search_output():
    from langchain_core.messages import ToolMessage, HumanMessage
    from backend.agent.executor import _prune_messages_for_llm
    import json

    full_payload = {
        "answer": "Factor 3 is Motor failure [1, p.2].",
        "citations": [
            {"filename": "motor_manual.pdf", "page": 2, "document_id": "doc-123"}
        ],
        "sources": [{"filename": "motor_manual.pdf"}]
    }
    
    messages = [
        HumanMessage("What is Factor 3?"),
        ToolMessage(
            content=json.dumps(full_payload),
            tool_call_id="call_123",
            name="search_documents"
        )
    ]
    
    pruned = _prune_messages_for_llm(messages)
    
    assert len(pruned) == 2
    assert isinstance(pruned[0], HumanMessage)
    assert pruned[0].content == "What is Factor 3?"
    
    assert isinstance(pruned[1], ToolMessage)
    assert "Search Answer: Factor 3 is Motor failure [1, p.2]." in pruned[1].content
    assert "Source Map:" in pruned[1].content
    assert "[1] = motor_manual.pdf (page 2) [id: doc-123]" in pruned[1].content
    # Crucially, snippets and sources should be pruned
    assert "citations" not in pruned[1].content
    assert "sources" not in pruned[1].content


def test_prune_messages_for_llm_ignores_non_json_messages():
    from langchain_core.messages import ToolMessage
    from backend.agent.executor import _prune_messages_for_llm

    messages = [
        ToolMessage(content="some raw error string", tool_call_id="call_456", name="sql_read")
    ]
    pruned = _prune_messages_for_llm(messages)
    assert len(pruned) == 1
    assert pruned[0].content == "some raw error string"


def test_ingest_document_missing_file_does_not_ask_approval():
    """If ingest_document is called on a file that does not exist, the agent loops back
    with an error immediately instead of blocking to ask for human approval."""
    ingest = _FakeIngestTool()
    llm = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": "missing_file_path_123.pdf"}),
        AIMessage(content="I cannot find that file. Please upload it first."),
    ])
    result = run_agent(
        "please ingest missing_file_path_123.pdf",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm,
    )
    # Since the file did not exist, the tools node did not return needs_approval status.
    # It returned the error to the LLM, which then generated the final text answer.
    assert result["status"] == "done"
    assert "cannot find" in result["answer"]
    assert ingest.calls == []


def test_ingest_document_already_ready_bypasses_approval(tmp_path):
    """If ingest_document is called on a file that is already ingested and ready,
    the tools node returns the ready status immediately without asking approval."""
    from unittest.mock import MagicMock, patch
    ingest = _FakeIngestTool()
    
    # Create the physical file so that os.path.isfile returns True
    report_file = tmp_path / "report.pdf"
    report_file.write_text("dummy")
    report_path = str(report_file).replace("\\", "/")

    llm = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": report_path}),
        AIMessage(content="The document is already ready. I will search it now."),
    ])

    # Mock PostgresStore so it returns that the document is ready
    mock_pg = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ("e27afb11-788e-44fa-8bb8-a7a9668a1b97", report_path, "ready")
    mock_pg.conn.cursor.return_value = mock_cur

    with patch("backend.storage.postgres_store.PostgresStore", return_value=mock_pg):
        result = run_agent(
            f"please ingest {report_path}",
            config=_CONFIG,
            registry={"ingest_document": ingest},
            llm=llm,
        )

    # Since the file is already ready, it should be marked as "done" without blocking for approval
    assert result["status"] == "done"
    assert "already ready" in result["answer"]
    assert ingest.calls == []  # Not actually called since it was already ingested and ready


# ── Guided procedure walkthrough wiring (ADDED 10-Aug) ────────────────────────
# session_id injection into the sync tools_node (the one run_agent() actually
# uses) for the new procedure tools, and _resolve_turn_procedure_state's
# system-prompt injection when a walkthrough is already active for the session.

class _FakeAdvanceProcedureTool:
    name = "advance_procedure_step"
    description = "Advance the active guided procedure walkthrough."
    input_schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"step_id": "2", "step_text": "Press the MASTER ON button.", "has_next": True}


def test_advance_procedure_step_gets_session_id_injected_by_sync_tools_node():
    # Real gap found while planning this feature: the sync tools_node (the one
    # run_agent() actually dispatches through) never injected session_id for ANY
    # tool -- only the unused async builder did, and only for search_documents.
    # The new procedure tools need session_id to read/write Postgres state, so
    # this must actually reach tool.run() as a kwarg, not just be in scope.
    tool = _FakeAdvanceProcedureTool()
    llm = _ScriptedLLM([
        _tool_call_message("advance_procedure_step", {"action": "next"}),
        AIMessage(content="Now press the MASTER ON button."),
    ])

    result = run_agent(
        "done",
        config=_CONFIG,
        registry={"advance_procedure_step": tool},
        llm=llm,
        session_id="session-abc",
    )

    assert result["status"] == "done"
    assert tool.calls == [{"action": "next", "session_id": "session-abc"}]


def test_advance_procedure_step_no_session_id_when_none_given():
    # Without a real session_id, the injection must not fire at all (matches
    # the existing `if name == ... and session_id:` guard already used for
    # search_documents) -- a tool relying on session state should see it
    # genuinely absent, not an empty string standing in for "no session".
    tool = _FakeAdvanceProcedureTool()
    llm = _ScriptedLLM([
        _tool_call_message("advance_procedure_step", {"action": "next"}),
        AIMessage(content="ok"),
    ])
    run_agent("done", config=_CONFIG, registry={"advance_procedure_step": tool}, llm=llm)
    assert "session_id" not in tool.calls[0]


def test_resolve_turn_procedure_state_none_when_disabled():
    from backend.agent.executor import _resolve_turn_procedure_state
    result = _resolve_turn_procedure_state("session-1", {"query": {"agent": {}}})
    assert result is None


def test_resolve_turn_procedure_state_none_when_no_session_id():
    from backend.agent.executor import _resolve_turn_procedure_state
    cfg = {"query": {"agent": {"procedure_walkthrough": {"enabled": True}}}}
    assert _resolve_turn_procedure_state("", cfg) is None


def test_resolve_turn_procedure_state_none_when_no_active_procedure():
    from unittest.mock import MagicMock, patch
    from backend.agent.executor import _resolve_turn_procedure_state
    cfg = {"query": {"agent": {"procedure_walkthrough": {"enabled": True}}}}
    store = MagicMock()
    store.get_session_active_procedure.return_value = None
    with patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        assert _resolve_turn_procedure_state("session-1", cfg) is None


def test_resolve_turn_procedure_state_returns_prompt_addition_when_active():
    from unittest.mock import MagicMock, patch
    from backend.agent.executor import _resolve_turn_procedure_state
    cfg = {"query": {"agent": {"procedure_walkthrough": {"enabled": True}}}}
    store = MagicMock()
    store.get_session_active_procedure.return_value = {
        "section_title": "1.1 Replacing the Workpiece Holder",
        "current_step": "2", "status": "in_progress",
        "steps": {"2": {"text": "Press the MASTER ON button.", "page": 5, "next": "3"}},
    }
    with patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = _resolve_turn_procedure_state("session-1", cfg)
    assert result is not None
    assert "ACTIVE GUIDED PROCEDURE" in result
    assert "Replacing the Workpiece Holder" in result
    assert "Press the MASTER ON button." in result
    assert "advance_procedure_step" in result


def test_resolve_turn_procedure_state_fails_open_on_db_error():
    from unittest.mock import patch
    from backend.agent.executor import _resolve_turn_procedure_state
    cfg = {"query": {"agent": {"procedure_walkthrough": {"enabled": True}}}}
    with patch("backend.storage.conversation_store.PostgresConversationStore",
              side_effect=RuntimeError("db down")):
        result = _resolve_turn_procedure_state("session-1", cfg)
    assert result is None  # no active procedure for THIS turn, not a crash


def test_run_agent_injects_active_procedure_note_into_system_prompt():
    from unittest.mock import MagicMock, patch
    store = MagicMock()
    store.get_session_active_procedure.return_value = {
        "section_title": "1.1 Replacing the Workpiece Holder",
        "current_step": "2", "status": "in_progress",
        "steps": {"2": {"text": "Press the MASTER ON button.", "page": 5, "next": "3"}},
    }
    cfg = {"query": {"agent": {"max_iterations": 5, "write_tools": [],
                                "procedure_walkthrough": {"enabled": True}}}}
    llm = _ScriptedLLM([AIMessage(content="ok, moving on")])

    with patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        run_agent("done", config=cfg, registry={}, llm=llm, session_id="session-1")

    system_msg = llm.invocations[0][0]
    assert "ACTIVE GUIDED PROCEDURE" in system_msg.content



