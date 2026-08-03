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
    assert result["answer"] == "The warranty period is 42 months."
    assert search.calls == [{"query": "what is the warranty period?"}]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "search_documents"
    assert "42" in result["tool_calls"][0]["result"]


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


def test_iteration_cap_terminates_a_runaway_tool_calling_loop():
    search = _FakeSearchTool()
    # The model never stops asking for the same tool — must not hang forever.
    llm = _ScriptedLLM([_tool_call_message("search_documents", {"query": "x"}) for _ in range(10)])

    result = run_agent(
        "loop forever",
        config={"query": {"agent": {"max_iterations": 3, "write_tools": []}}},
        registry={"search_documents": search},
        llm=llm,
    )

    # 3 loop iterations (the cap), no extra LLM call: executor.py's fallback now tries
    # a fast path first — recovering the answer directly from the last tool result's
    # own "answer" field (present here: _FakeSearchTool returns {"answer": "42", ...})
    # — only falling to a slow-path LLM synthesis call if that's unavailable.
    assert len(llm.invocations) == 3
    assert result["answer"] == "42"  # recovered via the fast path, not re-synthesized
    assert result["status"] == "done"  # cap hit outside the tools node -> no pending_approval


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



