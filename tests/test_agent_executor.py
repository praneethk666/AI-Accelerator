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


def test_write_tool_needs_approval_then_runs_once_approved():
    ingest = _FakeIngestTool()

    # First ask: the model wants to ingest — must be blocked, not executed.
    llm_first_ask = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": "/tmp/report.pdf"}),
    ])
    result = run_agent(
        "please ingest /tmp/report.pdf",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm_first_ask,
    )

    assert result["status"] == "needs_approval"
    assert result["answer"] is None
    assert result["pending"] == [{"id": "call_1", "name": "ingest_document", "args": {"file_path": "/tmp/report.pdf"}}]
    assert ingest.calls == []  # blocked — must not have run

    # User approves; caller re-invokes with approved_writes=True AND echoes the
    # approved call back (approval is bound to the exact name+args shown).
    llm_approved = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": "/tmp/report.pdf"}),
        AIMessage(content="Ingested — document_id doc-1, status ready."),
    ])
    result = run_agent(
        "please ingest /tmp/report.pdf",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm_approved,
        approved_writes=True,
        approved_calls=[{"name": "ingest_document", "args": {"file_path": "/tmp/report.pdf"}}],
    )

    assert result["status"] == "done"
    assert ingest.calls == [{"file_path": "/tmp/report.pdf"}]
    assert "doc-1" in result["answer"]


def test_approval_is_bound_to_approved_args():
    """Approving one file must NOT authorize a different one the model re-proposes."""
    ingest = _FakeIngestTool()
    llm = _ScriptedLLM([
        _tool_call_message("ingest_document", {"file_path": "/tmp/EVIL.pdf"}),
    ])
    result = run_agent(
        "please ingest /tmp/report.pdf",
        config=_CONFIG,
        registry={"ingest_document": ingest},
        llm=llm,
        approved_writes=True,
        approved_calls=[{"name": "ingest_document", "args": {"file_path": "/tmp/report.pdf"}}],
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

    assert len(llm.invocations) == 3  # stopped at the cap, not exhausted the script
    assert result["status"] == "done"  # cap hit outside the tools node -> no pending_approval
