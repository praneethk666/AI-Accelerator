"""Tests for the agent executor — the tool-calling loop.

All tests use a scripted mock model + fake tools (per the ticket: headless, no
network, no infra). The mock returns pre-baked AIMessages; the fakes record calls.
"""
import json
import os
import sys

import pytest
from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.executor import run_agent  # noqa: E402
from backend.agent_tools import build_agent_registry  # noqa: E402


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeSearch:
    """Read tool: runs without approval."""
    name = "fake_search"
    description = "search the test corpus"
    input_schema = {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}

    def __init__(self):
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        return {"hits": [f"hit for {query}"]}


class FakeIngest:
    """Write tool: must be approved."""
    name = "fake_ingest"
    writes = True
    description = "ingest a file into the test corpus"
    input_schema = {"type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"]}

    def __init__(self):
        self.calls = []

    def run(self, file_path):
        self.calls.append(file_path)
        return {"document_id": "d1", "status": "ready"}


class FakeBoom:
    name = "fake_boom"
    description = "always raises"
    input_schema = {"type": "object", "properties": {}}

    def run(self, **kwargs):
        raise RuntimeError("kaboom")


class MockLLM:
    """Scripted model: pops one canned AIMessage per invoke."""

    def __init__(self, script):
        self.script = list(script)
        self.advertised = None
        self.seen_messages = []

    def bind_tools(self, tools):
        self.advertised = [t["function"]["name"] for t in tools]
        return self

    def invoke(self, messages):
        self.seen_messages.append(list(messages))
        return self.script.pop(0)


def _call(name, args, cid):
    return {"name": name, "args": args, "id": cid}


# ── the loop ──────────────────────────────────────────────────────────────────
def test_loop_calls_one_tool_then_answers():
    search = FakeSearch()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"query": "torque spec"}, "c1")]),
        AIMessage(content="The torque spec is 42 Nm."),
    ])
    out = run_agent("what is the torque spec?", registry={"fake_search": search}, llm=llm)

    assert llm.advertised == ["fake_search"]          # tools advertised to the model
    assert search.calls == ["torque spec"]            # executor ran the pick
    assert out["answer"] == "The torque spec is 42 Nm."
    assert out["stopped"] == "final" and out["iterations"] == 2
    assert out["tool_calls"] == [{"tool": "fake_search", "args": {"query": "torque spec"},
                                  "ok": True, "result": {"hits": ["hit for torque spec"]}}]


def test_tool_result_is_fed_back_to_the_model():
    search = FakeSearch()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"query": "q"}, "c1")]),
        AIMessage(content="done"),
    ])
    run_agent("q?", registry={"fake_search": search}, llm=llm)
    # second invoke must contain a ToolMessage carrying the tool's JSON result
    second_turn = llm.seen_messages[1]
    tool_msgs = [m for m in second_turn if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "c1"
    assert json.loads(tool_msgs[0].content) == {"hits": ["hit for q"]}


def test_multiple_calls_in_one_turn_all_run():
    search = FakeSearch()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"query": "a"}, "c1"),
                                          _call("fake_search", {"query": "b"}, "c2")]),
        AIMessage(content="both done"),
    ])
    out = run_agent("a and b?", registry={"fake_search": search}, llm=llm)
    assert search.calls == ["a", "b"]
    assert [t["ok"] for t in out["tool_calls"]] == [True, True]


def test_no_tool_calls_means_direct_answer():
    llm = MockLLM([AIMessage(content="I can answer directly.")])
    out = run_agent("hi", registry={}, llm=llm)
    assert out["answer"] == "I can answer directly."
    assert out["iterations"] == 1 and out["tool_calls"] == []


def test_max_iterations_cap():
    # model keeps asking for tools forever -> loop stops at the cap
    calls = [AIMessage(content="", tool_calls=[_call("fake_search", {"query": "x"}, f"c{i}")])
             for i in range(10)]
    llm = MockLLM(calls)
    out = run_agent("loop", registry={"fake_search": FakeSearch()}, llm=llm, max_iterations=3)
    assert out["stopped"] == "max_iterations" and out["iterations"] == 3
    assert len(out["tool_calls"]) == 3


# ── write-approval gate ───────────────────────────────────────────────────────
def test_write_tool_runs_when_approved():
    ingest = FakeIngest()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_ingest", {"file_path": "a.pdf"}, "c1")]),
        AIMessage(content="ingested"),
    ])
    approvals = []

    def approve(name, args):
        approvals.append((name, args))
        return True

    out = run_agent("ingest a.pdf", registry={"fake_ingest": ingest}, llm=llm, approve=approve)
    assert approvals == [("fake_ingest", {"file_path": "a.pdf"})]  # asked first
    assert ingest.calls == ["a.pdf"]                               # then ran
    assert out["tool_calls"][0]["ok"] is True


def test_write_tool_refused_when_denied():
    ingest = FakeIngest()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_ingest", {"file_path": "a.pdf"}, "c1")]),
        AIMessage(content="ok, not ingesting"),
    ])
    out = run_agent("ingest a.pdf", registry={"fake_ingest": ingest}, llm=llm,
                    approve=lambda name, args: False)
    assert ingest.calls == []                                      # never ran
    assert out["tool_calls"][0]["ok"] is False
    assert "not approved" in out["tool_calls"][0]["result"]["error"]


def test_write_tool_refused_without_approval_callback():
    ingest = FakeIngest()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_ingest", {"file_path": "a.pdf"}, "c1")]),
        AIMessage(content="cannot ingest"),
    ])
    out = run_agent("ingest a.pdf", registry={"fake_ingest": ingest}, llm=llm)  # no approve
    assert ingest.calls == []                                      # safe default: refused
    assert out["tool_calls"][0]["ok"] is False
    assert "refused" in out["tool_calls"][0]["result"]["error"]


def test_read_tool_never_asks_for_approval():
    search = FakeSearch()
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"query": "q"}, "c1")]),
        AIMessage(content="done"),
    ])
    asked = []
    run_agent("q?", registry={"fake_search": search}, llm=llm,
              approve=lambda n, a: asked.append(n) or True)
    assert asked == [] and search.calls == ["q"]                   # ran directly


# ── failure surfaces, loop survives ───────────────────────────────────────────
def test_unknown_tool_is_reported_not_fatal():
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("made_up_tool", {}, "c1")]),
        AIMessage(content="sorry, no such tool"),
    ])
    out = run_agent("x", registry={}, llm=llm)
    assert out["stopped"] == "final"
    assert out["tool_calls"][0]["ok"] is False
    assert "unknown tool" in out["tool_calls"][0]["result"]["error"]


def test_bad_args_reported_not_fatal():
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"wrong_arg": 1}, "c1")]),
        AIMessage(content="bad args"),
    ])
    out = run_agent("x", registry={"fake_search": FakeSearch()}, llm=llm)
    assert out["tool_calls"][0]["ok"] is False
    assert "invalid arguments" in out["tool_calls"][0]["result"]["error"]


def test_provider_tool_validation_400_triggers_retry():
    # Groq-style: the API itself 400s on malformed model tool calls. The loop
    # must feed the error back and let the model retry, not crash.
    search = FakeSearch()

    class FlakyLLM(MockLLM):
        def __init__(self, script):
            super().__init__(script)
            self.raised = False

        def invoke(self, messages):
            if not self.raised:
                self.raised = True
                raise RuntimeError(
                    "Error code: 400 - tool call validation failed: parameters "
                    "for tool fake_search did not match schema (tool_use_failed)")
            return super().invoke(messages)

    llm = FlakyLLM([
        AIMessage(content="", tool_calls=[_call("fake_search", {"query": "q"}, "c1")]),
        AIMessage(content="recovered"),
    ])
    out = run_agent("q?", registry={"fake_search": search}, llm=llm, max_iterations=5)
    assert out["answer"] == "recovered" and out["stopped"] == "final"
    assert search.calls == ["q"]                      # the corrected retry ran
    # the corrective feedback message reached the model on the retry turn
    retry_turn = llm.seen_messages[0]
    assert any("did not match" in getattr(m, "content", "") for m in retry_turn)


def test_unrelated_provider_error_still_raises():
    class DeadLLM(MockLLM):
        def invoke(self, messages):
            raise RuntimeError("Error code: 500 - internal server error")

    with pytest.raises(RuntimeError, match="500"):
        run_agent("q?", registry={}, llm=DeadLLM([]))


def test_tool_exception_reported_not_fatal():
    llm = MockLLM([
        AIMessage(content="", tool_calls=[_call("fake_boom", {}, "c1")]),
        AIMessage(content="it failed"),
    ])
    out = run_agent("x", registry={"fake_boom": FakeBoom()}, llm=llm)
    assert out["stopped"] == "final"
    assert out["tool_calls"][0]["ok"] is False
    assert "kaboom" in out["tool_calls"][0]["result"]["error"]


# ── real registry wiring (no tools actually run) ──────────────────────────────
def test_real_registry_write_markers():
    reg = build_agent_registry()
    assert set(reg) == {"ingest_document", "search_documents"}
    assert getattr(reg["ingest_document"], "writes", False) is True     # gated
    assert getattr(reg["search_documents"], "writes", False) is False   # direct


def test_real_registry_advertises_to_model():
    llm = MockLLM([AIMessage(content="hello")])
    out = run_agent("hi", registry=build_agent_registry(), llm=llm)
    assert sorted(llm.advertised) == ["ingest_document", "search_documents"]
    assert out["answer"] == "hello"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
