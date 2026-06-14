"""Query-pipeline orchestration tests (no LLM / no infra — stub tools).

Proves run_query wires planner -> retrieval -> answerer through the same graph
engine, initializes state so direct-index reads don't KeyError, and degrades
gracefully when a query tool fails.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.registry import ToolRegistry
from backend.pipeline.query import run_query


class _PlannerStub:
    name = "query_planner"

    def run(self, state, config):
        state["sub_questions"] = [state["query"]]
        state["standalone_query"] = state["query"]
        return state


class _RetrievalStub:
    name = "retrieval"

    def run(self, state, config):
        # exercises the same direct-index reads the real tool uses
        subs = state["sub_questions"] or []
        _ = state["document_scope"]
        state["retrieved_chunks"] = [
            {"chunk_id": "c1", "text": f"hit for {subs[0]}"} if subs else {}
        ]
        return state


class _AnswererStub:
    name = "answerer"

    def run(self, state, config):
        chunks = state["retrieved_chunks"] or []
        state["answer"] = f"answer from {len(chunks)} chunk(s)"
        state["citations"] = []
        return state


def _registry():
    reg = ToolRegistry()
    reg.register(_PlannerStub())
    reg.register(_RetrievalStub())
    reg.register(_AnswererStub())
    return reg


CFG = {"query": {"steps": ["query_planner", "retrieval", "answerer"]}}


def test_query_runs_end_to_end():
    out = run_query("What is the torque spec?", _registry(), CFG)
    assert out["sub_questions"] == ["What is the torque spec?"]
    assert out["retrieved_chunks"]
    assert out["answer"] == "answer from 1 chunk(s)"
    assert out["errors"] == []


def test_query_initializes_state_fields():
    # no session/scope passed -> tools that read by direct index must still work
    out = run_query("hi", _registry(), CFG)
    for key in ("document_scope", "conversation_history", "citations", "errors"):
        assert key in out


def test_query_degrades_when_a_tool_fails():
    class _Boom:
        name = "retrieval"

        def run(self, state, config):
            raise RuntimeError("retrieval boom")

    reg = ToolRegistry()
    reg.register(_PlannerStub())
    reg.register(_Boom())
    reg.register(_AnswererStub())
    out = run_query("q", reg, CFG)
    assert any("retrieval boom" in str(e) for e in out["errors"])
    assert "answer" in out  # pipeline kept going to the answerer


if __name__ == "__main__":
    test_query_runs_end_to_end()
    test_query_initializes_state_fields()
    test_query_degrades_when_a_tool_fails()
    print("query pipeline tests passed")
