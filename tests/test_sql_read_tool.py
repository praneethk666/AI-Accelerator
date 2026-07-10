from __future__ import annotations

import sys
import types

import pytest

from backend.agent_tools import AgentTool, build_agent_registry
from backend.connectors import sql_read


class _FakeCursor:
    def __init__(self):
        self.description = [type("Col", (), {"name": "id"})(), type("Col", (), {"name": "name"})()]
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query):
        self.executed = query

    def fetchall(self):
        return [(1, "alpha"), (2, "beta")]


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


def test_sql_read_tool_registered_returns_rows_and_blocks_writes(monkeypatch):
    class _FakeIngestDocumentTool:
        name = "ingest_document"
        description = ""
        input_schema = {}

        def run(self, **kwargs):
            return {}

    class _FakeSearchDocumentsTool:
        name = "search_documents"
        description = ""
        input_schema = {}

        def run(self, **kwargs):
            return {}

    fake_ingest = types.ModuleType("backend.pipeline.ingest")
    fake_ingest.IngestDocumentTool = _FakeIngestDocumentTool
    fake_search = types.ModuleType("backend.retrieval.search_documents")
    fake_search.SearchDocumentsTool = _FakeSearchDocumentsTool
    monkeypatch.setitem(sys.modules, "backend.pipeline.ingest", fake_ingest)
    monkeypatch.setitem(sys.modules, "backend.retrieval.search_documents", fake_search)

    tool = build_agent_registry()["sql_read"]
    assert isinstance(tool, AgentTool)
    assert tool.input_schema["required"] == ["query"]

    fake = _FakeConn()
    connect_calls = []
    monkeypatch.setattr(sql_read, "sql_dsn_from_env", lambda: "postgresql://example")

    def fake_connect(dsn):
        connect_calls.append(dsn)
        return fake

    monkeypatch.setattr(sql_read, "_connect", fake_connect)

    result = tool.run("SELECT id, name FROM widgets", limit=50)

    assert connect_calls == ["postgresql://example"]
    assert fake.cursor_obj.executed == "SELECT id, name FROM widgets LIMIT 50"
    assert result == {
        "columns": ["id", "name"],
        "rows": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}],
        "row_count": 2,
    }

    blocked_queries = [
        "DELETE FROM widgets",
        "UPDATE widgets SET name = 'x'",
        "DROP TABLE widgets",
        "SELECT * INTO scratch_widgets FROM widgets",
        "SELECT nextval('widget_id_seq')",
        "SELECT * FROM widgets; DELETE FROM widgets",
    ]
    for query in blocked_queries:
        with pytest.raises(ValueError):
            tool.run(query)

    assert connect_calls == ["postgresql://example"]
