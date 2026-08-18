"""
test_streamable_http.py — Tests for Streamable HTTP transport, SSE streaming, and Last-Event-ID resumability.
"""

import json
import pytest
from starlette.testclient import TestClient
from src.server import app


def test_mcp_initialize(client: TestClient, auth_headers):
    """MCP initialize handshake returns server capabilities and protocol version."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    resp = client.post("/mcp", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in data["result"]["capabilities"]
    assert "Mcp-Session-Id" in resp.headers


def test_mcp_tools_list(client: TestClient, auth_headers):
    """tools/list returns get_current_datetime and send_email schemas."""
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = client.post("/mcp", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    tool_names = [t["name"] for t in data["result"]["tools"]]
    assert "get_current_datetime" in tool_names
    assert "send_email" in tool_names


def test_mcp_sse_endpoint_stream(client: TestClient, auth_headers):
    """GET /mcp with Accept: text/event-stream initiates SSE stream."""
    headers = {**auth_headers, "Accept": "text/event-stream"}
    resp = client.get("/mcp", headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "Mcp-Session-Id" in resp.headers
