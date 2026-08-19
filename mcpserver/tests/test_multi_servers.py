"""
test_multi_servers.py — Tests for specialized Gmail & Time MCP servers.
"""

import pytest
from starlette.testclient import TestClient
from src.servers.gmail_server import app as gmail_app
from src.servers.time_server import app as time_app


@pytest.fixture
def gmail_client():
    return TestClient(gmail_app)


@pytest.fixture
def time_client():
    return TestClient(time_app)


# ── Gmail Server Tests (Port 8101) ──────────────────────────────────────────

def test_gmail_server_health(gmail_client):
    res = gmail_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["server"] == "gmail-mcp-server"
    assert data["tools_count"] == 4


def test_gmail_tools_list(gmail_client):
    res = gmail_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    tools = data["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "send_email" in tool_names
    assert "read_inbox" in tool_names
    assert "search_emails" in tool_names
    assert "create_draft" in tool_names


def test_gmail_read_inbox(gmail_client):
    res = gmail_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "read_inbox", "arguments": {"max_results": 3}},
        },
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    content = data["result"]["content"][0]["text"]
    assert "emails" in content


def test_gmail_create_draft(gmail_client):
    res = gmail_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_draft",
                "arguments": {
                    "to": "vishalreddykonreddy@gmail.com",
                    "subject": "Test Draft",
                    "body": "This is a draft test.",
                },
            },
        },
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    content = data["result"]["content"][0]["text"]
    assert "draft_id" in content


# ── Time & System Server Tests (Port 8102) ──────────────────────────────────

def test_time_server_health(time_client):
    res = time_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["server"] == "time-system-mcp-server"
    assert data["tools_count"] == 3


def test_time_tools_list(time_client):
    res = time_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    tools = data["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "get_current_datetime" in tool_names
    assert "convert_timezone" in tool_names
    assert "get_system_uptime" in tool_names


def test_time_convert_timezone(time_client):
    res = time_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "convert_timezone",
                "arguments": {
                    "time_str": "2026-08-19 14:30:00",
                    "from_tz": "America/New_York",
                    "to_tz": "Asia/Kolkata",
                },
            },
        },
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    content = data["result"]["content"][0]["text"]
    assert "converted_time_iso" in content


def test_time_get_system_uptime(time_client):
    res = time_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_system_uptime", "arguments": {}},
        },
        headers={"Authorization": "Bearer vishal-test-token"},
    )
    assert res.status_code == 200
    data = res.json()
    content = data["result"]["content"][0]["text"]
    assert "uptime_seconds" in content
