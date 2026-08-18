"""
test_security_auth.py — Tests for Bearer token auth, Origin checking, and session identity binding.
"""

import pytest
from starlette.testclient import TestClient
from src.server import app
from src.auth.session_manager import session_manager


def test_health_probe_open(client: TestClient):
    """Health probe requires no authentication."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_ready_probe_open(client: TestClient):
    """Ready probe requires no authentication."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_mcp_unauthenticated_request_rejected(client: TestClient):
    """Calling /mcp without token returns 401."""
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == -32001


def test_mcp_invalid_token_rejected(client: TestClient):
    """Calling /mcp with invalid token returns 401."""
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": "Bearer bad-token-999"},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == -32001


def test_session_identity_binding_enforcement(client: TestClient):
    """A session ID created by Caller A cannot be used by Caller B."""
    sid = "session-binding-test-uuid"

    # 1. Caller A authenticates and creates session
    resp1 = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": "Bearer agent-token-alpha", "Mcp-Session-Id": sid},
    )
    assert resp1.status_code == 200

    # 2. Caller B attempts to use the same session ID
    resp2 = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        headers={"Authorization": "Bearer agent-token-beta", "Mcp-Session-Id": sid},
    )
    assert resp2.status_code == 403
    data2 = resp2.json()
    assert data2["error"]["code"] == -32005
    assert "belongs to another caller" in data2["error"]["message"]
