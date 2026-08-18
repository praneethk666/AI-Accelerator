"""
test_server_health.py — Unit tests for the /health and /ready probe endpoints.
"""

import pytest
from starlette.testclient import TestClient
from src.server import app


def test_health_check_returns_200():
    """GET /health returns status: healthy with HTTP 200."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "streamable-mcp-server"


def test_readiness_check_returns_200():
    """GET /ready returns status: ready with HTTP 200."""
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["tools_count"] == 2
