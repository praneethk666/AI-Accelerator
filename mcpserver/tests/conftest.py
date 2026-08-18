"""
conftest.py — Pytest configuration and shared test fixtures.
"""

import pytest
from starlette.testclient import TestClient

from src.auth.session_manager import session_manager
from src.config import load_config, reset_config
from src.security.rate_limiter import rate_limiter
from src.server import app


@pytest.fixture(autouse=True)
def clean_state():
    """Resets session manager, rate limiter, and config state before each test."""
    session_manager._sessions.clear()
    rate_limiter.reset()
    reset_config()
    yield
    session_manager._sessions.clear()
    rate_limiter.reset()
    reset_config()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer vishal-test-token"}
