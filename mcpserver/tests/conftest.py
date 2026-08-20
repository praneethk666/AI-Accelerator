"""
conftest.py — Pytest configuration and shared test fixtures.
"""

import pytest
from starlette.testclient import TestClient

from src.auth.session_manager import session_manager
from src.config import load_config, reset_config
from src.credentials.service import reset_credential_service
from src.security.rate_limiter import rate_limiter
from src.server import app


@pytest.fixture(autouse=True)
def clean_state():
    """Resets session manager, rate limiter, and config state before each test."""
    import src.globals as g
    reset_credential_service()
    session_manager._sessions.clear()
    rate_limiter.reset()
    reset_config()
    g.identity_service = None
    g.policy_engine = None
    yield
    session_manager._sessions.clear()
    rate_limiter.reset()
    reset_config()
    g.identity_service = None
    g.policy_engine = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


from src.auth.jwt_validator import default_jwt_validator
from src.config import JWTConfig
import jwt
from datetime import datetime, timedelta, timezone

@pytest.fixture
def auth_headers():
    cfg = load_config()
    if cfg.auth.jwt and cfg.auth.jwt.enabled:
        payload = {
            "sub": "vishal",
            "iss": cfg.auth.jwt.issuer,
            "aud": cfg.auth.jwt.audience,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        token = jwt.encode(payload, cfg.auth.jwt.secret_key, algorithm="HS256")
        cfg.auth.tokens["vishal"] = "vishal_engineer"
        return {"Authorization": f"Bearer {token}"}
    return {"Authorization": "Bearer vishal"}
