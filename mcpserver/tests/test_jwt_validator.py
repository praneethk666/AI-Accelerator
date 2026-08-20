import pytest
import jwt
from datetime import datetime, timedelta, timezone
from src.auth.jwt_validator import default_jwt_validator
from src.config import load_config, JWTConfig, AuthConfig, AppConfig
from src.auth.models import AuthContext

@pytest.fixture
def mock_config(monkeypatch):
    # Mock load_config to return our custom config
    cfg = AppConfig()
    cfg.auth.jwt = JWTConfig(
        enabled=True,
        secret_key="secret",
        issuer="my-issuer",
        audience="my-audience",
        algorithms=["HS256"]
    )
    monkeypatch.setattr("src.auth.jwt_validator.load_config", lambda: cfg)
    monkeypatch.setattr("src.auth.middleware.load_config", lambda: cfg)
    return cfg

def create_token(payload, secret="secret", algorithm="HS256"):
    return jwt.encode(payload, secret, algorithm=algorithm)

def test_valid_jwt(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "agent123",
        "iss": "my-issuer",
        "aud": "my-audience",
        "exp": now + timedelta(hours=1)
    }
    token = create_token(payload)
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is not None
    assert auth_ctx.client_id == "agent123"
    assert auth_ctx.subject == "agent123"

def test_invalid_signature(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "agent123",
        "iss": "my-issuer",
        "aud": "my-audience",
        "exp": now + timedelta(hours=1)
    }
    # Write with wrong secret
    token = create_token(payload, secret="wrong-secret")
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is None

def test_expired_jwt(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "agent123",
        "iss": "my-issuer",
        "aud": "my-audience",
        "exp": now - timedelta(hours=1) # Expired
    }
    token = create_token(payload)
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is None

def test_wrong_issuer(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "agent123",
        "iss": "wrong-issuer",
        "aud": "my-audience",
        "exp": now + timedelta(hours=1)
    }
    token = create_token(payload)
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is None

def test_wrong_audience(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "agent123",
        "iss": "my-issuer",
        "aud": "wrong-audience",
        "exp": now + timedelta(hours=1)
    }
    token = create_token(payload)
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is None

def test_missing_subject(mock_config):
    now = datetime.now(timezone.utc)
    payload = {
        "iss": "my-issuer",
        "aud": "my-audience",
        "exp": now + timedelta(hours=1)
    }
    token = create_token(payload)
    
    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is None

def test_malformed_jwt(mock_config):
    auth_ctx = default_jwt_validator.validate("not-a-token")
    assert auth_ctx is None
