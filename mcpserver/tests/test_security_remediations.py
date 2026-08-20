import asyncio
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

from src.auth.jwt_validator import default_jwt_validator
from src.config import AppConfig, CredentialsConfig, JWTConfig, OpenBaoConfig, load_config, reset_config
from src.credentials.providers.env_provider import EnvProvider
from src.credentials.providers.openbao import OpenBaoProvider
from src.credentials.service import get_credential_service, reset_credential_service
import src.credentials.service as credential_service_module
from src.server import app
from src.servers import gmail_server


def _auth_headers():
    return {"Authorization": "Bearer test-subject"}


def test_correlation_id_propagates_to_credential_service(monkeypatch, auth_headers):
    seen = {}

    class FakeService:
        def get(self, credential_id, caller_subject=None, correlation_id=None):
            seen["credential_id"] = credential_id
            seen["caller_subject"] = caller_subject
            seen["correlation_id"] = correlation_id
            return {"username": "smtp-user", "password": "smtp-pass"}

    monkeypatch.setattr(credential_service_module, "get_credential_service", lambda cfg: FakeService())
    monkeypatch.setattr(gmail_server, "validate_email_allowlist", lambda email, allowlist, caller: (True, ""))
    monkeypatch.setattr(gmail_server, "detect_prompt_injection", lambda text, field_name, caller: (False, ""))
    monkeypatch.setattr(gmail_server, "load_config", lambda: AppConfig())

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": "allowed@example.com", "subject": "hello", "body": "world"},
                },
            },
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert seen["credential_id"] == "smtp"
    assert seen["caller_subject"] == "vishal"
    assert isinstance(seen["correlation_id"], str)
    assert len(seen["correlation_id"]) == 32


def test_openbao_provider_selected_when_configured(monkeypatch):
    reset_config()
    monkeypatch.setenv("CREDENTIAL_PROVIDER", "openbao")
    monkeypatch.setenv("OPENBAO_URL", "http://bao.example")
    monkeypatch.setenv("OPENBAO_ROLE_ID", "role-123")
    monkeypatch.setenv("OPENBAO_SECRET_ID", "secret-123")

    cfg = load_config()
    reset_credential_service()
    service = get_credential_service(cfg.credentials)

    assert cfg.credentials.provider == "openbao"
    assert cfg.credentials.openbao.role_id == "role-123"
    assert cfg.credentials.openbao.secret_id == "secret-123"
    assert isinstance(service.provider, OpenBaoProvider)


def test_openbao_failure_does_not_fall_back_to_env(monkeypatch):
    reset_credential_service()
    cfg = CredentialsConfig(provider="openbao", openbao=OpenBaoConfig("http://bao.example", "role", "secret"))

    monkeypatch.setattr(OpenBaoProvider, "get_credential", lambda self, path: None)
    monkeypatch.setattr(EnvProvider, "get_credential", lambda self, path: (_ for _ in ()).throw(AssertionError("EnvProvider must not be called")))

    service = get_credential_service(cfg)
    assert service.get("smtp", caller_subject="tester") is None


def test_read_inbox_uses_credential_service(monkeypatch):
    called = {}

    class FakeService:
        def get(self, credential_id, caller_subject=None, correlation_id=None):
            called["credential_id"] = credential_id
            called["caller_subject"] = caller_subject
            called["correlation_id"] = correlation_id
            return {"username": "smtp-user", "password": "smtp-pass"}

    monkeypatch.setattr(credential_service_module, "get_credential_service", lambda cfg: FakeService())
    monkeypatch.setattr(gmail_server, "load_config", lambda: AppConfig())

    result = asyncio.run(
        gmail_server.handle_read_inbox(
            gmail_server.ReadInboxInput(max_results=1),
            caller="test-subject",
            correlation_id="cid-123",
        )
    )

    assert result["status"] == "success"
    assert called["credential_id"] == "smtp"
    assert called["caller_subject"] == "test-subject"
    assert called["correlation_id"] == "cid-123"


def test_no_credential_leakage_in_logs(caplog, monkeypatch):
    caplog.set_level("ERROR")

    class BoomProvider:
        def __init__(self, url, role_id, secret_id):
            self.url = url
            self.role_id = role_id
            self.secret_id = secret_id

        def get_credential(self, path):
            raise RuntimeError("secret-value-should-not-appear")

    monkeypatch.setattr("src.credentials.providers.openbao.hvac.Client", lambda url: type("C", (), {"auth": type("A", (), {"approle": type("B", (), {"login": lambda self, role_id, secret_id: {"auth": {"client_token": "token-should-not-appear"}}})()})(), "is_authenticated": lambda self: True, "secrets": type("S", (), {"kv": type("K", (), {"v2": type("V", (), {"read_secret_version": lambda self, path: (_ for _ in ()).throw(RuntimeError("secret-value-should-not-appear"))})()})()})()})())

    provider = OpenBaoProvider("http://bao.example", "role-123", "secret-123")
    assert provider.get_credential("mcp/smtp") is None

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret-value-should-not-appear" not in combined
    assert "token-should-not-appear" not in combined
    assert "Traceback" not in combined


def test_sanitized_openbao_failures(caplog, monkeypatch):
    caplog.set_level("ERROR")

    class FakeClient:
        def __init__(self, url):
            self.url = url
            self.auth = type("Auth", (), {"approle": type("AppRole", (), {"login": lambda self, role_id, secret_id: (_ for _ in ()).throw(RuntimeError("login-secret-leak"))})()})()
            self.secrets = type("Secrets", (), {"kv": type("KV", (), {"v2": type("V2", (), {"read_secret_version": lambda self, path: (_ for _ in ()).throw(RuntimeError("read-secret-leak"))})()})()})()

        def is_authenticated(self):
            return False

    monkeypatch.setattr("src.credentials.providers.openbao.hvac.Client", FakeClient)

    provider = OpenBaoProvider("http://bao.example", "role-123", "secret-123")
    assert provider.get_credential("mcp/smtp") is None

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "login-secret-leak" not in combined
    assert "read-secret-leak" not in combined
    assert "Traceback" not in combined


def test_rs256_jwks_validation(monkeypatch):
    reset_config()
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

    cfg = AppConfig()
    cfg.auth.jwt = JWTConfig(
        enabled=True,
        algorithms=["RS256"],
        issuer="issuer-1",
        audience="aud-1",
        jwks_url="https://example.test/jwks.json",
    )
    monkeypatch.setattr("src.auth.jwt_validator.load_config", lambda: cfg)

    class FakeSigningKey:
        key = public_pem

    class FakeJWKClient:
        def __init__(self, url):
            self.url = url

        def get_signing_key_from_jwt(self, token):
            return FakeSigningKey()

    monkeypatch.setattr("jwt.PyJWKClient", FakeJWKClient)

    token = jwt.encode(
        {
            "sub": "agent-rs",
            "iss": "issuer-1",
            "aud": "aud-1",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "kid-1"},
    )

    auth_ctx = default_jwt_validator.validate(token)
    assert auth_ctx is not None
    assert auth_ctx.client_id == "agent-rs"
