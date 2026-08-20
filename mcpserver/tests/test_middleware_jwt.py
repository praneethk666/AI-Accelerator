import pytest
import jwt
from datetime import datetime, timedelta, timezone
from starlette.testclient import TestClient
from src.server import app

@pytest.fixture
def override_config(monkeypatch):
    from src.config import AppConfig, AuthConfig, JWTConfig
    
    cfg = AppConfig()
    cfg.auth.jwt = JWTConfig(
        enabled=True,
        secret_key="my-secret-key",
        issuer="test-issuer",
        audience="test-audience",
        algorithms=["HS256"]
    )
    # Give some valid subject binding in IdentityService or fallback
    cfg.auth.tokens = {"agent-007": "agent-007"} 
    
    def mock_load_config(*args, **kwargs):
        return cfg
        
    monkeypatch.setattr("src.auth.middleware.load_config", mock_load_config)
    monkeypatch.setattr("src.auth.jwt_validator.load_config", mock_load_config)
    
    # Also patch Identity Service so token fallback triggers or identity maps properly
    # In middleware, if g.identity_service is active, we should stub it
    import src.globals as g
    
    class MockIdentityService:
        def resolve(self, subject: str):
            if subject == "agent-007":
                return {"agentId": "agent-007", "profile": {}, "role": "admin"}
            return None
            
    g.identity_service = MockIdentityService()
    
    return cfg

def create_valid_token(sub="agent-007"):
    payload = {
        "sub": sub,
        "iss": "test-issuer",
        "aud": "test-audience",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")

client = TestClient(app)

def test_missing_authentication_returns_401(override_config):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert response.status_code == 401
    assert "Missing authentication credential" in response.text

def test_valid_jwt_success(override_config):
    token = create_valid_token()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"

def test_jwt_maps_to_existing_identity(override_config):
    # Subject matches agent-007 in MockIdentityService
    token = create_valid_token("agent-007")
    
    # We can perform a tools/list request which triggers policy eval
    # Just checking auth success is enough to prove mapping worked
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

def test_mismatched_identity_session_rejection(override_config):
    # 1. Agent 007 creates session
    token1 = create_valid_token("agent-007")
    resp1 = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": f"Bearer {token1}"}
    )
    session_id = resp1.headers.get("Mcp-Session-Id")
    assert session_id is not None
    
    # 2. Agent 008 tries to reuse it
    # Setup agent 008 in mock
    import src.globals as g
    
    class MockIdentityServiceDual:
        def resolve(self, subject: str):
            if subject in ["agent-007", "agent-008"]:
                return {"agentId": subject, "profile": {}, "role": "admin"}
            return None
    g.identity_service = MockIdentityServiceDual()
    
    token2 = create_valid_token("agent-008")
    resp2 = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        headers={
            "Authorization": f"Bearer {token2}",
            "Mcp-Session-Id": session_id
        }
    )
    
    # Should get 403 Session Mismatch
    assert resp2.status_code == 403
    assert "Session ID belongs to another" in resp2.text

def test_expired_jwt_returns_401(override_config):
    payload = {
        "sub": "agent-007",
        "iss": "test-issuer",
        "aud": "test-audience",
        "exp": datetime.now(timezone.utc) - timedelta(hours=1)
    }
    token = jwt.encode(payload, "my-secret-key", algorithm="HS256")
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
