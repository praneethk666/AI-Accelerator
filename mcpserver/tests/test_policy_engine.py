import pytest
from src.security.identity_service import IdentityService
from src.security.policy_engine import PolicyEngine
from src.orchestration.resolver import OrchestrationResolver

@pytest.fixture
def sample_agents():
    return {
        "agentId": "test-agent",
        "credential": {
            "source": "dummy-token"
        },
        "role": "test-role",
        "mcpServers": [
            {
                "server": "local",
                "toolPolicy": {"allow": ["send_email"], "deny": ["dangerous_tool"]}
            },
            {
                "server": "remote-db",
                "toolPolicy": {"allow": ["query_db"]}
            }
        ]
    }

@pytest.fixture
def sample_roles():
    return {
        "classificationLevels": [
            {"name": "public", "description": "Public", "externalShareAllowed": True},
            {"name": "secret", "description": "Secret", "externalShareAllowed": False}
        ],
        "roles": {
            "test-role": {
                "description": "A test role",
                "allow": ["*"],
                "deny": ["global_deny_tool"]
            },
            "strict-role": {
                "description": "Strict",
                "allow": ["get_current_datetime"]
            }
        }
    }

def test_allowed_agent(sample_agents, sample_roles):
    identity = {
        "agentId": "test-agent",
        "profile": sample_agents,
        "role": sample_roles["roles"]["test-role"]
    }
    pe = PolicyEngine(sample_roles)
    res = pe.evaluate(identity, "local", "send_email")
    assert res == "ALLOW"

def test_denied_tool(sample_agents, sample_roles):
    identity = {
        "agentId": "test-agent",
        "profile": sample_agents,
        "role": sample_roles["roles"]["test-role"]
    }
    pe = PolicyEngine(sample_roles)
    res = pe.evaluate(identity, "local", "dangerous_tool")
    assert res == "DENY"

def test_denied_server(sample_agents, sample_roles):
    identity = {
        "agentId": "test-agent",
        "profile": sample_agents,
        "role": sample_roles["roles"]["test-role"]
    }
    pe = PolicyEngine(sample_roles)
    # Agent doesn't have bound server 'unknown-server'
    res = pe.evaluate(identity, "unknown-server", "send_email")
    assert res == "DENY"

def test_role_restrictions(sample_agents, sample_roles):
    identity = {
        "agentId": "strict-agent",
        "profile": {
            "agentId": "strict-agent",
            "mcpServers": [{"server": "local", "toolPolicy": {"allow": ["send_email", "get_current_datetime"]}}]
        },
        "role": sample_roles["roles"]["strict-role"]
    }
    pe = PolicyEngine(sample_roles)
    # Even though agent allows send_email, role restricts it
    res = pe.evaluate(identity, "local", "send_email")
    assert res == "DENY"
    
    # get_current_datetime is allowed by role and agent
    res = pe.evaluate(identity, "local", "get_current_datetime")
    assert res == "ALLOW"

def test_global_role_deny(sample_agents, sample_roles):
    identity = {
        "agentId": "test-agent",
        "profile": sample_agents, # agent allows send_email, but not global_deny_tool
        "role": sample_roles["roles"]["test-role"] # role denies global_deny_tool
    }
    pe = PolicyEngine(sample_roles)
    res = pe.evaluate(identity, "local", "global_deny_tool")
    assert res == "DENY"
