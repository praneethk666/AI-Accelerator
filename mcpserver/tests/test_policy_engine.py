import pytest
from src.security.policy_engine import PolicyEngine

@pytest.fixture
def policy_engine():
    roles_config = {
        "classificationLevels": [{"name": "public"}, {"name": "secret"}]
    }
    return PolicyEngine(roles_config)

def test_allowed_tool(policy_engine):
    identity = {
        "agentId": "vishal",
        "profile": {
            "mcpServers": [
                {
                    "server": "local",
                    "toolPolicy": {"allow": ["ping"]}
                }
            ]
        },
        "role": {"allow": ["ping"]}
    }
    assert policy_engine.evaluate(identity, "local", "ping") == "ALLOW"

def test_unknown_identity(policy_engine):
    assert policy_engine.evaluate(None, "local", "ping") == "DENY"
    assert policy_engine.evaluate({}, "local", "ping") == "DENY"

def test_unknown_role(policy_engine):
    identity = {
        "agentId": "vishal",
        "profile": {
            "mcpServers": [{"server": "local", "toolPolicy": {"allow": ["ping"]}}]
        },
        "role": None
    }
    assert policy_engine.evaluate(identity, "local", "ping") == "DENY"

def test_unknown_server(policy_engine):
    identity = {
        "agentId": "vishal",
        "profile": {
            "mcpServers": [{"server": "local", "toolPolicy": {"allow": ["ping"]}}]
        },
        "role": {"allow": ["ping"]}
    }
    assert policy_engine.evaluate(identity, "unknown_server", "ping") == "DENY"

def test_unknown_tool(policy_engine):
    identity = {
        "agentId": "vishal",
        "profile": {
            "mcpServers": [{"server": "local", "toolPolicy": {"allow": ["ping"]}}]
        },
        "role": {"allow": ["ping"]}
    }
    assert policy_engine.evaluate(identity, "local", "unknown_tool") == "DENY"

def test_cross_server_deny(policy_engine):
    identity = {
        "agentId": "vishal",
        "profile": {
            "mcpServers": [
                {"server": "server_a", "toolPolicy": {"allow": ["tool_a"]}},
                {"server": "server_b", "toolPolicy": {"allow": ["tool_b"]}}
            ]
        },
        "role": {"allow": ["tool_a", "tool_b"]}
    }
    assert policy_engine.evaluate(identity, "server_a", "tool_a") == "ALLOW"
    assert policy_engine.evaluate(identity, "server_a", "tool_b") == "DENY"
    assert policy_engine.evaluate(identity, "server_b", "tool_a") == "DENY"
    assert policy_engine.evaluate(identity, "server_b", "tool_b") == "ALLOW"
