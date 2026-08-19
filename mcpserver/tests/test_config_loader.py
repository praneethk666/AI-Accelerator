import pytest
import yaml
import jsonschema
from pathlib import Path
from src.config_schema_loader import ConfigLoader

def test_valid_configuration(tmp_path):
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    server_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "schemaVersion": {"type": "string"},
            "servers": {"type": "object"}
        },
        "required": ["schemaVersion", "servers"]
    }
    with open(schema_dir / "server_schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(server_schema, f)
        
    valid_server_config = {
        "schemaVersion": "1.0",
        "servers": {
            "test-server": {
                "description": "Test",
                "version": "1.0.0",
                "transport": {"kind": "stdio", "command": "echo"},
                "auth": {"type": "api_key", "enforcement": {"toolsList": True, "toolsCall": True}, "revocationList": {"source": "${env:L}", "refreshIntervalSec": 10}},
                "network": {"allowedSourceCidrs": [], "requireMutualTLS": False},
                "auditLog": {"destination": "dest", "tamperEvident": True, "durabilityMode": "sync"},
                "dataIsolation": {"directBackingStoreAccessProhibited": True}
            }
        }
    }
    with open(config_dir / "servers.yaml", "w", encoding="utf-8") as f:
        yaml.dump(valid_server_config, f)
        
    loader = ConfigLoader(str(config_dir), str(schema_dir))
    configs = loader.load_and_validate()
    
    assert "servers" in configs
    assert configs["servers"]["schemaVersion"] == "1.0"

def test_invalid_reference(tmp_path):
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Just write some simple configs
    with open(config_dir / "servers.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"servers": {}}, f)
        
    with open(config_dir / "agents.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"agentId": "test", "mcpServers": [{"server": "unknown-server", "toolPolicy": {"allow": ["*"]}}]}, f)
        
    loader = ConfigLoader(str(config_dir), str(schema_dir))
    with pytest.raises(ValueError, match="Agent test references unknown server: unknown-server"):
        loader.load_and_validate()
