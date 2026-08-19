import os
import yaml
import jsonschema
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_dir: str, schema_dir: str):
        self.config_dir = Path(config_dir)
        self.schema_dir = Path(schema_dir)
        self.schemas: Dict[str, Any] = {}
        self._load_schemas()

    def _load_schemas(self):
        schema_files = {
            "agents": "agent_schema.yaml",
            "roles": "roles_schema.yaml",
            "servers": "server_schema.yaml",
            "orchestration": "orchestration_schema.yaml",
        }
        for key, filename in schema_files.items():
            schema_path = self.schema_dir / filename
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.schemas[key] = yaml.safe_load(f)

    def load_and_validate(self) -> Dict[str, Any]:
        """Loads and validates all configurations against their schemas."""
        configs = {}
        config_files = {
            "agents": "agents.yaml",
            "roles": "roles.yaml",
            "servers": "servers.yaml",
            "orchestration": "orchestration.yaml",
        }
        
        for key, filename in config_files.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                configs[key] = {}
                continue
                
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                configs[key] = data
                
            if key in self.schemas:
                try:
                    jsonschema.validate(instance=data, schema=self.schemas[key])
                except jsonschema.ValidationError as e:
                    logger.error(f"Validation error in {filename}: {e.message}")
                    raise ValueError(f"Invalid configuration in {filename}: {e.message}")

        self._validate_references(configs)
        return configs

    def _validate_references(self, configs: Dict[str, Any]):
        """Validates cross-references between loaded configurations."""
        agents = configs.get("agents", {})
        roles = configs.get("roles", {}).get("roles", {})
        servers = configs.get("servers", {}).get("servers", {})

        # Ensure agent references valid role and valid servers
        # agents is expected to be single profile per file or a list if aggregated.
        # But wait, agent_schema.yaml is "One file per agent." 
        # Typically agents are in a directory or a single aggregated file.
        # If agents.yaml is an aggregation, let's treat it as a dict mapping id->detail.
        if isinstance(agents, dict):
            # The schema says "Defines which MCP servers and tools a single agent may use... One file per agent."
            # So if agents.yaml is just one agent profile:
            agent_id = agents.get("agentId")
            role = agents.get("role")
            if role and role not in roles:
                # We skip strictly failing here if roles.yaml wasn't completely loaded or it's external,
                # but let's log a warning or raise.
                # If we expect the roles to be there:
                logger.warning(f"Agent {agent_id} references unknown role: {role}")

            mcp_servers = agents.get("mcpServers", [])
            for binding in mcp_servers:
                server_name = binding.get("server")
                if server_name and server_name not in servers:
                    raise ValueError(f"Agent {agent_id} references unknown server: {server_name}")
                    
        # Orchestration cross references
        orchestration = configs.get("orchestration", {}).get("capabilities", {})
        for cap_name, cap_def in orchestration.items():
            for binding in cap_def.get("bindings", []):
                server_name = binding.get("server")
                if server_name and server_name not in servers:
                    raise ValueError(f"Orchestration binding '{binding.get('bindingId')}' references unknown server: {server_name}")

