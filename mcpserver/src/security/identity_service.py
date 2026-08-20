import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class IdentityService:
    def __init__(self, agents_config: Dict[str, Any], roles_config: Dict[str, Any]):
        self.agents = {}
        if isinstance(agents_config, dict) and "agentId" in agents_config:
            self.agents[agents_config["agentId"]] = agents_config
        elif isinstance(agents_config, dict):
            for k, v in agents_config.items():
                if isinstance(v, dict) and "agentId" in v:
                    self.agents[v["agentId"]] = v
                elif isinstance(v, dict):
                    self.agents[k] = v
        elif isinstance(agents_config, list):
            for agent in agents_config:
                if "agentId" in agent:
                    self.agents[agent["agentId"]] = agent

        self.roles = roles_config.get("roles", {})
        self.classification_levels = roles_config.get("classificationLevels", [])

    def resolve(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Validates the subject against the agent configurations and returns
        the resolved agent identity and role context.
        """
        agent = self.agents.get(subject_id)
        if agent:
            cred = agent.get("credential", {})
            # Check expiration
            expires_at = cred.get("expiresAt")
            if expires_at:
                try:
                    dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    if datetime.now(dt.tzinfo or timezone.utc) > dt:
                        logger.warning(f"Agent {subject_id} credential expired.")
                        return None
                except ValueError:
                    pass
            
            return {
                "agentId": subject_id,
                "profile": agent,
                "role": self._resolve_role(agent.get("role"))
            }

        logger.warning(f"Authentication failed: subject '{subject_id}' does not map to any active agent.")
        return None

    def _resolve_role(self, role_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not role_name:
            return None
        return self.roles.get(role_name)
