import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PolicyEngine:
    def __init__(self, roles_config: Dict[str, Any]):
        self.classification_levels = roles_config.get("classificationLevels", [])
        self.rank_map = {lvl["name"]: idx for idx, lvl in enumerate(self.classification_levels)}

    def evaluate(self, identity: Dict[str, Any], server_name: str, tool_name: str) -> str:
        """
        Evaluates whether the agent can execute the tool on the given server.
        Returns 'ALLOW', 'DENY', or 'REQUIRE_APPROVAL'.
        """
        agent = identity.get("profile", {})
        role = identity.get("role")

        # 1. Check Role Ceilings
        if role:
            role_allow = role.get("allow", [])
            role_deny = role.get("deny", [])
            
            if tool_name in role_deny:
                logger.warning(f"DENY: Tool {tool_name} explicitly denied by role {agent.get('role')}.")
                return "DENY"
                
            if "*" not in role_allow and tool_name not in role_allow:
                logger.warning(f"DENY: Tool {tool_name} not in role {agent.get('role')} allowlist.")
                return "DENY"

        # 2. Check Agent Tool Policy
        mcp_servers = agent.get("mcpServers", [])
        server_binding = next((s for s in mcp_servers if s.get("server") == server_name), None)
        
        if not server_binding:
            logger.warning(f"DENY: Server {server_name} not bound to agent {identity.get('agentId')}.")
            return "DENY"

        policy = server_binding.get("toolPolicy", {})
        allow = policy.get("allow", [])
        deny = policy.get("deny", [])

        if tool_name in deny:
            logger.warning(f"DENY: Tool {tool_name} explicitly denied by agent profile.")
            return "DENY"

        if "*" not in allow and tool_name not in allow:
            logger.warning(f"DENY: Tool {tool_name} not in agent profile allowlist.")
            return "DENY"

        # 3. Check Overrides (Require Approval)
        overrides = server_binding.get("toolOverrides", {}).get(tool_name, {})
        if overrides.get("requireConfirmation"):
            return "REQUIRE_APPROVAL"

        return "ALLOW"
