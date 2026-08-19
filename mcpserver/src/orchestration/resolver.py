import logging
from typing import Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

class OrchestrationResolver:
    def __init__(self, orchestration_config: Dict[str, Any], health_checker: Optional[Callable[[str], bool]] = None):
        self.capabilities = orchestration_config.get("capabilities", {})
        self.health_checker = health_checker or (lambda _: True)

    def resolve(self, requested_name: str) -> Tuple[Optional[str], str]:
        """
        Takes a requested orchestrator capability (e.g. 'send-notification')
        or a direct tool name (e.g. 'send_email').
        
        Returns (server_name, tool_name) where server_name is the target
        server providing the capability. Skip unhealthy servers.
        """
        if requested_name in self.capabilities:
            cap = self.capabilities[requested_name]
            bindings = cap.get("bindings", [])
            for binding in bindings:
                server_name = binding.get("server", "local")
                if self.health_checker(server_name):
                    return server_name, binding.get("tool")
            
        return "local", requested_name
