from typing import Optional, Any
from src.security.identity_service import IdentityService
from src.security.policy_engine import PolicyEngine
from src.orchestration.resolver import OrchestrationResolver

identity_service: Optional[IdentityService] = None
policy_engine: Optional[PolicyEngine] = None
orchestration_resolver: Optional[OrchestrationResolver] = None
agent_runtime: Optional[Any] = None
