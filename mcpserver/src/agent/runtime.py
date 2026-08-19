from typing import Callable, Optional
from .state import WorkflowState, StepResult
from .planner import Planner
from .decision import DecisionEngine
from .execution import ExecutionEngine
from .evaluator import ResultEvaluator
from src.security.audit import AuditLogger
import time

class AgentRuntime:
    def __init__(
        self,
        planner: Planner,
        decision_engine: DecisionEngine,
        execution_engine: ExecutionEngine,
        evaluator: ResultEvaluator,
        policy_checker: Callable[[str, str], str],
        audit_logger: Optional[AuditLogger] = None,
        idempotent_tools: list = None
    ):
        self.planner = planner
        self.decision_engine = decision_engine
        self.execution_engine = execution_engine
        self.evaluator = evaluator
        self.policy_checker = policy_checker
        self.audit_logger = audit_logger or AuditLogger()
        self.idempotent_tools = idempotent_tools or []
        
    async def process_request(self, request_id: str, request: str, user_identity: dict, user_approval_for_step: str = None) -> WorkflowState:
        state = WorkflowState(request_id=request_id, original_request=request)
        plan = self.planner.create_plan(request)
        state.plan = plan
        
        while state.current_step_index < len(state.plan):
            step = state.plan[state.current_step_index]
            
            # If we need approval
            if step.status == StepResult.WAITING_FOR_APPROVAL:
                if user_approval_for_step == step.step_id:
                    # Approved!
                    step.status = None
                else:
                    break # Still waiting
            
            capability = self.decision_engine.choose_capability(step)
            if not capability:
                state.failures.append(f"Could not choose capability for: {step.description}")
                step.status = StepResult.FAILED
                state.completed_steps.append(step)
                state.current_step_index += 1
                continue
                
            tool_name, tool_args = capability
            step.tool_name = tool_name
            step.tool_args = tool_args
            
            # Resume approval skipped check
            policy_decision = "ALLOW" if user_approval_for_step == step.step_id else self.policy_checker(user_identity, tool_name)
            
            if policy_decision == "DENY":
                state.decisions.append(f"Policy denied for {tool_name}")
                step.status = StepResult.BLOCKED
                step.result = f"Policy check failed for {tool_name}"
                state.completed_steps.append(step)
                
                role_str = user_identity.get("profile", {}).get("role", "unknown") if isinstance(user_identity, dict) else str(user_identity)
                self.audit_logger.log(
                    agent="Runtime", role=role_str, capability=step.description, 
                    server="local", tool=tool_name, policy_decision=policy_decision, 
                    execution_status="BLOCKED", duration_ms=0, result_metadata={"reason": "Policy denied"}
                )
                break
                
            if policy_decision == "REQUIRE_APPROVAL" and user_approval_for_step != step.step_id:
                state.decisions.append(f"Approval required for {tool_name}")
                step.status = StepResult.WAITING_FOR_APPROVAL
                step.result = "Waiting for user approval"
                break
            
            # Execution & Retry logic
            retries = 3 if tool_name in self.idempotent_tools else 0
            current_try = 0
            result_str = ""
            status = StepResult.FAILED
            exec_metadata = {}
            
            start_time = time.time()
            
            while current_try <= retries:
                is_retry = current_try > 0
                result_str = await self.execution_engine.execute(tool_name, tool_args)
                status = self.evaluator.evaluate(step, result_str)
                exec_metadata = {"result_str": result_str}
                
                if status != StepResult.FAILED:
                    break
                    
                # Determine if operational failure (simple string matching for mock)
                if any(err in result_str.lower() for err in ["timeout", "connection", "temporary"]):
                    current_try += 1
                    continue
                else:
                    # Non-retriable failure (e.g., auth, invalid args, scope violation)
                    break
                    
            duration_ms = (time.time() - start_time) * 1000
            
            is_fallback = False
            if status == StepResult.FAILED and current_try > retries:
                # Need to fallback if possible
                if state.current_step_index + 1 < len(state.plan) and "fallback" in state.plan[state.current_step_index + 1].description.lower():
                    is_fallback = True
            
            audit_role = user_identity.get("profile", {}).get("role", "unknown") if isinstance(user_identity, dict) else str(user_identity)
            
            self.audit_logger.log(
                agent="Runtime", role=audit_role, capability=step.description, 
                server="local", tool=tool_name, policy_decision=policy_decision, 
                execution_status=status.name, duration_ms=duration_ms, result_metadata=exec_metadata,
                is_retry=(current_try > 0), is_fallback=is_fallback
            )
            
            step.result = result_str
            step.status = status
            state.completed_steps.append(step)
            
            if status == StepResult.SUCCESS:
                state.current_step_index += 1
            elif status == StepResult.FAILED:
                state.failures.append(f"Tool {tool_name} failed: {result_str}")
                if is_fallback:
                     state.current_step_index += 1
                else:
                    break
            elif status == StepResult.INSUFFICIENT:
                if state.current_step_index + 1 < len(state.plan):
                    state.current_step_index += 1
                else:
                    break
            else:
                state.current_step_index += 1
                
        return state
