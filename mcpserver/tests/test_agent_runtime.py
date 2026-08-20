import pytest
from src.agent.state import StepResult
from src.agent.planner import Planner
from src.agent.decision import DecisionEngine
from src.agent.execution import ExecutionEngine
from src.agent.evaluator import ResultEvaluator
from src.agent.runtime import AgentRuntime
from src.security.audit import AuditLogger

def mock_policy_checker(role: str, tool: str) -> str:
    if role == "unauthorized":
        return "DENY"
    if role == "denied_tool_user" and tool == "denied_tool":
        return "DENY"
    if tool == "require_approval_tool":
        return "REQUIRE_APPROVAL"
    return "ALLOW"

class MockExecutor:
    def __init__(self):
        self.call_counts = {}

    async def execute(self, tool_name: str, args: dict) -> str:
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        
        if tool_name == "failing_tool":
            return "ERROR: invalid arguments"
        if tool_name == "timeout_tool":
            return "ERROR: connection timeout"
        if tool_name == "insufficient_tool":
            return "INSUFFICIENT information"
        if tool_name == "denied_tool":
            return "This shouldn't be executed"
        if tool_name == "require_approval_tool":
            return "Executed require_approval_tool successfully"
        return f"Executed {tool_name} successfully"

@pytest.fixture
def audit_logger():
    return AuditLogger()

@pytest.fixture
def mock_executor():
    return MockExecutor()

class MockLLMClient:
    def generate_json(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        if "policy denied" in prompt_lower:
            # Planner prompt
            if "description of first action" in prompt_lower: # basic check if it is planner prompt
                return {"steps": [{"step_id": "1", "description": "Call denied tool"}]}
            return {"tool_name": "denied_tool", "arguments": {}}
            
        if "denied tool" in prompt_lower:
            return {"tool_name": "denied_tool", "arguments": {}}
        
        if "do timeout" in prompt_lower:
            if "description of first action" in prompt_lower:
                return {"steps": [{"step_id": "1", "description": "do timeout"}]}
            return {"tool_name": "timeout_tool", "arguments": {}}
            
        if "run approval task" in prompt_lower:
            if "description of first action" in prompt_lower:
                return {"steps": [{"step_id": "1", "description": "run approval task"}]}
            return {"tool_name": "require_approval_tool", "arguments": {}}
            
        # Default fallback
        if "description of first action" in prompt_lower:
             return {"steps": [{"step_id": "1", "description": "execute some task"}]}
        return {"tool_name": "simple_tool", "arguments": {}}

@pytest.fixture
def mock_llm_client():
    return MockLLMClient()

@pytest.fixture
def agent_runtime(audit_logger, mock_executor, mock_llm_client):
    return AgentRuntime(
        planner=Planner(llm_client=mock_llm_client),
        decision_engine=DecisionEngine(llm_client=mock_llm_client, available_tools=[]),
        execution_engine=ExecutionEngine(mock_executor.execute),
        evaluator=ResultEvaluator(),
        policy_checker=mock_policy_checker,
        audit_logger=audit_logger,
        idempotent_tools=["timeout_tool"]
    )

@pytest.mark.asyncio
async def test_simple_request(agent_runtime):
    state = await agent_runtime.process_request("1", "execute some task", "admin")
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].status == StepResult.SUCCESS

@pytest.mark.asyncio
async def test_policy_denied(agent_runtime, audit_logger):
    state = await agent_runtime.process_request("2", "policy denied", "denied_tool_user")
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].status == StepResult.BLOCKED
    assert "Policy denied for denied_tool" in state.decisions
    # Check audit log hash chain mutated
    assert audit_logger._last_hash != "0" * 64

@pytest.mark.asyncio
async def test_retry_for_timeout(agent_runtime, mock_executor):
    # 'timeout_tool' returns "ERROR: connection timeout", which triggers retry.
    # It is marked as idempotent, so it will retry up to 3 times (total 4 calls).
    # Since we didn't add it to Planner/DecisionEngine in tests, let's patch DecisionEngine for the test
    original_choose = agent_runtime.decision_engine.choose_capability
    agent_runtime.decision_engine.choose_capability = lambda step: ("timeout_tool", {}) if "timeout" in step.description.lower() else original_choose(step)
    
    state = await agent_runtime.process_request("3", "do timeout", "admin")
    assert mock_executor.call_counts.get("timeout_tool", 0) == 4
    assert state.completed_steps[0].status == StepResult.FAILED

@pytest.mark.asyncio
async def test_no_retry_for_non_idempotent(agent_runtime, mock_executor):
    # Similar to above, but a tool not in idempotent_tools
    original_choose = agent_runtime.decision_engine.choose_capability
    agent_runtime.decision_engine.choose_capability = lambda step: ("non_idempotent_timeout", {}) if "timeout" in step.description.lower() else original_choose(step)
    
    # Needs to return timeout error
    original_exec = agent_runtime.execution_engine.execute
    async def mock_execute(name, args):
        if name == "non_idempotent_timeout":
            return "ERROR: connection timeout"
        return await original_exec(name, args)
    
    agent_runtime.execution_engine.execute = mock_execute
    
    state = await agent_runtime.process_request("4", "do timeout", "admin")
    # Only 1 call because not idempotent
    assert state.completed_steps[0].status == StepResult.FAILED

@pytest.mark.asyncio
async def test_approval_flow(agent_runtime, mock_executor):
    original_choose = agent_runtime.decision_engine.choose_capability
    agent_runtime.decision_engine.choose_capability = lambda step: ("require_approval_tool", {}) if "approval" in step.description.lower() else original_choose(step)
    
    # 1. First run, it should pause for approval
    state1 = await agent_runtime.process_request("5", "run approval task", "admin")
    assert state1.plan[0].status == StepResult.WAITING_FOR_APPROVAL
    assert "require_approval_tool" not in mock_executor.call_counts
    
    # 2. Resuming with approval
    state2 = await agent_runtime.process_request("5", "run approval task", "admin", user_approval_for_step=state1.plan[0].step_id)
    assert mock_executor.call_counts.get("require_approval_tool", 0) == 1
    assert state2.completed_steps[0].status == StepResult.SUCCESS
