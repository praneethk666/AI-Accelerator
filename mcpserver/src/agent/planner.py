import json
import logging
from typing import List, Protocol, Dict, Any
from .state import StepInfo

logger = logging.getLogger(__name__)

class LLMClientProtocol(Protocol):
    """
    Structural Type Definition (Protocol) for LLM Clients.
    Any class (like GroqClient or GeminiClient) that implements these matching 
    function signatures will automatically satisfy this type requirement natively 
    without needing formal inheritance.
    """
    def generate_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a prompt against the language model and strictly parses the output into a dictionary.
        The '...' symbol is intentional valid Python syntax for Protocol stubs.
        """
        ...

class Planner:
    def __init__(self, llm_client: LLMClientProtocol):
        self.llm_client = llm_client
        
    def create_plan(self, request: str) -> List[StepInfo]:
        prompt = f"""
You are an expert AI orchestrator. Break down the following user request into a sequence of logical steps.
Return ONLY valid JSON in the following format:
{{
    "steps": [
        {{"step_id": "1", "description": "Description of first action"}},
        {{"step_id": "2", "description": "Description of next action"}}
    ]
}}

User Request: {request}
"""
        try:
            response = self.llm_client.generate_json(prompt)
            steps_data = response.get("steps", [])
            
            plan = []
            for item in steps_data:
                plan.append(StepInfo(
                    step_id=str(item.get("step_id")), 
                    description=item.get("description")
                ))
            
            if not plan:
                logger.warning("LLM returned an empty plan, falling back to a single generic step.")
                return [StepInfo(step_id="1", description=request)]
                
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic plan: {e}")
            # Fallback for safety if LLM is down
            return [StepInfo(step_id="1", description=request)]
