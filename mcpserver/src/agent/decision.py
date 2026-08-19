import json
import logging
from typing import Optional, Dict, Any, Tuple, Protocol, List
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

class DecisionEngine:
    def __init__(self, llm_client: LLMClientProtocol, available_tools: List[Dict[str, Any]]):
        self.llm_client = llm_client
        self.available_tools = available_tools

    def choose_capability(self, step: StepInfo) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Format the tools with full schema so the LLM knows exactly which keys to extract
        tools_list = json.dumps(self.available_tools, indent=2)
        
        prompt = f"""
You are an AI decision engine routing tasks to tools.
Available tools and their full JSON schemas:
{tools_list}

Select the SINGLE best tool to accomplish the following task.
Extract any necessary arguments from the task description exactly following the REQUIRED parameters of the tool schema. 
For example, if the tool requires "to", do not use "recipient".
You must emit valid JSON ONLY.

Format:
{{
    "tool_name": "the_tool",
    "arguments": {{"arg1": "value1"}}
}}

Task description: {step.description}
"""
        try:
            response = self.llm_client.generate_json(prompt)
            tool_name = response.get("tool_name")
            tool_args = response.get("arguments", {})
            
            if not tool_name or tool_name == "null":
                logger.warning(f"LLM determined no tool could fulfill: {step.description}")
                return None
                
            return tool_name, tool_args
            
        except Exception as e:
            logger.error(f"Failed to choose capability dynamically: {e}")
            return None
