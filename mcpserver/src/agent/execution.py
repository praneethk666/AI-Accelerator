import logging
from typing import Dict, Any, Callable, Awaitable

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, tool_executor: Callable[[str, Dict[str, Any]], Awaitable[str]]):
        self.tool_executor = tool_executor

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        try:
            return await self.tool_executor(tool_name, tool_args)
        except Exception as e:
            logger.error(f"Execution failed for {tool_name}: {e}")
            return f"ERROR: {str(e)}"
