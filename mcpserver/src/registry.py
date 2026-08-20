"""
registry.py — Unified Master Tool Registry for MCP Server (Port 8100).
Registers all 7 tools across Gmail and Time/System domains and supports
dynamic remote tool registration via ServerRegistry.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from src.servers.gmail_server import (
    GMAIL_TOOLS,
    SendEmailInput,
    ReadInboxInput,
    SearchEmailsInput,
    CreateDraftInput,
    handle_send_email,
    handle_read_inbox,
    handle_search_emails,
    handle_create_draft,
)
from src.servers.time_server import (
    TIME_TOOLS,
    GetCurrentDateTimeInput,
    ConvertTimezoneInput,
    GetSystemUptimeInput,
    handle_get_current_datetime,
    handle_convert_timezone,
    handle_get_system_uptime,
)

logger = logging.getLogger(__name__)

# Master list of all tools available on the Unified MCP Gateway (Port 8100)
TOOL_DEFINITIONS = GMAIL_TOOLS + TIME_TOOLS


class ToolRegistry:
    def __init__(self):
        # Tools dynamically registered or initially built-in.
        self.tool_definitions: List[Dict[str, Any]] = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
                "_local_handler": t["handler"],
                "_model_cls": t["model_cls"],
                "_is_local": True
            }
            for t in TOOL_DEFINITIONS
        ]
        self._server_registry = None

    def set_server_registry(self, registry):
        self._server_registry = registry

    def register_remote_tools(self, server_name: str, tools: List[Dict[str, Any]]):
        """Registers a list of tools dynamically discovered from a remote server."""
        for tool in tools:
            tool["_is_local"] = False
            tool["_remote_server"] = server_name
            self.tool_definitions.append(tool)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard tool manifests for MCP tools/list requests."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self.tool_definitions
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], caller: Optional[str] = None, correlation_id: Optional[str] = None) -> dict:
        """Executes the requested tool by name, locally or remotely."""
        for tool_meta in self.tool_definitions:
            if tool_meta["name"] == tool_name:
                if tool_meta.get("_is_local"):
                    # Local execution
                    model_cls = tool_meta["_model_cls"]
                    handler = tool_meta["_local_handler"]
                    validated_input = model_cls(**(arguments or {}))
                    # Attempt to pass correlation_id if handler accepts it, else don't
                    import inspect
                    sig = inspect.signature(handler)
                    kwargs = {"caller": caller}
                    if "correlation_id" in sig.parameters:
                        kwargs["correlation_id"] = correlation_id
                    return await handler(validated_input, **kwargs)
                else:
                    # Remote execution
                    server_name = tool_meta.get("_remote_server")
                    if self._server_registry:
                        client = self._server_registry.get_client(server_name)
                        if client:
                            logger.info(f"Proxying tool {tool_name} to server {server_name}")
                            return await client.call_tool(tool_name, arguments)
                        else:
                            raise RuntimeError(f"Client for server '{server_name}' not found.")
                    else:
                        raise RuntimeError(f"ServerRegistry not initialized for remote tool execution.")

        raise KeyError(f"Tool '{tool_name}' not found.")


global_tool_registry = ToolRegistry()


def get_tool_definitions() -> List[Dict[str, Any]]:
    return global_tool_registry.get_tool_definitions()


async def execute_tool(tool_name: str, arguments: Dict[str, Any], caller: Optional[str] = None, correlation_id: Optional[str] = None) -> dict:
    return await global_tool_registry.execute_tool(tool_name, arguments, caller=caller, correlation_id=correlation_id)
