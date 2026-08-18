"""
registry.py — Tool registration for MCP server.
Registers get_current_datetime and send_email tools with schemas and handlers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from src.tools.datetime_tool import (
    GetCurrentDateTimeInput,
    get_current_datetime_handler,
)
from src.tools.email_tool import (
    SendEmailInput,
    send_email_handler,
)

logger = logging.getLogger(__name__)

# Registry of MCP Tools metadata and handlers
TOOL_DEFINITIONS = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current host system date and time with optional IANA timezone conversion (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone name (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata', 'Europe/London'). Defaults to system local time if omitted.",
                }
            },
            "required": [],
        },
        "model_cls": GetCurrentDateTimeInput,
        "handler": get_current_datetime_handler,
    },
    {
        "name": "send_email",
        "description": "Sends an email notification via server SMTP with strict security guardrails (recipient allowlist checking, per-caller rate limits, and prompt injection defense).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address (e.g. 'ops@company.com', 'manager@company.com'). Must be in the authorized allowlist.",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line of the email. Must not contain prompt injection sequences.",
                },
                "body": {
                    "type": "string",
                    "description": "Body content of the email. Must not contain prompt injection sequences.",
                },
            },
            "required": ["to", "subject", "body"],
        },
        "model_cls": SendEmailInput,
        "handler": send_email_handler,
    },
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Returns tool manifests for MCP tools/list requests."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["inputSchema"],
        }
        for t in TOOL_DEFINITIONS
    ]


async def execute_tool(tool_name: str, arguments: Dict[str, Any], caller: Optional[str] = None) -> dict:
    """Executes the requested tool by name with validated arguments."""
    for tool_meta in TOOL_DEFINITIONS:
        if tool_meta["name"] == tool_name:
            model_cls = tool_meta["model_cls"]
            handler = tool_meta["handler"]
            # Validate input using Pydantic model
            validated_input = model_cls(**(arguments or {}))
            return await handler(validated_input, caller=caller)

    raise KeyError(f"Tool '{tool_name}' not found.")


def register_all(mcp_server: Any) -> None:
    """
    If using FastMCP or MCPServer instance, registers all tools directly.
    """
    if hasattr(mcp_server, "tool"):
        @mcp_server.tool(name="get_current_datetime", description="Returns host system date and time with optional IANA timezone conversion.")
        async def get_current_datetime(timezone: Optional[str] = None) -> dict:
            return await get_current_datetime_handler(GetCurrentDateTimeInput(timezone=timezone))

        @mcp_server.tool(name="send_email", description="Sends an email notification via server SMTP with strict security guardrails.")
        async def send_email(to: str, subject: str, body: str) -> dict:
            return await send_email_handler(SendEmailInput(to=to, subject=subject, body=body))

        logger.info("Registered tools with MCP Server instance.")
