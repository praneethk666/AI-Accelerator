"""
tools.py — Registers the mail_send MCP tool into the server.

This file:
  - Defines what the AI agent sees (tool name, description, input schema)
  - Orchestrates auth → client → error handling
  - Imports NOTHING from google libraries directly (that stays in client.py)

The register() function is called by registry.py at server startup.

Design note: business logic lives in _mail_send_handler() so it can be
unit-tested directly without touching FastMCP's private internals.
"""

import logging
from mcp.server.mcpserver.server import MCPServer

from src.channels.mail.schemas import MailSendInput
from src.channels.mail.client import send_message
from src.channels.mail.auth import get_credentials
from src.common.errors import handle_gmail_error

logger = logging.getLogger(__name__)


async def _mail_send_handler(input: MailSendInput, channel_cfg) -> dict:
    """
    Core business logic for mail_send — fully testable without MCP plumbing.

    Separated from @mcp.tool() so unit tests can call this directly
    without depending on FastMCP private internals.
    """
    logger.info(
        f"mail_send invoked | to={input.to} | subject={input.subject!r}"
    )

    try:
        credentials = await get_credentials(channel_cfg)
        result = await send_message(
            credentials=credentials,
            to=input.to,
            subject=input.subject,
            body=input.body,
            cc=input.cc,
            bcc=input.bcc,
        )
        logger.info(f"mail_send success | message_id={result.get('id')}")
        return {
            "status": "sent",
            "message_id": result.get("id"),
            "recipients": input.to,
        }

    except RuntimeError as e:
        logger.warning(f"mail_send failed — auth not ready: {e}")
        return {
            "status": "failed",
            "error": "not_authenticated",
            "message": str(e),
            "action": "Visit http://localhost:8100/auth/gmail/start to authenticate Gmail.",
        }

    except Exception as e:
        return handle_gmail_error(e)


def register(mcp: MCPServer, channel_cfg) -> None:
    """
    Register all mail channel tools into the MCP server.
    Called once at startup by registry.py.
    """

    @mcp.tool()
    async def mail_send(input: MailSendInput) -> dict:
        """
        Send an email via Gmail to one or more recipients.

        Use this tool to send notifications, alerts, and status updates.

        For factory machine downtime alerts, include:
        - Machine name and ID
        - Location (floor, line)
        - Time the machine stopped
        - Error code (if available)
        - Urgency level

        Pass ALL relevant recipients in the 'to' list so everyone
        is notified in a single call.

        Returns:
          {"status": "sent", "message_id": "<gmail_id>"} on success
          {"status": "failed", "error": "<code>", "message": "<detail>"} on failure
        """
        return await _mail_send_handler(input, channel_cfg)
