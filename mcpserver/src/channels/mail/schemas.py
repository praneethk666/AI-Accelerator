"""
schemas.py — Pydantic input model for the mail_send MCP tool.

The MCP SDK auto-generates the JSON Schema (shown to the AI agent) from this model.
Field descriptions are critical — they are what the AI agent reads to know
how to call the tool correctly.
"""

from pydantic import BaseModel, Field
from typing import Optional


class MailSendInput(BaseModel):
    """Input schema for the mail_send tool."""

    to: list[str] = Field(
        description=(
            "List of recipient email addresses. "
            "Example: ['manager@factory.com', 'technician@factory.com']. "
            "All addresses in this list receive the email."
        )
    )
    subject: str = Field(
        description=(
            "Email subject line. "
            "For machine alerts, include the machine name and event type. "
            "Example: 'ALERT: Machine #3 stopped at 18:02'"
        )
    )
    body: str = Field(
        description=(
            "Email body as plain text. "
            "Include all details the recipient needs: machine name, location, "
            "stop time, error code, and any relevant context. "
            "Example: 'Machine #3 on Floor B stopped at 18:02.\\nError code: E-404.\\nPlease investigate immediately.'"
        )
    )
    cc: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of CC recipient email addresses. "
            "These recipients receive a copy but are not the primary addressees."
        )
    )
    bcc: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of BCC recipient email addresses. "
            "These recipients receive a hidden copy — other recipients cannot see them."
        )
    )
