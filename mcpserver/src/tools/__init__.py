"""
tools package — Mock tools for MCP server.
"""

from src.tools.datetime_tool import get_current_datetime_handler, GetCurrentDateTimeInput
from src.tools.email_tool import send_email_handler, SendEmailInput

__all__ = [
    "get_current_datetime_handler",
    "GetCurrentDateTimeInput",
    "send_email_handler",
    "SendEmailInput",
]
