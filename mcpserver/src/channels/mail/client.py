"""
client.py — Async wrapper over the Gmail API send operation.

IMPORTANT: This is the ONLY file in the entire project that imports
Google API types (googleapiclient, google.oauth2, etc.).
No other file outside channels/mail/ should import these directly.

The Google gmail API client is synchronous. We wrap it in asyncio's
run_in_executor() to keep the MCP server's event loop non-blocking.
"""

import asyncio
import base64
import logging
from email.mime.text import MIMEText
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)


# ── Internal sync helpers (run in thread pool) ────────────────────────────────

def _build_raw_message(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> dict:
    """
    Build a Gmail API message dict with base64-encoded MIME content.
    Plain text only (as per design decision).
    """
    msg = MIMEText(body, "plain", "utf-8")
    msg["to"] = ", ".join(to)
    msg["subject"] = subject
    if cc:
        msg["cc"] = ", ".join(cc)
    if bcc:
        msg["bcc"] = ", ".join(bcc)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"raw": raw}


def _sync_send_message(
    credentials: Credentials,
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]],
    bcc: Optional[list[str]],
) -> dict:
    """
    Synchronous Gmail API call.
    Runs in a thread pool via run_in_executor — never call directly from async code.
    Returns the Gmail API response dict (contains 'id', 'threadId', 'labelIds').
    """
    service = build("gmail", "v1", credentials=credentials)
    raw_message = _build_raw_message(to, subject, body, cc, bcc)
    result = (
        service.users()
        .messages()
        .send(userId="me", body=raw_message)
        .execute()
    )
    logger.info(
        f"Gmail message sent | id={result.get('id')} | to={to} | subject={subject!r}"
    )
    return result


# ── Public async interface ────────────────────────────────────────────────────

async def send_message(
    credentials: Credentials,
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> dict:
    """
    Send an email via Gmail API.

    Wraps the synchronous Google API client in asyncio's thread pool executor
    so the MCP event loop is never blocked while waiting for the network call.

    Args:
        credentials: Valid Google OAuth2 credentials (from auth.get_credentials)
        to:          List of recipient email addresses
        subject:     Email subject line
        body:        Plain text email body
        cc:          Optional list of CC addresses
        bcc:         Optional list of BCC addresses

    Returns:
        Gmail API response dict with 'id' and 'threadId' on success.

    Raises:
        googleapiclient.errors.HttpError on API errors (handled by tools.py)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Uses default ThreadPoolExecutor
        _sync_send_message,
        credentials,
        to,
        subject,
        body,
        cc,
        bcc,
    )
