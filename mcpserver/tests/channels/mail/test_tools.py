"""
test_tools.py — Unit tests for the mail_send MCP tool.

All Gmail API calls are mocked — no real network needed.

Strategy: test _mail_send_handler() directly (the extracted business logic
function) rather than going through FastMCP private internals. This is more
reliable across SDK versions and tests the actual logic, not the wiring.

Tests:
  1. Successful send → correct response structure
  2. Missing auth → actionable error with action URL
  3. Gmail rate limit (HTTP 429) → rate_limited error
  4. Missing required 'to' field → Pydantic ValidationError
  5. Missing required 'subject' field → Pydantic ValidationError
  6. Multiple recipients → all stored correctly
  7. Optional cc and bcc fields → stored correctly
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

# Import the module explicitly so patch() can resolve it
import src.channels.mail.tools  # noqa: F401 — required for patch target resolution
from mcp.server.mcpserver.server import MCPServer
from src.channels.mail.tools import _mail_send_handler
from src.channels.mail.schemas import MailSendInput


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_channel_cfg():
    """Minimal mail channel config for testing."""
    cfg = MagicMock()
    cfg.credentials_path = "credentials/google_credentials.json"
    cfg.token_path = "credentials/gmail_token.json"
    cfg.scopes = ["https://www.googleapis.com/auth/gmail.send"]
    return cfg


# ── Test 1: Successful send ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mail_send_success(mock_channel_cfg):
    """mail_send returns {status: sent, message_id, recipients} on success."""
    mock_credentials = MagicMock()
    mock_send_result = {"id": "msg_abc123", "threadId": "thread_xyz"}

    with patch(
        "src.channels.mail.tools.get_credentials",
        new=AsyncMock(return_value=mock_credentials),
    ), patch(
        "src.channels.mail.tools.send_message",
        new=AsyncMock(return_value=mock_send_result),
    ):
        result = await _mail_send_handler(
            MailSendInput(
                to=["manager@factory.com", "tech@factory.com"],
                subject="ALERT: Machine #3 stopped",
                body="Machine #3 on Floor B stopped at 18:02. Error: E-404.",
            ),
            mock_channel_cfg,
        )

    assert result["status"] == "sent"
    assert result["message_id"] == "msg_abc123"
    assert "manager@factory.com" in result["recipients"]
    assert "tech@factory.com" in result["recipients"]


# ── Test 2: Gmail not authenticated ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_mail_send_not_authenticated(mock_channel_cfg):
    """mail_send returns actionable error when OAuth not set up."""
    with patch(
        "src.channels.mail.tools.get_credentials",
        new=AsyncMock(
            side_effect=RuntimeError(
                "Gmail not authenticated. Visit http://localhost:8100/auth/gmail/start"
            )
        ),
    ):
        result = await _mail_send_handler(
            MailSendInput(
                to=["tech@factory.com"],
                subject="Alert",
                body="Test message",
            ),
            mock_channel_cfg,
        )

    assert result["status"] == "failed"
    assert result["error"] == "not_authenticated"
    assert "auth/gmail/start" in result["action"]


# ── Test 3: Gmail rate limit (HTTP 429) ───────────────────────────────────────

@pytest.mark.asyncio
async def test_mail_send_rate_limited(mock_channel_cfg):
    """mail_send returns rate_limited error on HTTP 429."""
    from googleapiclient.errors import HttpError

    mock_resp = MagicMock()
    mock_resp.status = 429
    mock_resp.reason = "Too Many Requests"

    with patch(
        "src.channels.mail.tools.get_credentials",
        new=AsyncMock(return_value=MagicMock()),
    ), patch(
        "src.channels.mail.tools.send_message",
        new=AsyncMock(side_effect=HttpError(mock_resp, b"Rate limit exceeded")),
    ):
        result = await _mail_send_handler(
            MailSendInput(
                to=["manager@factory.com"],
                subject="Alert",
                body="Test",
            ),
            mock_channel_cfg,
        )

    assert result["status"] == "failed"
    assert result["error"] == "rate_limited"
    assert "60 seconds" in result["message"]


# ── Test 4: Pydantic validation — missing 'to' field ─────────────────────────

def test_mail_send_missing_to_field():
    """MailSendInput raises ValidationError when 'to' is missing."""
    with pytest.raises(ValidationError) as exc_info:
        MailSendInput(
            subject="Missing recipients",
            body="This should fail",
        )
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("to",) for e in errors)


# ── Test 5: Pydantic validation — missing 'subject' field ────────────────────

def test_mail_send_missing_subject():
    """MailSendInput raises ValidationError when 'subject' is missing."""
    with pytest.raises(ValidationError) as exc_info:
        MailSendInput(
            to=["someone@example.com"],
            body="Missing subject",
        )
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("subject",) for e in errors)


# ── Test 6: Multiple recipients ───────────────────────────────────────────────

def test_mail_send_multiple_recipients():
    """MailSendInput accepts a list of multiple email addresses."""
    recipients = [
        "manager@factory.com",
        "technician@factory.com",
        "supervisor@factory.com",
    ]
    input_model = MailSendInput(
        to=recipients,
        subject="Multi-recipient test",
        body="This email goes to multiple people.",
    )
    assert len(input_model.to) == 3
    assert input_model.cc is None
    assert input_model.bcc is None


# ── Test 7: Optional cc and bcc ───────────────────────────────────────────────

def test_mail_send_with_cc_bcc():
    """MailSendInput correctly stores optional cc and bcc lists."""
    input_model = MailSendInput(
        to=["primary@factory.com"],
        subject="Test with CC/BCC",
        body="Testing optional fields.",
        cc=["safety@factory.com"],
        bcc=["audit@factory.com"],
    )
    assert input_model.cc == ["safety@factory.com"]
    assert input_model.bcc == ["audit@factory.com"]


