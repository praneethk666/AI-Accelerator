"""
email_tool.py — Tool handler for send_email(to, subject, body).
Enforces:
1. Recipient allowlist validation
2. Prompt injection defense and security logging
3. Per-caller rate limiting
4. Server-side SMTP delivery with safe demo simulation mode
"""

import asyncio
from datetime import datetime, timezone
import email.utils
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import smtplib
from typing import Optional, Union
import uuid

from pydantic import BaseModel, EmailStr, Field, field_validator

from src.config import load_config
from src.security.allowlist import validate_email_allowlist
from src.security.injection_detector import detect_prompt_injection
from src.security.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class SendEmailInput(BaseModel):
    to: str = Field(
        ...,
        description="Recipient email address (e.g. 'ops@company.com', 'manager@company.com'). Must be in the authorized allowlist.",
    )
    subject: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Email subject line. Must not contain prompt injection sequences.",
    )
    body: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Email body text. Must not contain prompt injection sequences.",
    )

    @field_validator("to")
    @classmethod
    def validate_recipient_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Recipient 'to' field must not be empty.")
        return v.strip()


def _send_smtp_sync(
    host: str,
    port: int,
    username: str,
    password: str,
    sender: str,
    to: str,
    subject: str,
    body: str,
    use_tls: bool = True,
) -> str:
    """Synchronous SMTP sender run in executor."""
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg["Message-ID"] = email.utils.make_msgid(domain=host if host != "localhost" else "mcp.local")

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(host, port, timeout=10) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.sendmail(sender, [to], msg.as_string())

    return msg["Message-ID"]


async def send_email_handler(
    input_data: SendEmailInput,
    caller: Optional[str] = None,
) -> dict:
    """
    Executes email sending with strict security guardrails.
    """
    cfg = load_config()
    caller_id = caller or "anonymous_caller"

    logger.info(
        f"send_email invoked | caller={caller_id} | to={input_data.to} | subject={input_data.subject!r}"
    )

    # 1. Rate Limiting Check
    rate_cfg = cfg.security.email.rate_limit
    allowed, rate_err = rate_limiter.check_and_record(
        caller=caller_id,
        action="send_email",
        max_calls=rate_cfg.max_calls,
        window_seconds=rate_cfg.window_seconds,
    )
    if not allowed:
        return {
            "status": "failed",
            "error": "rate_limited",
            "message": rate_err,
        }

    # 2. Prompt Injection Guardrail
    if cfg.security.email.prompt_injection_guard.enabled:
        # Check subject
        is_inj_subj, subj_reason = detect_prompt_injection(
            text=input_data.subject, field_name="subject", caller=caller_id
        )
        if is_inj_subj:
            return {
                "status": "failed",
                "error": "prompt_injection_blocked",
                "message": f"Security violation: {subj_reason}",
            }

        # Check body
        is_inj_body, body_reason = detect_prompt_injection(
            text=input_data.body, field_name="body", caller=caller_id
        )
        if is_inj_body:
            return {
                "status": "failed",
                "error": "prompt_injection_blocked",
                "message": f"Security violation: {body_reason}",
            }

    # 3. Recipient Allowlist Check
    allowlist = cfg.security.email.allowlist
    is_allowed, allowlist_err = validate_email_allowlist(
        email=input_data.to, allowlist=allowlist, caller=caller_id
    )
    if not is_allowed:
        return {
            "status": "failed",
            "error": "recipient_not_allowed",
            "message": allowlist_err,
        }

    # 4. Email Delivery (SMTP or Safe Simulation)
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    smtp_cfg = cfg.smtp

    if smtp_cfg.mode.lower() == "smtp":
        try:
            loop = asyncio.get_event_loop()
            msg_id = await loop.run_in_executor(
                None,
                _send_smtp_sync,
                smtp_cfg.host,
                smtp_cfg.port,
                smtp_cfg.username,
                smtp_cfg.password,
                smtp_cfg.sender_address,
                input_data.to,
                input_data.subject,
                input_data.body,
                smtp_cfg.use_tls,
            )
            logger.info(f"Email sent via SMTP | message_id={msg_id} | to={input_data.to}")
            return {
                "status": "sent",
                "message_id": msg_id,
                "recipient": input_data.to,
                "subject": input_data.subject,
                "timestamp": timestamp_iso,
                "delivery_mode": "smtp",
            }
        except Exception as exc:
            logger.error(f"SMTP sending failed: {exc}", exc_info=True)
            return {
                "status": "failed",
                "error": "smtp_delivery_failed",
                "message": f"Failed to send email via SMTP server: {exc}",
            }
    else:
        # Simulation / Safe Demo Mode
        simulated_msg_id = f"sim-{uuid.uuid4().hex[:12]}@mcp.local"
        logger.info(
            f"Email dispatched in simulation mode | message_id={simulated_msg_id} | "
            f"caller={caller_id} | to={input_data.to} | subject={input_data.subject!r}"
        )
        return {
            "status": "sent",
            "message_id": simulated_msg_id,
            "recipient": input_data.to,
            "subject": input_data.subject,
            "timestamp": timestamp_iso,
            "delivery_mode": "simulation",
            "note": "Message processed and verified by security guardrails in demo simulation mode.",
        }
