"""
output_guard.py — Checkpoint 3: PII masking on AI answer before DB write.

Critical guarantees:
  1. PII masking runs BEFORE safe_answer is set — the DB never sees raw PII.
  2. Fail mode = SAFE_REPLY — if masking crashes, return generic safe message.
     NEVER pass raw content through on output-stage failure.
  3. The redacted AIMessage MUST carry the same `id` as the original to trigger
     LangGraph's add_messages overwrite (not append).
     Without same-id: raw PII message stays in state and gets checkpointed.

PII redaction order: GSTIN → PAN → Aadhaar → Credit card → Email → UPI → Phone
"""
from __future__ import annotations

import logging
import re
import time

from backend.guardrails.guard_decision import GuardDecision, PolicyDecision, SAFE_REPLY_MESSAGE
from backend.guardrails.safe_wrapper import guardrail_safe

logger = logging.getLogger(__name__)

# ── Same PII patterns as input_guard — kept in sync manually ─────────────────
# (imported via input_guard to avoid duplication)
from backend.guardrails.input_guard import (
    _GSTIN, _PAN, _AADHAAR, _CREDIT_CARD, _EMAIL, _UPI, _PHONE,
)

_PII_RULES = [
    (_GSTIN,       "GSTIN"),
    (_PAN,         "PAN"),
    (_AADHAAR,     "AADHAAR"),
    (_CREDIT_CARD, "CARD"),
    (_EMAIL,       "EMAIL"),
    (_UPI,         "UPI"),
    (_PHONE,       "PHONE"),
]


def _mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Apply all PII masks in detection order.
    Returns (masked_text, list_of_types_found).
    """
    found: list[str] = []
    for pattern, label in _PII_RULES:
        new, n = pattern.subn(f"[{label} REDACTED]", text)
        if n:
            found.append(label)
            text = new
    return text, found


@guardrail_safe("output")
def mask_output(
    answer: str,
    *,
    config: dict | None = None,
    version: str = "1.0.0",
) -> GuardDecision:
    """
    Mask PII in the AI's answer.

    Returns GuardDecision with sanitized_value = masked answer.
    On any failure, @guardrail_safe("output") returns SAFE_REPLY automatically.
    """
    t0 = time.perf_counter()
    g_cfg = (config or {}).get("guardrails", {})
    out_cfg = g_cfg.get("output", {})

    if not out_cfg.get("pii_mask", True):
        # PII masking disabled in config
        return GuardDecision(
            allowed=True,
            policy=PolicyDecision.ALLOW,
            stage="output",
            event_type="output_ok",
            risk_score=0,
            sanitized_value=answer,
            latency_ms=(time.perf_counter() - t0) * 1000,
            guardrail_version=version,
        )

    masked, found = _mask_pii(answer)
    latency = (time.perf_counter() - t0) * 1000

    if found:
        logger.info("output_guard: masked PII types=%s", found)
        return GuardDecision(
            allowed=True,
            policy=PolicyDecision.REDACT,
            stage="output",
            event_type="pii_masked",
            risk_score=30,
            reason=f"PII types masked: {found}",
            sanitized_value=masked,
            latency_ms=latency,
            guardrail_version=version,
        )

    return GuardDecision(
        allowed=True,
        policy=PolicyDecision.ALLOW,
        stage="output",
        event_type="output_ok",
        risk_score=0,
        sanitized_value=masked,
        latency_ms=latency,
        guardrail_version=version,
    )
