"""
retrieval_guard.py — Checkpoint 2: scan tool outputs before returning to agent.

Traverses tool output recursively. If any string is found containing injection,
redacts it to "[CONTENT REDACTED: injection detected]" and logs a WARN/BLOCK event.

Features:
  - Byte-safe 100KB truncation before regex scanning (prevents backtracking CPU hangs).
  - Normalization of scanned copy.
  - Sticky canary rollout check.
  - Async wrapper to offload regex processing to a background thread pool.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.guardrails.guard_decision import GuardDecision, PolicyDecision, GuardEvidence
from backend.guardrails.normalizer import normalize
from backend.guardrails.input_guard import _INJECTION_PATTERNS
from backend.guardrails.safe_wrapper import guardrail_safe
from backend.guardrails.rollout import should_apply_guard

logger = logging.getLogger(__name__)

_DEFAULT_SCAN_MAX_BYTES = 100_000


def _safe_truncate(s: str, max_bytes: int) -> str:
    """Truncate string to max_bytes, without splitting multi-byte characters."""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", "ignore")


def _score_string(s: str) -> int:
    """Check normalized string for injection patterns. Returns 100 if found, else 0."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(s):
            return 100
    return 0


def _scan_recursive(obj: Any, max_bytes: int, depth: int = 0, max_depth: int = 8) -> tuple[Any, list[str]]:
    """
    Recursively scan structures. Replaces injected text with redaction string.
    Returns (cleaned_obj, list_of_fired_rules).
    """
    if depth > max_depth:
        return obj, []

    if isinstance(obj, str):
        safe_str = _safe_truncate(obj, max_bytes)
        normalized = normalize(safe_str)
        score = _score_string(normalized)
        if score >= 100:
            return "[CONTENT REDACTED: injection detected]", ["chunk_injection_found"]
        return obj, []

    if isinstance(obj, dict):
        cleaned_dict = {}
        fired = []
        for k, v in obj.items():
            # Also scan key names if they are user-controlled (e.g. metadata keys)
            k_clean, k_fired = _scan_recursive(k, max_bytes, depth + 1, max_depth)
            v_clean, v_fired = _scan_recursive(v, max_bytes, depth + 1, max_depth)
            cleaned_dict[k_clean] = v_clean
            fired.extend(k_fired)
            fired.extend(v_fired)
        return cleaned_dict, list(set(fired))

    if isinstance(obj, list):
        cleaned_list = []
        fired = []
        for item in obj:
            item_clean, item_fired = _scan_recursive(item, max_bytes, depth + 1, max_depth)
            cleaned_list.append(item_clean)
            fired.extend(item_fired)
        return cleaned_list, list(set(fired))

    return obj, []


@guardrail_safe("retrieval")
def scan_tool_output(
    result: Any,
    *,
    config: dict | None = None,
    session_id: str = "",
    version: str = "1.0.0",
) -> GuardDecision:
    """
    Scans a tool's output for injection attempts.
    Returns GuardDecision containing the cleaned object in sanitized_value.
    """
    t0 = time.perf_counter()
    g_cfg = (config or {}).get("guardrails", {})
    ret_cfg = g_cfg.get("retrieval", {})

    # Check rollout percentage
    pct = int(g_cfg.get("rollout", {}).get("retrieval_guard_pct", 50))
    if not should_apply_guard(pct, session_id):
        # Canary gate skipped
        return GuardDecision(
            allowed=True,
            policy=PolicyDecision.ALLOW,
            stage="retrieval",
            event_type="retrieval_rollout_skip",
            sanitized_value=result,
            latency_ms=(time.perf_counter() - t0) * 1000,
            guardrail_version=version,
        )

    if not ret_cfg.get("chunk_injection_scan", True):
        # Scan disabled in config
        return GuardDecision(
            allowed=True,
            policy=PolicyDecision.ALLOW,
            stage="retrieval",
            event_type="retrieval_scan_disabled",
            sanitized_value=result,
            latency_ms=(time.perf_counter() - t0) * 1000,
            guardrail_version=version,
        )

    max_bytes = int(ret_cfg.get("scan_max_bytes", _DEFAULT_SCAN_MAX_BYTES))
    cleaned_result, fired = _scan_recursive(result, max_bytes)
    latency = (time.perf_counter() - t0) * 1000

    if fired:
        logger.warning("retrieval_guard: injection found in tool output, redacting.")
        return GuardDecision(
            allowed=True, # retrieval guard fails open (WARN/REDACT), doesn't block agent node
            policy=PolicyDecision.REDACT,
            stage="retrieval",
            event_type="retrieval_injection_redacted",
            risk_score=35,
            reason="Injection detected in tool output chunk",
            sanitized_value=cleaned_result,
            latency_ms=latency,
            guardrail_version=version,
            rule_id="chunk_injection_found",
        )

    return GuardDecision(
        allowed=True,
        policy=PolicyDecision.ALLOW,
        stage="retrieval",
        event_type="retrieval_ok",
        risk_score=0,
        sanitized_value=cleaned_result,
        latency_ms=latency,
        guardrail_version=version,
    )


async def scan_tool_output_async(
    result: Any,
    config: dict | None = None,
    session_id: str = "",
    version: str = "1.0.0",
) -> GuardDecision:
    """Async wrapper to run scan_tool_output in a background thread."""
    return await asyncio.to_thread(
        scan_tool_output,
        result,
        config=config,
        session_id=session_id,
        version=version
    )
