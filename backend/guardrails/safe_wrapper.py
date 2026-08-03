"""
safe_wrapper.py — @guardrail_safe decorator.

Wraps any guardrail check so that:
  - Exceptions are NEVER propagated to the caller.
  - Fail mode is determined by stage (OPEN / SAFE_REPLY) — NOT a global flag.
  - Bypass events are logged to the ring buffer immediately (no DB needed).

Usage:
    @guardrail_safe("input")
    def check_input(text: str, ...) -> GuardDecision:
        ...
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Optional

from backend.guardrails.guard_decision import (
    FailMode, GuardDecision, PolicyDecision, SAFE_REPLY_MESSAGE, STAGE_FAIL_MODES,
)

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency — ring_buffer imports nothing from here.
def _ring_buffer():
    from backend.guardrails.ring_buffer import add_event
    return add_event


def guardrail_safe(stage: str, version: str = "1.0.0"):
    """
    Decorator factory. Wraps a guardrail check function.

    On exception:
      - input/retrieval (OPEN):   returns allowed=True, bypassed=True
      - output (SAFE_REPLY):      returns allowed=False, policy=BLOCK, bypassed=True
                                  safe_answer is SAFE_REPLY_MESSAGE, NEVER raw content

    On normal return: returns the GuardDecision as-is.
    """
    fail_mode = STAGE_FAIL_MODES.get(stage, FailMode.OPEN)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> GuardDecision:
            t0 = time.perf_counter()
            try:
                result: GuardDecision = fn(*args, **kwargs)
                result.latency_ms = (time.perf_counter() - t0) * 1000
                return result
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000
                logger.warning(
                    "guardrail_safe[%s] caught exception in %s: %s — applying %s",
                    stage, fn.__name__, exc, fail_mode.value,
                )
                if fail_mode == FailMode.SAFE_REPLY:
                    decision = GuardDecision(
                        allowed=False,
                        policy=PolicyDecision.BLOCK,
                        stage=stage,
                        event_type="guard_crash_safe_reply",
                        risk_score=100,
                        reason=f"Guard crashed: {type(exc).__name__}",
                        sanitized_value=SAFE_REPLY_MESSAGE,
                        latency_ms=latency,
                        guardrail_version=version,
                        bypassed=True,
                        hard_block=True,
                    )
                else:
                    # OPEN — allow through, log bypass
                    decision = GuardDecision(
                        allowed=True,
                        policy=PolicyDecision.ALLOW,
                        stage=stage,
                        event_type="guard_crash_bypass",
                        risk_score=0,
                        reason=f"Guard crashed (fail-open): {type(exc).__name__}",
                        latency_ms=latency,
                        guardrail_version=version,
                        bypassed=True,
                    )
                try:
                    _ring_buffer()(decision, session_id="")
                except Exception:
                    pass   # ring buffer itself must not crash anything
                return decision
        return wrapper
    return decorator
