"""
event_logger.py — async fire-and-forget guardrail event logger.

Writes to two targets:
  1. ring_buffer   — always, immediately (in-memory, for OTel metrics and bypass alerts)
  2. Postgres       — async via threading (non-blocking; failure → ring buffer only)

Never blocks the request path. Never raises.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING

from backend.guardrails.ring_buffer import add_event as rb_add
from backend.core.tracing import record_guardrail_decision

if TYPE_CHECKING:
    from backend.guardrails.guard_decision import GuardDecision

logger = logging.getLogger(__name__)

# Event types that also go to the security_audit_log table
_SECURITY_EVENT_TYPES = {
    "injection_blocked",
    "session_cumulative_block",
    "guard_crash_safe_reply",
    "query_too_long",
}


def log_event(decision: "GuardDecision", session_id: str = "", config: dict | None = None) -> None:
    """
    Log a guardrail decision. Call this from LangGraph nodes.
    Non-blocking: Postgres writes happen in a background thread.
    """
    # 1. Ring buffer (always, synchronously — never fails)
    rb_add(decision, session_id)
    try:
        record_guardrail_decision(decision)
    except Exception:
        pass

    # 2. Postgres (background thread — fire-and-forget)
    if (config or {}).get("guardrails", {}).get("logging", {}).get("enabled", True):
        t = threading.Thread(
            target=_write_postgres,
            args=(decision, session_id, config),
            daemon=True,
        )
        t.start()


def _write_postgres(decision: "GuardDecision", session_id: str, config: dict | None) -> None:
    """Write to guardrail_events (and security_audit_log if applicable). Never raises."""
    try:
        from backend.storage.postgres_store import dsn_from_env
        import psycopg2

        db_url = dsn_from_env()
        if not db_url:
            return

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Operational log — all events
        cur.execute(
            """
            INSERT INTO guardrail_events
              (session_id, stage, event_type, policy, risk_score,
               allowed, bypassed, hard_block, rule_id, guardrail_version,
               latency_ms, detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                session_id,
                decision.stage,
                decision.event_type,
                decision.policy.value if hasattr(decision.policy, "value") else str(decision.policy),
                decision.risk_score,
                decision.allowed,
                decision.bypassed,
                decision.hard_block,
                decision.rule_id,
                decision.guardrail_version,
                decision.latency_ms,
                json.dumps({"reason": decision.reason}),
            ),
        )

        # Security audit log — BLOCK / injection events only
        if decision.event_type in _SECURITY_EVENT_TYPES:
            cur.execute(
                """
                INSERT INTO security_audit_log
                  (session_id, event_type, risk_score, rule_id,
                   guardrail_version, detail)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (
                    session_id,
                    decision.event_type,
                    decision.risk_score,
                    decision.rule_id,
                    decision.guardrail_version,
                    json.dumps({"reason": decision.reason}),
                ),
            )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as exc:
        # Postgres write failure must never affect the request — just log it.
        logger.debug("guardrail event_logger postgres write failed: %s", exc)
