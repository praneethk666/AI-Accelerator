"""
ring_buffer.py — thread-safe, fixed-size in-memory event buffer.

Stores the last N guardrail decisions in memory. Used by:
  - Bypass rate observable gauge (OTel) — reads last 5 minutes of events
  - Fallback when Postgres is unavailable (events queued here, not lost)

Does NOT block the request path. Never raises.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.guardrails.guard_decision import GuardDecision

_BUFFER_SIZE = 500

_lock = threading.Lock()
_events: deque[tuple[float, "GuardDecision", str]] = deque(maxlen=_BUFFER_SIZE)
# Each entry: (timestamp, GuardDecision, session_id)


def add_event(decision: "GuardDecision", session_id: str) -> None:
    """Add a decision to the ring buffer. Thread-safe. Never raises."""
    try:
        with _lock:
            _events.append((time.monotonic(), decision, session_id))
    except Exception:
        pass


def get_recent(minutes: int = 5) -> list[tuple[float, "GuardDecision", str]]:
    """Return events from the last *minutes* minutes. Thread-safe. Never raises."""
    try:
        cutoff = time.monotonic() - minutes * 60
        with _lock:
            return [(ts, d, sid) for ts, d, sid in _events if ts >= cutoff]
    except Exception:
        return []


def get_bypass_rate(minutes: int = 5) -> float:
    """Fraction of recent events where bypassed=True. Used by OTel gauge callback."""
    try:
        recent = get_recent(minutes)
        if not recent:
            return 0.0
        bypassed = sum(1 for _, d, _ in recent if d.bypassed)
        return bypassed / len(recent)
    except Exception:
        return 0.0


def drain_all() -> list[tuple[float, "GuardDecision", str]]:
    """Pop all events (used for async Postgres flush). Thread-safe."""
    try:
        with _lock:
            items = list(_events)
            _events.clear()
            return items
    except Exception:
        return []
