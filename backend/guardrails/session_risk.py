"""
session_risk.py — multi-turn rolling risk accumulator.

Tracks cumulative risk score across multiple turns within one session.
Detects multi-turn injection probing: each turn is individually low-score
(WARN) but together they exceed the session block threshold.

Deployment note:
  Single Uvicorn worker (current setup): in-process deque is correct.
  Multi-worker: Redis required. Set guardrails.session_risk.redis_url in config.
  Failure to set Redis in multi-worker = silent fragmentation (each worker sees
  only its own turns). This is an accepted residual risk for single-worker deploys.

Thread-safe. Never raises.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_MINUTES = 30
_DEFAULT_BLOCK_THRESHOLD = 150


class SessionRiskAccumulator:
    def __init__(
        self,
        window_minutes: int = _DEFAULT_WINDOW_MINUTES,
        session_block_threshold: int = _DEFAULT_BLOCK_THRESHOLD,
    ):
        self._lock = threading.Lock()
        self._sessions: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._window_sec = window_minutes * 60
        self._threshold = session_block_threshold

    def add_and_check(self, session_id: str, score: int) -> tuple[bool, int]:
        """
        Record *score* for *session_id* and check cumulative total.
        Returns (should_block: bool, cumulative_score: int).
        Never raises.
        """
        if not session_id:
            return False, score
        try:
            now = time.monotonic()
            with self._lock:
                q = self._sessions[session_id]
                # Expire old entries outside rolling window
                while q and (now - q[0][0]) > self._window_sec:
                    q.popleft()
                q.append((now, score))
                cumulative = sum(s for _, s in q)
            return cumulative >= self._threshold, cumulative
        except Exception as exc:
            logger.warning("SessionRiskAccumulator.add_and_check error: %s", exc)
            return False, score

    def get_cumulative(self, session_id: str) -> int:
        """Current cumulative score for a session (within rolling window)."""
        try:
            now = time.monotonic()
            with self._lock:
                q = self._sessions.get(session_id, deque())
                return sum(s for ts, s in q if (now - ts) <= self._window_sec)
        except Exception:
            return 0

    @classmethod
    def from_config(cls, config: dict) -> "SessionRiskAccumulator":
        sr = (config.get("guardrails") or {}).get("session_risk") or {}
        return cls(
            window_minutes=int(sr.get("window_minutes", _DEFAULT_WINDOW_MINUTES)),
            session_block_threshold=int(sr.get("session_block_threshold", _DEFAULT_BLOCK_THRESHOLD)),
        )


# Module-level singleton
_accumulator: SessionRiskAccumulator | None = None


def get_accumulator(config: dict | None = None) -> SessionRiskAccumulator:
    global _accumulator
    if _accumulator is None:
        _accumulator = SessionRiskAccumulator.from_config(config or {})
    return _accumulator
