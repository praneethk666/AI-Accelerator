"""
token_quota.py — per-session token budget enforcer (Denial-of-Wallet protection).

Uses reserve-and-reconcile pattern to prevent burst loophole:
  - reserve(n) at request START: atomically checks + deducts estimate from budget.
  - reconcile(reserved, actual) at request END: adjusts for actual usage.
  
This prevents 50 concurrent requests all seeing usage=0 before any finishes.

Storage: in-process (single worker). Redis upgrade path: set redis_url in config.
Fail mode: OPEN — quota check crash → allow request through.
Never raises.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_MINUTES   = 60
_DEFAULT_TOKENS_PER_WINDOW = 500_000
_DEFAULT_RESERVE_TOKENS   = 2_000


class TokenQuotaEnforcer:
    def __init__(
        self,
        window_minutes:    int = _DEFAULT_WINDOW_MINUTES,
        tokens_per_window: int = _DEFAULT_TOKENS_PER_WINDOW,
    ):
        self._lock = threading.Lock()
        # Each entry: (timestamp, tokens)
        self._usage: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._window_sec = window_minutes * 60
        self._limit = tokens_per_window

    def _current_total(self, session_id: str, now: float) -> int:
        q = self._usage[session_id]
        while q and (now - q[0][0]) > self._window_sec:
            q.popleft()
        return sum(t for _, t in q)

    def reserve(self, session_id: str, tokens: int) -> bool:
        """
        Atomically check budget and reserve *tokens*.
        Returns True if reservation succeeded (request may proceed).
        Returns False if budget exhausted (return 429 to caller).
        Fail-open: exception → returns True (allow through).
        """
        if not session_id:
            return True
        try:
            now = time.monotonic()
            with self._lock:
                current = self._current_total(session_id, now)
                if current + tokens > self._limit:
                    logger.warning(
                        "token_quota: session=%s current=%d reserve=%d limit=%d → DENIED",
                        session_id, current, tokens, self._limit,
                    )
                    return False
                self._usage[session_id].append((now, tokens))
                return True
        except Exception as exc:
            logger.warning("token_quota.reserve error (fail-open): %s", exc)
            return True

    def reconcile(self, session_id: str, reserved: int, actual: int) -> None:
        """
        Replace the reservation entry with actual usage.
        If actual > reserved, charges the overage. If actual < reserved, refunds.
        Never raises.
        """
        if not session_id:
            return
        try:
            now = time.monotonic()
            with self._lock:
                q = self._usage[session_id]
                # Find and remove the most recent entry matching reserved
                for i in range(len(q) - 1, -1, -1):
                    if q[i][1] == reserved:
                        # Remove it (deque doesn't support index deletion; rebuild)
                        items = list(q)
                        items.pop(i)
                        if actual > 0:
                            items.append((now, actual))
                        q.clear()
                        q.extend(items)
                        return
                # If reservation not found (shouldn't happen), just append actual
                if actual > 0:
                    q.append((now, actual))
        except Exception as exc:
            logger.warning("token_quota.reconcile error: %s", exc)

    def check_only(self, session_id: str) -> bool:
        """Returns True if session is currently over budget (no reservation made)."""
        if not session_id:
            return False
        try:
            now = time.monotonic()
            with self._lock:
                return self._current_total(session_id, now) >= self._limit
        except Exception:
            return False

    @classmethod
    def from_config(cls, config: dict) -> "TokenQuotaEnforcer":
        tq = (config.get("guardrails") or {}).get("token_quota") or {}
        return cls(
            window_minutes=int(tq.get("window_minutes", _DEFAULT_WINDOW_MINUTES)),
            tokens_per_window=int(tq.get("tokens_per_window", _DEFAULT_TOKENS_PER_WINDOW)),
        )


# Module-level singleton
_enforcer: TokenQuotaEnforcer | None = None


def get_enforcer(config: dict | None = None) -> TokenQuotaEnforcer:
    global _enforcer
    if _enforcer is None:
        _enforcer = TokenQuotaEnforcer.from_config(config or {})
    return _enforcer


def get_reserve_tokens(config: dict) -> int:
    tq = (config.get("guardrails") or {}).get("token_quota") or {}
    return int(tq.get("reserve_tokens", _DEFAULT_RESERVE_TOKENS))
