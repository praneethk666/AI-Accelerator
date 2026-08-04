"""Sliding-window TOKEN-budget pacing — complements backend/core/pacing.py's
fixed-interval REQUEST pacing, for providers whose real binding constraint is
tokens-per-minute rather than requests-per-minute.

Real finding, 3-Aug: gpt-4o-mini figure captioning kept hitting 429 "tokens per
min" during a real ingestion run even with pacing.pace()'s min_interval_s tuned
down to 1.0 (a real, previously-live-validated fix for a DIFFERENT incident).
pacing.py's own docstring is explicit about the limitation: it paces REQUEST
rate, built for free-tier RPM caps (Gemma ~15/min, Groq ~30/min) — it has no
notion of how big any individual call is. A 200K TPM ceiling can be blown by
enough large calls even at a slow, steady request rate; a small burst of
token-heavy image crops is exactly that case.

This tracks actual TOKEN usage in a trailing 60s window per key and blocks a new
call only long enough for enough of the window to age out — not a fixed
interval, so it doesn't over-throttle small/cheap calls or under-throttle large
ones. Use ALONGSIDE pacing.pace(), not instead of it: RPM and TPM are different
constraints and a provider can hit either one.

Usage:
    from backend.core import token_pacer
    token_pacer.wait_for_tokens("vision", estimated_tokens, tpm_limit=200_000)
    ... make the call ...
"""
from __future__ import annotations

import threading
import time
from collections import deque

_guard = threading.Lock()
_budgets: dict[str, "_TokenBudget"] = {}


class _TokenBudget:
    def __init__(self, window_s: float = 60.0):
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()
        self._window_s = window_s

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self._window_s:
            self._events.popleft()

    def wait_for(self, estimated_tokens: int, tpm_limit: int) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._prune(now)
                used = sum(t for _, t in self._events)
                if used + estimated_tokens <= tpm_limit:
                    self._events.append((now, estimated_tokens))
                    return
                # Not enough headroom yet -- sleep until the oldest event in the
                # window ages out, then re-check (another thread may have freed
                # up room, or consumed it, in the meantime).
                oldest_ts = self._events[0][0]
                sleep_for = self._window_s - (now - oldest_ts) + 0.05
                time.sleep(max(sleep_for, 0.05))


def wait_for_tokens(key: str, estimated_tokens: int, tpm_limit: int) -> None:
    """Block until `estimated_tokens` more can be spent under `tpm_limit` in the
    trailing 60s for `key`, then record the spend. No-op when tpm_limit <= 0
    (opt-in per key/provider — a provider with no known TPM ceiling, or a paid
    tier high enough to never matter, just doesn't set one)."""
    if not tpm_limit or tpm_limit <= 0 or estimated_tokens <= 0:
        return
    with _guard:
        budget = _budgets.setdefault(key, _TokenBudget())
    budget.wait_for(estimated_tokens, tpm_limit)
