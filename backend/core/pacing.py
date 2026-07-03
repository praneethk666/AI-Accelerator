"""Proactive, thread-safe rate pacing for free-tier model APIs.

Free tiers cap REQUESTS PER MINUTE (e.g. Gemma ~15/min, Groq ~30/min), so the right
spacing is 60/RPM seconds (≈4s, ≈2s) — NOT a minute. Rather than only reacting to
429s with backoff (which still wastes a failed call + a wait), we pace calls to stay
just under the limit so most 429s never happen; the client's backoff stays as a
safety net for the rest.

Usage:
    from backend.core import pacing
    pacing.pace("vision", min_interval_s)   # blocks until >= min_interval since the
                                            # last "vision" call, then stamps now

Keys are independent (e.g. "vision" vs "enrichment") because they hit different
providers/limits. min_interval_s <= 0 disables pacing for that key (paid tiers).
The per-key lock is held across the sleep on purpose: concurrent callers queue, so
the spacing holds even when several vision threads run at once.
"""
from __future__ import annotations

import threading
import time

_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_last: dict[str, float] = {}


def pace(key: str, min_interval_s: float) -> None:
    """Block until at least `min_interval_s` has elapsed since the last call for
    `key`, then record now. No-op when min_interval_s <= 0."""
    if not min_interval_s or min_interval_s <= 0:
        return
    with _guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        wait = _last.get(key, 0.0) + min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[key] = time.monotonic()
