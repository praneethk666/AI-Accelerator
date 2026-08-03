"""
rollout.py — sticky per-session canary rollout helper.

A session is consistently in or out of the canary treatment group.
This prevents attackers from retrying the same payload to bypass checks when rollout < 100%.
"""
from __future__ import annotations

import hashlib


def should_apply_guard(pct: int, session_id: str) -> bool:
    """
    Returns True if the session falls into the canary rollout percentage.
    Deterministic based on session_id hash.
    """
    if pct >= 100:
        return True
    if pct <= 0:
        return False
    if not session_id:
        return True  # default to safe behavior if session_id is missing
    
    bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest(), 16) % 100
    return bucket < pct
