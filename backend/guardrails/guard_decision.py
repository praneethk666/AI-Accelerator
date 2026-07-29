"""
guard_decision.py — shared data models for guardrail decisions and evidence.
All guardrail functions return GuardDecision; PolicyEngine consumes GuardEvidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PolicyDecision(str, Enum):
    ALLOW  = "allow"
    WARN   = "warn"
    REDACT = "redact"
    BLOCK  = "block"


class FailMode(str, Enum):
    OPEN       = "open"        # input / retrieval: crash → allow + log bypass
    CLOSED     = "closed"      # strict: crash → block (flip after staging validates)
    SAFE_REPLY = "safe_reply"  # output ONLY: crash → generic safe message, never raw


# Per-stage default fail modes — never change output to OPEN.
STAGE_FAIL_MODES: dict[str, FailMode] = {
    "input":     FailMode.OPEN,
    "retrieval": FailMode.OPEN,
    "output":    FailMode.SAFE_REPLY,
}

SAFE_REPLY_MESSAGE = (
    "I'm unable to complete this response safely right now. "
    "Please try again or rephrase your question."
)


@dataclass
class GuardDecision:
    allowed:   bool
    policy:    PolicyDecision
    stage:     str                       # 'input' | 'retrieval' | 'output'
    event_type: str                      # 'injection_blocked' | 'pii_redacted' | …
    risk_score: int = 0                  # 0-100 continuous score
    reason:    Optional[str] = None
    sanitized_value: Optional[str] = None  # cleaned query or answer if modified
    latency_ms: float = 0.0
    rule_id:   Optional[str] = None
    guardrail_version: str = "1.0.0"
    bypassed:  bool = False              # True when fail-open/safe-reply kicked in
    hard_block: bool = False             # True for deterministic high-confidence events
    session_cumulative_score: int = 0


@dataclass
class GuardEvidence:
    """Produced by each guard stage; consumed by PolicyEngine.evaluate()."""
    stage:      str
    risk_score: int
    events:     list[str] = field(default_factory=list)
    rule_ids:   list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    bypassed:   bool = False
    hard_block: bool = False             # if True, PolicyEngine short-circuits to BLOCK
