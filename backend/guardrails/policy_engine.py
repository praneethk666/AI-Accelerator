"""
policy_engine.py — central decision arbiter.

Guards produce GuardEvidence (risk scores + fired rules).
PolicyEngine aggregates evidence across ALL stages using weighted scoring,
then issues a single PolicyDecision for the whole request.

Key design:
  - Cross-stage weighted sum (not categorical winner-takes-all).
  - Two independent WARN signals (score 35 each) correctly compound to BLOCK.
  - hard_block=True from any stage → BLOCK immediately (high-confidence deterministic events).
"""
from __future__ import annotations

import logging

from backend.guardrails.guard_decision import GuardEvidence, PolicyDecision
from backend.guardrails.risk_score import (
    DEFAULT_BLOCK_THRESHOLD, DEFAULT_WARN_THRESHOLD, STAGE_WEIGHTS,
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    def __init__(
        self,
        block_threshold: int = DEFAULT_BLOCK_THRESHOLD,
        warn_threshold:  int = DEFAULT_WARN_THRESHOLD,
        stage_weights:   dict[str, float] | None = None,
    ):
        self._block = block_threshold
        self._warn  = warn_threshold
        self._weights = stage_weights or STAGE_WEIGHTS

    def evaluate(self, evidences: list[GuardEvidence]) -> PolicyDecision:
        """
        Aggregate evidence from all guard stages and return one PolicyDecision.

        Step 1: Hard-block override — any deterministic BLOCK event wins immediately.
        Step 2: Weighted score sum across stages — two WARN signals compound to BLOCK.
        Step 3: Threshold comparison → BLOCK | WARN | ALLOW.
        """
        if not evidences:
            return PolicyDecision.ALLOW

        # Step 1: Hard-block override
        for ev in evidences:
            if ev.hard_block:
                logger.debug(
                    "PolicyEngine: hard_block from stage=%s events=%s → BLOCK",
                    ev.stage, ev.events,
                )
                return PolicyDecision.BLOCK

        # Step 2: Weighted cross-stage score sum
        total_score = 0.0
        for ev in evidences:
            weight = self._weights.get(ev.stage, 1.0)
            weighted = ev.risk_score * weight
            total_score += weighted
            if ev.risk_score > 0:
                logger.debug(
                    "PolicyEngine: stage=%s score=%d weight=%.1f weighted=%.1f",
                    ev.stage, ev.risk_score, weight, weighted,
                )

        total_score = min(total_score, 100.0)
        logger.debug("PolicyEngine: total_weighted_score=%.1f", total_score)

        # Step 3: Threshold comparison
        if total_score >= self._block:
            return PolicyDecision.BLOCK
        if total_score >= self._warn:
            return PolicyDecision.WARN
        return PolicyDecision.ALLOW

    @classmethod
    def from_config(cls, config: dict) -> "PolicyEngine":
        """Build a PolicyEngine from the guardrails section of global.yaml."""
        g = (config.get("guardrails") or {}).get("policy") or {}
        weights = g.get("stage_weights") or STAGE_WEIGHTS
        return cls(
            block_threshold=int(g.get("block_threshold", DEFAULT_BLOCK_THRESHOLD)),
            warn_threshold=int(g.get("warn_threshold",  DEFAULT_WARN_THRESHOLD)),
            stage_weights=weights,
        )


def get_engine(config: dict | None = None) -> PolicyEngine:
    return PolicyEngine.from_config(config or {})
