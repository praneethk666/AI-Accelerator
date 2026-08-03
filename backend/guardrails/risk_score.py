"""
risk_score.py — additive risk score rules (0-100).

Each rule contributes a fixed number of points if it fires. The total across
all contributing rules determines the PolicyDecision in PolicyEngine.

Stage weights are applied by PolicyEngine when summing across stages:
  input × 1.0, retrieval × 1.2, output × 1.5

Individual rule scores (additive within a stage):
"""
from __future__ import annotations

# ── Thresholds (also in config/global.yaml — config values override these) ─────
DEFAULT_BLOCK_THRESHOLD = 80
DEFAULT_WARN_THRESHOLD  = 40

# ── Per-rule score contributions ───────────────────────────────────────────────
RULE_SCORES: dict[str, int] = {
    # Injection signals
    "regex_injection_match":        30,
    "unicode_homoglyph_detected":   20,
    "zero_width_chars_detected":    15,
    "structural_instruction_verb":  20,   # "act as", "pretend", "roleplay", "ignore" + instruction
    "base64_entropy_detected":      30,   # high-entropy base64-like string in query

    # PII in query (warn, not block — we redact and continue)
    "pii_in_query":                 25,

    # Retrieval stage
    "chunk_injection_found":        35,

    # Output stage
    "groundedness_unsupported":     50,
    "pii_in_output":                30,

    # Deterministic hard-blocks (hard_block=True, score goes to 100)
    "query_too_long":              100,
    "guard_crash_safe_reply":      100,
}

# Stage weights for cross-stage score aggregation in PolicyEngine
STAGE_WEIGHTS: dict[str, float] = {
    "input":     1.0,
    "retrieval": 1.2,
    "output":    1.5,
}


def score_from_rules(fired_rules: list[str]) -> int:
    """Sum scores for all fired rules, capped at 100."""
    total = sum(RULE_SCORES.get(r, 0) for r in fired_rules)
    return min(total, 100)
