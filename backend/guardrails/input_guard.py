"""
input_guard.py — Checkpoint 1: checks applied before the agent processes a query.

Checks (fast-first order):
  1. Query length cap         — hard block, no scoring
  2. PII detection + redact   — redact in query copy, continue with WARN score
  3. Injection regex          — pattern match on NFKC-normalized copy
  4. Structural heuristics    — imperative instruction verbs ("act as", "ignore all")
  5. Basic entropy check      — catches obvious base64 injection (disabled by default)

PII detect order (prevents substring conflicts):
  GSTIN → PAN → Aadhaar → Credit card → Email → UPI → Phone

Fail mode: OPEN — crash → allow + bypassed=True logged.
"""
from __future__ import annotations

import re
import time
import logging

from backend.guardrails.guard_decision import GuardDecision, PolicyDecision, FailMode
from backend.guardrails.normalizer import normalize
from backend.guardrails.risk_score import score_from_rules
from backend.guardrails.safe_wrapper import guardrail_safe

logger = logging.getLogger(__name__)

# ── Config defaults (overridden by global.yaml at runtime) ────────────────────
_DEFAULT_MAX_QUERY_CHARS = 2000

# ── PII Patterns (detection order matters — GSTIN before PAN) ─────────────────

# 1. GSTIN: 15 chars  e.g. 22AAAAA0000A1Z5
_GSTIN = re.compile(
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"
)

# 2. PAN: 10 chars  e.g. ABCDE1234F
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

# 3. Aadhaar: 12 digits (space/hyphen variants)
_AADHAAR = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")

# 4. Credit card (basic 16-digit, not Luhn-validated here — fast check only)
_CREDIT_CARD = re.compile(r"\b(?:\d[ \-]?){15,16}\b")

# 5. Email (runs before UPI so no double-masking)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# 6. UPI — allowlist-based (avoids email collision)
_UPI_HANDLES = {
    "okaxis", "okhdfcbank", "okicici", "oksbi", "ybl", "ibl", "axisbank",
    "upi", "paytm", "apl", "abfspay", "allbank", "aubank", "barodampay",
    "centralbank", "dbs", "equitas", "fbl", "federal", "finobank", "hdfcbank",
    "icici", "idbi", "idfcbank", "indus", "jkb", "jsb", "karb", "kotak",
    "kvb", "lvb", "mahb", "nsdl", "obc", "pingpay", "postpay", "pnb",
    "rbl", "sbi", "scb", "sib", "syndicate", "timecosmos", "uco", "union",
    "utbi", "vijb",
}
_UPI = re.compile(
    r"\b[\w.\-]+@(" + "|".join(re.escape(h) for h in _UPI_HANDLES) + r")\b",
    re.IGNORECASE,
)

# 7. Indian phone: 10 digits starting with 6-9, optional +91/0 prefix
_PHONE = re.compile(r"(?:\+91|0)?[6-9]\d{9}\b")

# ── Injection Patterns (normalized copy only) ─────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+\w+", re.I),
    re.compile(r"(system|human|assistant)\s*:\s*\[", re.I),
    re.compile(r"</?(system|human|assistant|instruction|context)>", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|training)", re.I),
    re.compile(r"DAN\s+mode|do\s+anything\s+now", re.I),
    re.compile(r"jailbreak|prompt\s+injection", re.I),
]

# ── Structural instruction heuristics (score: 20 → WARN, not BLOCK alone) ────
_INSTRUCTION_VERBS = re.compile(
    r"\b(pretend|roleplay|imagine|act\s+as|behave\s+as|simulate|impersonate|forget\s+everything)\b",
    re.I,
)


# ── PII redaction helpers ─────────────────────────────────────────────────────

def _redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Return (redacted_text, list_of_fired_rules).
    Detection order: GSTIN → PAN → Aadhaar → Credit card → Email → UPI → Phone
    Email must run before UPI to prevent double-masking.
    """
    fired: list[str] = []

    def _sub(pattern, label, t):
        new, n = pattern.subn(f"[{label} REDACTED]", t)
        if n:
            fired.append("pii_in_query")
        return new

    text = _sub(_GSTIN,       "GSTIN",       text)
    text = _sub(_PAN,         "PAN",         text)
    text = _sub(_AADHAAR,     "AADHAAR",     text)
    text = _sub(_CREDIT_CARD, "CARD",        text)
    text = _sub(_EMAIL,       "EMAIL",       text)
    text = _sub(_UPI,         "UPI",         text)
    text = _sub(_PHONE,       "PHONE",       text)
    return text, list(set(fired))


# ── Main guard function ───────────────────────────────────────────────────────

@guardrail_safe("input")
def check_input(
    query: str,
    *,
    config: dict | None = None,
    session_id: str = "",
    version: str = "1.0.0",
) -> GuardDecision:
    """
    Run all input checks. Returns a GuardDecision.

    If the query is safe: allowed=True, sanitized_value=redacted_query (may differ
    from original if PII was found and redacted).
    If the query is blocked: allowed=False, sanitized_value=None.
    """
    t0 = time.perf_counter()
    g_cfg = (config or {}).get("guardrails", {})
    inp_cfg = g_cfg.get("input", {})
    max_chars = int(inp_cfg.get("max_query_chars", _DEFAULT_MAX_QUERY_CHARS))

    fired_rules: list[str] = []
    sanitized = query

    # ── 1. Length cap (hard block — deterministic) ────────────────────────────
    if len(query) > max_chars:
        return GuardDecision(
            allowed=False,
            policy=PolicyDecision.BLOCK,
            stage="input",
            event_type="query_too_long",
            risk_score=100,
            reason=f"Query length {len(query)} exceeds limit {max_chars}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            guardrail_version=version,
            hard_block=True,
        )

    # ── 2. PII detection + redaction ─────────────────────────────────────────
    if inp_cfg.get("pii_redact", True):
        sanitized, pii_rules = _redact_pii(query)
        fired_rules.extend(pii_rules)

    # Normalize a COPY for injection scanning (original/sanitized unchanged)
    normalized = normalize(sanitized)

    # ── 3. Injection regex patterns ───────────────────────────────────────────
    if inp_cfg.get("injection_check", True):
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(normalized):
                fired_rules.append("regex_injection_match")
                break   # one match is enough to score this rule

    # ── 4. Structural heuristics ──────────────────────────────────────────────
    if inp_cfg.get("injection_check", True):
        if _INSTRUCTION_VERBS.search(normalized):
            fired_rules.append("structural_instruction_verb")

    # ── 5. Score and decide ───────────────────────────────────────────────────
    block_threshold = int(g_cfg.get("policy", {}).get("block_threshold", 80))
    warn_threshold  = int(g_cfg.get("policy", {}).get("warn_threshold", 40))

    score = score_from_rules(fired_rules)
    latency = (time.perf_counter() - t0) * 1000

    if score >= block_threshold:
        return GuardDecision(
            allowed=False,
            policy=PolicyDecision.BLOCK,
            stage="input",
            event_type="injection_blocked",
            risk_score=score,
            reason=f"Rules fired: {fired_rules}",
            latency_ms=latency,
            guardrail_version=version,
        )

    if score >= warn_threshold:
        return GuardDecision(
            allowed=True,
            policy=PolicyDecision.WARN,
            stage="input",
            event_type="input_warn",
            risk_score=score,
            reason=f"Rules fired: {fired_rules}",
            sanitized_value=sanitized,
            latency_ms=latency,
            guardrail_version=version,
        )

    return GuardDecision(
        allowed=True,
        policy=PolicyDecision.ALLOW,
        stage="input",
        event_type="input_ok",
        risk_score=score,
        sanitized_value=sanitized if sanitized != query else None,
        latency_ms=latency,
        guardrail_version=version,
    )
