"""
trust_registry.py — default trust scores per document type.

Used by the TokenBudgetManager to rank retrieved chunks:
  rank = rerank_score * document_trust_score

Allows the system to prioritize chunks from highly-trusted official sources (manuals, policies)
over less structured or derived sources (OCR images, spreadsheets) when selecting context
for the LLM.
"""
from __future__ import annotations

# Default trust registry values (can be overridden by global config)
DEFAULT_TRUST_SCORES: dict[str, float] = {
    "manual":              1.0,
    "datasheet":           1.0,
    "contract":            0.95,
    "policy":              0.95,
    "report":              0.90,
    "financial_statement": 0.90,
    "invoice":             0.85,
    "purchase_order":      0.85,
    "spreadsheet":         0.80,
    "presentation":        0.75,
    "research_paper":      0.70,
    "image":               0.60,   # OCR-derived text
    "unknown":             0.50,
}


def get_trust_score(doc_type: str, config: dict | None = None) -> float:
    """Return the trust score for *doc_type*. Defaults to DEFAULT_TRUST_SCORES if unset."""
    g = (config or {}).get("guardrails", {})
    tr = g.get("trust_registry") or {}
    if doc_type in tr:
        try:
            return float(tr[doc_type])
        except (ValueError, TypeError):
            pass
    return DEFAULT_TRUST_SCORES.get(doc_type, DEFAULT_TRUST_SCORES["unknown"])
