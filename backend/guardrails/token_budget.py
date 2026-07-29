"""
token_budget.py — context window token budget manager.

Selects the best chunks to pass to the LLM context window.
Ranks chunks by: rerank_score * document_trust_score
Greedily selects chunks until the configured token limit is reached.
Prevents context explosion and Denial-of-Wallet token bloat.
"""
from __future__ import annotations

import logging
from backend.guardrails.trust_registry import get_trust_score

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTEXT_TOKENS = 8000


def _estimate_tokens(text: str) -> int:
    """Fast, dependency-free token estimator (approx 4 characters per token)."""
    if not text:
        return 0
    return len(text) // 4


class TokenBudgetManager:
    def __init__(self, max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS, config: dict | None = None):
        self._max_tokens = max_context_tokens
        self._config = config

    def select_chunks(self, chunks: list[dict], budget_tokens: int | None = None) -> tuple[list[dict], int, int]:
        """
        Rank chunks by (rerank_score * trust_score) and select greedily.
        Each chunk is dict containing 'text' and 'score' (and optionally 'doc_type').

        Returns (selected_chunks, total_tokens, dropped_count).
        """
        max_budget = budget_tokens or self._max_tokens
        
        # 1. Rank chunks
        scored_pairs = []
        for c in chunks:
            # Score is usually in '_score' or 'score' or 'similarity'
            score = float(c.get("_score") or c.get("score") or c.get("similarity") or 0.0)
            doc_type = c.get("doc_type") or c.get("metadata", {}).get("doc_type", "unknown")
            trust = get_trust_score(doc_type, self._config)
            rank = score * trust
            scored_pairs.append((rank, c))

        # Sort descending by rank
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        selected: list[dict] = []
        used_tokens = 0
        dropped_count = 0

        # 2. Greedy select
        for _, chunk in scored_pairs:
            text = chunk.get("text", "")
            tokens = _estimate_tokens(text)
            
            if used_tokens + tokens > max_budget:
                dropped_count += 1
                continue
            
            selected.append(chunk)
            used_tokens += tokens

        logger.debug(
            "TokenBudgetManager: input_chunks=%d selected=%d used_tokens=%d max_budget=%d dropped=%d",
            len(chunks), len(selected), used_tokens, max_budget, dropped_count,
        )
        return selected, used_tokens, dropped_count

    @classmethod
    def from_config(cls, config: dict) -> "TokenBudgetManager":
        tb = (config.get("guardrails") or {}).get("token_budget") or {}
        return cls(
            max_context_tokens=int(tb.get("max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS)),
            config=config,
        )
