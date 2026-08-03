"""backend/guardrails/ambiguity_detector.py

Confidence-based Ambiguity Detector.

Analyses top reranked chunks:
If top candidates belong to distinct entities and their score delta is small (< 0.12),
flags the retrieval result as ambiguous and extracts candidate options for request_clarification.
If one candidate dominates (delta >= 0.12), returns is_ambiguous = False.
"""
from __future__ import annotations

import logging
from backend.agent.context_manager import detect_explicit_entity

logger = logging.getLogger(__name__)


def check_ambiguity(reranked_chunks: list[dict], max_delta: float = 0.12) -> tuple[bool, list[str]]:
    """Check if top reranked chunks contain multiple close-scoring distinct entities.

    Returns:
        (is_ambiguous, candidate_options_list)
    """
    if len(reranked_chunks) < 2:
        return False, []

    top_chunk = reranked_chunks[0]
    top_score = top_chunk.get("rerank_score") or top_chunk.get("score") or 0.0

    if top_score == 0.0:
        return False, []

    candidate_entities = set()
    first_entity = detect_explicit_entity(
        (top_chunk.get("content") or "") + " " + str(top_chunk.get("metadata") or {})
    )
    if first_entity:
        candidate_entities.add(first_entity)

    # Check candidates within max_delta of top_score
    for chunk in reranked_chunks[1:5]:
        score = chunk.get("rerank_score") or chunk.get("score") or 0.0
        delta = top_score - score
        if delta <= max_delta:
            content = (chunk.get("content") or "") + " " + str(chunk.get("metadata") or {})
            detected = detect_explicit_entity(content)
            if detected:
                candidate_entities.add(detected)

    # If 2 or more distinct entities are tied in top candidates, query is ambiguous
    if len(candidate_entities) >= 2:
        options = [f"{e} Model" for e in sorted(candidate_entities)]
        options.append("Show All Models")
        logger.info(
            "Ambiguity detected in retrieval! Multiple entities tied within score delta %.2f: %s",
            max_delta,
            options,
        )
        return True, options

    return False, []
