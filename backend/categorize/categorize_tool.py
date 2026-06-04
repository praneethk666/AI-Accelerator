"""Pipeline tool entrypoint for document categorization.

This module provides a stable `run()` function consumed by the pipeline graph.

Contract:
- Accept: file_path, state, deployment(optional) depending on pipeline wiring.
- Always write the following state fields:
    - state["route"]
    - state["document_type"]
    - state["industry"]
    - state["categorization_confidence"]
- Keep `state["reasoning"]` and `state["errors"]`.
- Never crash: wrap run() in try/except and fall back to route=text_default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .classifier import categorize


def run(file_path: str, state: Dict[str, Any], deployment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Ensure error list exists.
    state.setdefault("errors", [])
    try:
        return categorize(file_path=file_path, state=state, deployment=deployment)
    except Exception as e:
        state["errors"].append(f"categorize_tool: exception {type(e).__name__}: {e}")
        # Minimal guaranteed fields
        state["route"] = "text_default"
        state["document_type"] = "report"
        state["industry"] = (deployment or {}).get("default_industry", "automotive")
        state["categorization_confidence"] = 0.0
        state["reasoning"] = "categorize failed; returning safe fallback"
        return {
            "route": state["route"],
            "document_type": state["document_type"],
            "industry": state["industry"],
            "confidence": state["categorization_confidence"],
            "reasoning": state["reasoning"],
        }

