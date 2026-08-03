"""
Pipeline tool entrypoint for document categorization.

Contract:
- Pipeline calls: tool.run(state, config)
- file_path is read from state["file_path"]
- Always write the following state fields:
    - state["route"]   #what is the exact state field written : route,document_type,industry,confidence,reasoning,errors
    - state["document_type"]
    - state["industry"]
    - state["confidence"]
- Keep state["reasoning"] and state["errors"]
- Never crash: wrap run() in try/except and fall back to route=text_default
"""

from __future__ import annotations

from typing import Any, Dict

from .classifier import categorize, detect_file_type


def _forced_result(file_path: str, config: Dict[str, Any]) -> Dict[str, Any] | None:
    """A narrow-scope deployment (a customer whose documents are ALWAYS one
    known type -- e.g. "this install only ever ingests automotive service
    manuals") can skip the classify LLM call entirely: the answer would be
    identical every time anyway, so paying for it (and the page-render +
    vision-call latency) is pure waste. Set deployment.force_document_type to
    enable; force_industry/force_route default from it if unset.

    file_type detection stays real/local (extension-based, free) since it's
    needed for extractor routing regardless of whether classification ran."""
    dep = config.get("deployment") or {}
    doc_type = dep.get("force_document_type")
    if not doc_type:
        return None
    type_to_route = config.get("type_to_route") or {}
    return {
        "route": dep.get("force_route") or type_to_route.get(doc_type, "text_default"),
        "document_type": doc_type,
        "industry": dep.get("force_industry") or config.get("default_industry", "general"),
        "confidence": 1.0,
        "file_type": detect_file_type(file_path),
        "reasoning": "deployment.force_document_type set; classification skipped",
        "errors": [],
    }


def _augment_pdf_kind(result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """For PDFs, classify digital/scanned/mixed so the graph picks the right
    PDF extractor (pdf_digital/scanned_pdf/mixed_pdf). categorize is the document
    -inspection step, so detection belongs here. Safe default: digital."""
    if state.get("file_type") != "pdf" or result.get("pdf_kind"):
        return result
    try:
        from backend.extraction.detector import detect_pdf_type

        kind, _ = detect_pdf_type(state["file_path"])
        result["pdf_kind"] = kind
    except Exception:
        result["pdf_kind"] = "digital"
    return result


class CategorizeTool:
    name = "categorize"

    def run(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline entrypoint.

        Args:
            state: Shared pipeline state. Must contain:
                state["file_path"]

            config: Global configuration loaded by the pipeline
                (config/global.yaml)

        Returns:
            Updated state dictionary.
        """

        state.setdefault("errors", [])

        file_path = state.get("file_path")

        if not file_path:
            state["errors"].append(
                "categorize_tool: missing file_path in state"
            )

            state.setdefault("route", "text_default")
            state.setdefault("document_type", "report")
            state.setdefault(
                "industry",
                config.get("default_industry", "automotive")
            )
            state.setdefault("confidence", 0.0)
            state.setdefault("file_type", "unknown")
            state.setdefault(
                "reasoning",
                "missing file_path; returning safe fallback"
            )

            return {
                "route": state["route"],
                "document_type": state["document_type"],
                "industry": state["industry"],
                "confidence": state["confidence"],
                "file_type": state["file_type"],
                "reasoning": state["reasoning"],
                "errors": state.get("errors", []),
            }

        try:
            forced = _forced_result(file_path, config)
            if forced is not None:
                return _augment_pdf_kind(forced, state)

            # Pass full config to categorize; it handles structure internally
            result = categorize(
                file_path=file_path,
                state=state,
                config=config,
                deployment=config.get("deployment", {})
            )
            return _augment_pdf_kind(result, state)

        except Exception as e:
            state["errors"].append(
                f"categorize_tool: exception {type(e).__name__}: {e}"
            )

            state["route"] = "text_default"
            state["document_type"] = "report"

            state["industry"] = (
                config.get("default_industry", "automotive")
            )

            state["confidence"] = 0.0
            state["file_type"] = "unknown"

            state["reasoning"] = (
                "categorize failed; returning safe fallback"
            )

            return {
                "route": state["route"],
                "document_type": state["document_type"],
                "industry": state["industry"],
                "confidence": state["confidence"],
                "file_type": state["file_type"],
                "reasoning": state["reasoning"],
                "errors": state.get("errors", []),
            }