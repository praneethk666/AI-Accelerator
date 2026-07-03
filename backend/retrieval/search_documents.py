"""Agent-friendly wrapper around the existing query pipeline.

Expose one callable entrypoint that reuses run_query() and formats the result as
an answer payload with citations and source metadata.
"""
from __future__ import annotations

import os
from typing import Any

from backend.core.config import load_config
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState
from backend.pipeline.default_registry import build_default_registry


def search_documents(
    query: str,
    document_scope: list[str] | None = None,
    *,
    registry: ToolRegistry | None = None,
    config: dict | None = None,
    session_id: str = "",
    conversation_history: list | None = None,
) -> dict[str, Any]:
    """Answer a question with citations using the existing query pipeline."""
    final = _run_query(
        query,
        registry or build_default_registry(),
        config or _load_default_config(),
        session_id=session_id,
        document_scope=document_scope,
        conversation_history=conversation_history,
    )
    citations = list(final.get("citations") or [])
    return {
        "answer": final.get("answer", ""),
        "citations": citations,
        "sources": _build_sources(citations),
    }


class SearchDocumentsTool:
    """Registry-friendly wrapper so agents can call document search as one tool."""

    name = "search_documents"
    signature = "search_documents(query: str, document_scope: list[str] | None = None)"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        result = search_documents(
            state.get("query", ""),
            state.get("document_scope"),
            config=config,
            session_id=state.get("session_id", ""),
            conversation_history=state.get("conversation_history"),
        )
        state["answer"] = result["answer"]
        state["citations"] = result["citations"]
        state["sources"] = result["sources"]
        return state

    def call(
        self,
        query: str,
        document_scope: list[str] | None = None,
        *,
        config: dict | None = None,
        session_id: str = "",
        conversation_history: list | None = None,
    ) -> dict[str, Any]:
        return search_documents(
            query,
            document_scope,
            config=config,
            session_id=session_id,
            conversation_history=conversation_history,
        )


def _load_default_config() -> dict:
    config_path = os.getenv("CONFIG_PATH", "config/global.yaml")
    return load_config(config_path)


def _run_query(*args, **kwargs):
    from backend.pipeline.query import run_query

    return run_query(*args, **kwargs)


def _build_sources(citations: list[dict]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for citation in citations:
        source = {
            "filename": citation.get("filename"),
            "page": citation.get("page"),
            "sheet": citation.get("sheet"),
            "slide": citation.get("slide"),
            "summary": citation.get("summary"),
            "snippet": citation.get("snippet"),
            "image_path": citation.get("image_path"),
        }
        key = (
            source["filename"],
            source["page"],
            source["sheet"],
            source["slide"],
            source["image_path"],
            source["snippet"],
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources
