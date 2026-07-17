"""Agent-friendly wrapper around the existing query pipeline.

Expose one callable entrypoint that reuses run_query() and formats the result as
an answer payload with citations and source metadata.
"""
from __future__ import annotations

import os
from typing import Any

from backend.core.config import load_config
from backend.core.registry import ToolRegistry
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
        "trace_id": final.get("trace_id"),
    }


class SearchDocumentsTool:
    """Agent-callable tool: answer a question from the ingested corpus with citations.

    Conforms to the AgentTool protocol in backend/agent_tools.py (name +
    description + input_schema + run(**kwargs) -> dict), so the agent-executor can
    advertise and dispatch it alongside ingest_document. Wraps search_documents() —
    it does NOT reimplement retrieval.
    """

    name = "search_documents"
    description = (
        "Search the ingested documents and answer a question, returning the answer "
        "with citations and a deduplicated list of source references. Optionally "
        "restrict the search to specific document ids."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer from the documents.",
            },
            "document_scope": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "string"},
                    {"type": "null"},
                ],
                "description": "Optional list of document ids to restrict the search to."
                                "Pass an array of ids, e.g. [\"4cf1a34e-...\"] — even for a "
                                "single document, still wrap it in an array. Pass null (or "
                                "omit) to search all documents."
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, document_scope: list[str] | None = None) -> dict[str, Any]:
        if isinstance(document_scope, str):
            document_scope = [document_scope]
        return search_documents(query, document_scope)

    __call__ = run


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