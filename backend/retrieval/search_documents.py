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
        "restrict the search to specific document ids or filenames."
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
                    {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ],
                "description": "Optional list of document ids or filenames to restrict the search to. "
                               "Can be an array of strings, a single string, or null."
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, document_scope: list[str] | str | None = None) -> dict[str, Any]:
        if not document_scope or document_scope in ("null", "None"):
            document_scope = None
        elif isinstance(document_scope, str):
            document_scope = [document_scope]

        from backend.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            docs = store.list_documents()
        finally:
            store.close()

        # Build maps: id -> id, and filename -> id
        id_set = {str(d["document_id"]) for d in docs}
        filename_map = {}
        for d in docs:
            fname = d.get("filename")
            if fname:
                filename_map[fname.lower()] = str(d["document_id"])

        resolved_scope = []

        if document_scope:
            for item in document_scope:
                item_str = str(item).strip()
                item_lower = item_str.lower()
                if item_str in id_set:
                    resolved_scope.append(item_str)
                elif item_lower in filename_map:
                    resolved_scope.append(filename_map[item_lower])
                else:
                    # check for filename without extension
                    matched = False
                    for fname, fid in filename_map.items():
                        name_part, _ = os.path.splitext(fname)
                        if len(name_part) >= 3 and name_part == item_lower:
                            resolved_scope.append(fid)
                            matched = True
                            break
                    if not matched:
                        resolved_scope.append(item_str)

        # If no scope resolved from document_scope, check the query text itself
        if not resolved_scope:
            query_lower = query.lower()
            for fname, fid in filename_map.items():
                if fname in query_lower:
                    resolved_scope.append(fid)
                else:
                    name_part, _ = os.path.splitext(fname)
                    if len(name_part) >= 3 and name_part in query_lower:
                        import re
                        pattern = r'\b' + re.escape(name_part) + r'\b'
                        if re.search(pattern, query_lower):
                            resolved_scope.append(fid)

        final_scope = list(set(resolved_scope)) if resolved_scope else None
        return search_documents(query, final_scope)

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
            "filename":    citation.get("filename"),
            "page":        citation.get("page"),
            "document_id": citation.get("document_id"),
            "score":       citation.get("score"),
            "sheet":       citation.get("sheet"),
            "slide":       citation.get("slide"),
            "summary":     citation.get("summary"),
            "snippet":     citation.get("snippet"),
            "image_path":  citation.get("image_path"),
        }
        key = (
            source["document_id"],
            source["page"],
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    # Sort by score descending — highest-confidence source first.
    sources.sort(key=lambda s: (s.get("score") or 0.0), reverse=True)
    return sources