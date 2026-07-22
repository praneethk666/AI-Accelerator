"""Agent-friendly wrapper around the existing query pipeline.

Expose one callable entrypoint that reuses run_query() and formats the result as
an answer payload with citations and source metadata.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from backend.core.config import load_config

logger = logging.getLogger(__name__)
from backend.core.registry import ToolRegistry
from backend.pipeline.default_registry import build_default_registry


def search_documents(
    query: str,
    document_scope: list[str] | None = None,
    *,
    doc_type: str | None = None,
    industry: str | None = None,
    registry: ToolRegistry | None = None,
    config: dict | None = None,
    session_id: str = "",
    conversation_history: list | None = None,
) -> dict[str, Any]:
    """Answer a question with citations using the existing query pipeline.

    doc_type / industry are OPTIONAL soft metadata filters: pass them only when the
    query clearly implies a scope (e.g. 'in the invoices…'); they narrow the candidate
    chunks by their ingest-time tags. Omit them to search the whole corpus.
    """
    final = _run_query(
        query,
        registry or build_default_registry(),
        config or _load_default_config(),
        session_id=session_id,
        document_scope=document_scope,
        doc_type=doc_type,
        industry=industry,
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
        "restrict to specific document ids/filenames (document_scope), or SOFT-filter "
        "by doc_type / industry — but pass those ONLY when the question clearly implies "
        "a scope (e.g. 'in the invoices…'), and use a value you saw from list_documents. "
        "When unsure, omit all filters and search the whole corpus."
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
            "doc_type": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Optional soft filter by document type (e.g. 'invoice', 'manual', "
                               "'cad_drawing'). Use ONLY when the question clearly implies a type "
                               "and it matches a type you saw via list_documents. Else null.",
            },
            "industry": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Optional soft filter by industry (e.g. 'finance', 'manufacturing'). "
                               "Use sparingly — only when the question clearly names an industry. Else null.",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, document_scope: list[str] | str | None = None,
            doc_type: str | None = None, industry: str | None = None) -> dict[str, Any]:
        if not document_scope or document_scope in ("null", "None"):
            document_scope = None
        elif isinstance(document_scope, str):
            document_scope = [document_scope]
        doc_type = None if doc_type in (None, "", "null", "None") else str(doc_type).strip().lower()
        industry = None if industry in (None, "", "null", "None") else str(industry).strip().lower()

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
                    # filename without extension (exact stem match, >=3 chars)
                    stem_hit = next(
                        (fid for fname, fid in filename_map.items()
                         if os.path.splitext(fname)[0] == item_lower and len(item_lower) >= 3),
                        None,
                    )
                    if stem_hit:
                        resolved_scope.append(stem_hit)
                    else:
                        # Unresolvable scope item: DROP it. Injecting it as a document_id
                        # would match zero points and yield a false 'not found'. If nothing
                        # resolves we fall through to a whole-corpus search.
                        logger.warning(
                            "search_documents: document_scope item %r matched no ingested "
                            "id/filename — ignoring it", item_str,
                        )

        # Auto-scope from the query TEXT only on a STRONG, intentional signal: a full
        # filename WITH extension appearing verbatim in the question. NOT a bare stem —
        # on a large corpus of generic names ('report.pdf', 'data.pdf') a stem match
        # would silently restrict a normal question to one (usually wrong) document.
        if not resolved_scope:
            query_lower = query.lower()
            for fname, fid in filename_map.items():
                if "." in fname and fname in query_lower:
                    resolved_scope.append(fid)

        final_scope = list(set(resolved_scope)) if resolved_scope else None
        return search_documents(query, final_scope, doc_type=doc_type, industry=industry)

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
            "document_id": citation.get("document_id"),
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