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
    raw_user_prompt: str | None = None,
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
        raw_user_prompt=raw_user_prompt,
    )
    citations = list(final.get("citations") or [])
    
    excel_doc_ids = []
    for c in citations:
        fname = c.get("filename", "")
        if fname.lower().endswith((".xlsx", ".xls", ".csv")):
            doc_id = c.get("document_id")
            if doc_id and doc_id not in excel_doc_ids:
                excel_doc_ids.append(doc_id)
                
    answer = final.get("answer", "")
    if excel_doc_ids:
        docs_str = ", ".join(str(d) for d in excel_doc_ids)
        directive = (
            f"\n\n[SYSTEM DIRECTIVE FOR AI ONLY - DO NOT SHOW TO USER: Part of this answer "
            f"was retrieved from an Excel file (Document ID: {docs_str}). "
            f"If the user asked for a calculation, aggregation, or if columns are missing, "
            f"you MUST immediately call excel_tool on this Document ID to fulfill the request. "
            f"Otherwise, just answer the question normally and DO NOT output this directive.]"
        )
        answer += directive

    res = {
        "answer": answer,
        "citations": citations,
        "sources": _build_sources(citations),
        "trace_id": final.get("trace_id"),
    }
    if final.get("ambiguity"):
        res["ambiguity"] = final["ambiguity"]
    return res


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
        "restrict to specific document ids/filenames (document_scope). "
        "Search queries across the whole corpus when document_scope is omitted."
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
                               "Can be an array of strings, a single string, or null. NEVER guess, infer, or invent a filename. Leave null to search all documents."
            },
            "file_type": {
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
                "description": "Optional file extension to restrict the search to (e.g. '.pdf', '.docx', '.xlsx'). Leave null to search all supported types."
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, document_scope: list[str] | str | None = None,
            file_type: list[str] | str | None = None,
            doc_type: str | None = None, industry: str | None = None, session_id: str | None = None,
            raw_user_prompt: str | None = None) -> dict[str, Any]:
        if not document_scope or document_scope in ("null", "None"):
            document_scope = None
        elif isinstance(document_scope, str):
            document_scope = [document_scope]
        # Ignore doc_type / industry soft filters to prevent missing relevant document chunks
        doc_type = None
        industry = None

        from backend.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            docs = store.list_documents()
        finally:
            store.close()

        # Build maps: id -> id, and filename -> list[id] (to safely handle duplicates)
        id_set = {str(d["document_id"]) for d in docs}
        filename_to_ids: dict[str, list[str]] = {}
        for d in docs:
            fname = d.get("filename")
            if fname:
                k = fname.lower().strip()
                filename_to_ids.setdefault(k, []).append(str(d["document_id"]))

        resolved_scope: list[str] = []

        if document_scope:
            for item in document_scope:
                item_str = str(item).strip()
                item_lower = item_str.lower()
                if item_str in id_set:
                    resolved_scope.append(item_str)
                elif item_lower in filename_to_ids:
                    resolved_scope.extend(filename_to_ids[item_lower])
                else:
                    # filename without extension (exact stem match, >=3 chars)
                    matched_stem = False
                    for fname, fids in filename_to_ids.items():
                        if os.path.splitext(fname)[0] == item_lower and len(item_lower) >= 3:
                            resolved_scope.extend(fids)
                            matched_stem = True
                    if not matched_stem:
                        logger.warning(
                            "search_documents: document_scope item %r matched no ingested "
                            "id/filename — ignoring it", item_str,
                        )

        # Auto-scope from the query TEXT only on a STRONG, intentional signal: a full
        # filename WITH extension appearing verbatim in the question.
        if not resolved_scope:
            query_lower = query.lower()
            for fname, fids in filename_to_ids.items():
                if "." in fname and fname in query_lower:
                    resolved_scope.extend(fids)

        final_scope = list(set(resolved_scope)) if resolved_scope else None

        explicit_spreadsheet_terms = ["in excel", "from spreadsheet", ".xlsx", ".csv", "spreadsheet"]
        target_prompt = (raw_user_prompt or "").lower()
        query_lower = (query or "").lower()
        if any(term in target_prompt for term in explicit_spreadsheet_terms) or any(term in query_lower for term in explicit_spreadsheet_terms):
            return {
                "answer": (
                    "Error: The user query explicitly targets Excel/spreadsheet data. "
                    "Using search_documents for Excel data is prohibited. "
                    "If you do not know the filename, call list_documents() to find the spreadsheet filename, then use excel_tool()."
                ),
                "citations": [],
                "sources": []
            }

        # Fail fast if the user explicitly requested spreadsheet file types
        if file_type:
            if isinstance(file_type, str):
                file_type = [file_type]
            if any(ft.lower() in (".xlsx", ".xls", ".csv") for ft in file_type if ft):
                return {
                    "answer": (
                        "Error: You restricted the search to a spreadsheet file type. "
                        "Using search_documents for spreadsheets is strictly prohibited. "
                        "Please call the 'excel_tool' tool instead."
                    ),
                    "citations": [],
                    "sources": [],
                }

        # We intentionally allow final_scope to remain None (global search).
        # We handle spreadsheet exclusion for BM25 natively inside retrieval.py
        # so that _excel_keyword_inject can still scan spreadsheets globally.

        # Prohibit search_documents on spreadsheet files; redirect to excel_tool
        if final_scope:
            for d in docs:
                fid = str(d["document_id"])
                if fid in final_scope:
                    fname = d.get("filename", "")
                    if fname.lower().endswith((".xlsx", ".xls", ".csv")):
                        return {
                            "answer": (
                                f"Error: The document '{fname}' is a spreadsheet. "
                                f"Using search_documents for spreadsheets is strictly prohibited. "
                                f"Please call the 'excel_tool' tool with this filename/id to query and inspect its contents instead."
                            ),
                            "citations": [],
                            "sources": [],
                        }

        return search_documents(query, final_scope, doc_type=None, industry=None, session_id=session_id, raw_user_prompt=raw_user_prompt)

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
            "sheet_index": citation.get("sheet_index"),
            "slide":       citation.get("slide"),
            "summary":     citation.get("summary"),
            "snippet":     citation.get("snippet"),
            "image_path":  citation.get("image_path"),
        }
        # Include slide and sheet in the deduplication key so that citations from different
        # slides or sheets are not collapsed down to a single generic (document_id, None) entry.
        key = (
            source["document_id"],
            source["page"],
            source["slide"],
            source["sheet"],
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    # Preserve the exact citation order returned by the LLM answerer
    # so the frontend PDF viewer opens directly to the primary cited page.
    return sources