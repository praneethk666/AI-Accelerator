"""Agent-callable tool: list what's already been ingested.

Exists so the agent can see how many documents exist (and their filenames/types)
before deciding whether a reference like "this invoice" is unambiguous or needs
the user to pick one. Thin wrapper over PostgresStore.list_documents() — no new
storage logic.
"""
from __future__ import annotations

from typing import Any


class ListDocumentsTool:
    """Agent-callable tool: list ingested documents (id, filename, type, industry, status).

    Conforms to the AgentTool protocol in backend/agent_tools.py.
    """

    name = "list_documents"
    description = (
        "List documents already ingested and searchable, with their id, filename, "
        "document_type, industry, and status. Use this before search_documents when "
        "a question refers to 'this document'/'the invoice' etc. ambiguously and you "
        "need to know what's actually available, or when the user asks what's been "
        "ingested."
    )
    # A truly empty parameter object breaks Groq/Llama-3.3 tool-calling — it falls
    # back to malformed text-based function syntax for the whole turn (reproduced:
    # every other tool call in the same turn fails validation, not just this one).
    # One optional property keeps the schema non-empty without adding a real arg.
    input_schema = {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string",
                "description": "Optional: only list documents of this document_type (e.g. 'invoice').",
            },
        },
            "industry": {
                "type": "string",
                "description": "Optional: only list documents tagged with this industry "
                                "(e.g. 'healthcare', 'finance'). Matches the industry value "
                                "shown in each document's row, case-insensitively.",
            },
    }

    def run(self, document_type: str | None = None, industry: str | None = None) -> dict[str, Any]:
        from backend.storage.postgres_store import PostgresStore

        store = PostgresStore()
        try:
            docs = store.list_documents()
        finally:
            store.close()
        if document_type:
            docs = [d for d in docs if d.get("document_type") == document_type]
        if industry:
            docs = [d for d in docs
                    if (d.get("industry") or "").strip().lower() == industry.strip().lower()]
        return {
            "count": len(docs),
            "documents": [
                {
                    "document_id": d.get("document_id"),
                    "filename": d.get("filename"),
                    "document_type": d.get("document_type"),
                    "industry": d.get("industry"),
                    "status": d.get("status"),
                }
                for d in docs
            ],
        }

    __call__ = run
