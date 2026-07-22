"""Agent-callable tool: fetch a document page's full raw content, bypassing chunking.

Why this exists: chunking is inherently lossy — no fixed strategy handles every
document layout (validated live, 21-Jul: an alarm-code reference page split a
label like "Alarm code: 11H" from the corrective-action table that explains what
to do about it, because a heading sat between them). Rather than chase every
layout pattern in the chunker (unbounded — a losing battle), this gives the agent
a "Parent Document Retrieval" / "Auto-Merging Retrieval" style escape hatch: search
stays precise at the chunk level (fast, cheap, works most of the time), but when a
retrieved chunk looks fragmented or incomplete, the agent can zoom out to the
FULL PAGE for real context — a scoped, on-demand expansion, not a blanket
page-by-page traversal (which would be slow and defeat the point of the index).

Reads document_blocks (backend/storage/postgres_store.py: write_blocks/get_blocks)
— the raw extractor output persisted BEFORE chunking, in reading order — so this
reconstructs the page exactly as extracted, independent of how it got chunked.
"""
from __future__ import annotations

from typing import Any


class GetPageContextTool:
    """Agent-callable tool: full raw content of one page, in reading order.

    Conforms to the AgentTool protocol in backend/agent_tools.py.
    """

    name = "get_page_context"
    description = (
        "Fetch the FULL raw content of one page from a document, in reading order — "
        "bypassing chunking entirely. Use this when a search_documents result looks "
        "fragmented or incomplete (e.g. a bare label/code with no explanation, or a "
        "table whose surrounding context is unclear) and you need the complete page "
        "to answer confidently. Pass the document_id and page from a search_documents "
        "citation's 'sources' entry (source.document_id, source.page)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "document_id from a search_documents citation's source.",
            },
            "page": {
                "type": "integer",
                "description": "Page number from that same citation's source.",
            },
        },
        "required": ["document_id", "page"],
    }

    def run(self, document_id: str = "", page: int | str | None = None) -> dict[str, Any]:
        if not document_id or page is None:
            return {"error": "document_id and page are both required."}
        try:
            page_no = int(page)
        except (TypeError, ValueError):
            return {"error": f"page must be an integer, got {page!r}."}

        from backend.storage.postgres_store import PostgresStore

        store = PostgresStore()
        try:
            blocks = store.get_blocks(document_id)
        finally:
            store.close()

        if not blocks:
            return {
                "error": f"No extracted content found for document_id {document_id!r}. "
                         "It may not have finished ingesting, or the id is wrong — "
                         "check search_documents' citations for the correct document_id."
            }

        page_blocks = [
            b for b in blocks
            if isinstance(b.get("source_ref"), dict) and b["source_ref"].get("page") == page_no
        ]
        if not page_blocks:
            pages_seen = sorted({
                b["source_ref"].get("page") for b in blocks
                if isinstance(b.get("source_ref"), dict) and b["source_ref"].get("page") is not None
            })
            return {
                "error": f"No content found on page {page_no} of this document.",
                "pages_available": pages_seen[:20],
            }

        parts = [b.get("text") for b in page_blocks if (b.get("text") or "").strip()]
        return {
            "document_id": document_id,
            "page": page_no,
            "content": "\n\n".join(parts),
        }

    __call__ = run
