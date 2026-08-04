"""Agent-callable tool: find other documents that share an EXACT drawing/CAD-sheet
identifier with one the agent already knows about.

Why this exists, and why it's not redundant with search_documents: semantic search
finds documents about similar TOPICS; this finds documents PROVABLY about the same
DRAWING. A JTEKT drawing number ("00-83547001-0") or CAD sheet ID ("MS03AAA789AB")
is a strict, machine-generated identifier — if a manual's maintenance section and a
CAD sheet both cite the same one, that's a certain link, not a similarity guess.
This is exactly the "correlate CAD/manual/circuit" capability real industrial
corpora need and embeddings alone don't give you (validated live, 3-Aug, against
the real Toyoda/JTEKT document set).

Reads backend/categorize/id_graph.py's find_documents_by_id(), which queries the
metadata.mentioned_ids_flat tag every block gets at ingestion time (see
backend/pipeline/ingest.py) — no separate index to maintain.
"""
from __future__ import annotations

from typing import Any


class FindRelatedDocumentsTool:
    """Agent-callable tool: documents sharing an exact drawing/part identifier.

    Conforms to the AgentTool protocol in backend/agent_tools.py.
    """

    name = "find_related_documents"
    description = (
        "Find OTHER ingested documents that reference the SAME drawing number or "
        "CAD sheet ID as one you already know about (e.g. spotted in a citation's "
        "text or an image_ground answer, like '00-83547001-0' or 'MS03AAA789AB'). "
        "These are strict, machine-generated identifiers -- a match is a PROVEN "
        "link between documents, not a similarity guess. Use this when a question "
        "spans document types (e.g. you found a drawing number mentioned in a "
        "manual and need the actual CAD drawing, or vice versa). Pass the EXACT "
        "identifier string as you found it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": "The exact drawing number or CAD sheet ID to search "
                "for, e.g. '00-83547001-0' or 'MS03AAA789AB'.",
            },
        },
        "required": ["identifier"],
    }

    def run(self, identifier: str = "") -> dict[str, Any]:
        identifier = (identifier or "").strip()
        if not identifier:
            return {"error": "identifier is required — pass the exact drawing "
                             "number or CAD sheet ID string."}

        from backend.categorize.id_graph import find_documents_by_id
        rows = find_documents_by_id(identifier)
        if not rows:
            return {
                "identifier": identifier,
                "documents": [],
                "note": "No ingested document mentions this exact identifier. It "
                        "may not be ingested yet, or the identifier may be slightly "
                        "off (check exact spelling/formatting).",
            }

        by_doc: dict[str, dict] = {}
        for r in rows:
            doc_id = str(r["document_id"])
            entry = by_doc.setdefault(doc_id, {"document_id": doc_id, "mentions": []})
            if len(entry["mentions"]) < 5:   # cap per-document mentions in the payload
                entry["mentions"].append({
                    "page": (r.get("source_ref") or {}).get("page"),
                    "type": r.get("type"),
                    "snippet": (r.get("text") or "")[:200],
                })

        from backend.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            doc_meta = {str(d["document_id"]): d for d in store.list_documents()}
        finally:
            store.close()

        documents = []
        for doc_id, entry in by_doc.items():
            meta = doc_meta.get(doc_id, {})
            documents.append({
                "document_id": doc_id,
                "filename": meta.get("filename"),
                "document_type": meta.get("document_type"),
                "mentions": entry["mentions"],
            })

        return {"identifier": identifier, "documents": documents}

    __call__ = run
