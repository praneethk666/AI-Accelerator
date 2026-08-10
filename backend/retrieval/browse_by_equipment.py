"""Agent-callable tool: browse ingested content by the machine/component/
doc_category tags folder_router.py derives from a folder-structured corpus, rather
than searching for it semantically.

Why this exists, and why it's opt-in: search_documents already discards doc_type/
industry soft filters (backend/retrieval/search_documents.py) after a real,
confirmed incident where they silently hid relevant chunks -- an always-on filter
that can be WRONG is worse than no filter at all. This tool avoids repeating that
mistake by never running automatically: the agent calls it explicitly, only when
it already has real machine/component context (e.g. from a prior search's
citations, or the user naming the equipment directly), and is instructed to fall
back to search_documents on an empty result rather than trust a miss as "nothing
exists." It's a metadata browse (no embedding call), so it finds tagged content
semantic search might miss and vice versa -- the two are complementary, not a
replacement for each other.

Reads chunk["tags"] payload fields (machine/component/doc_category) that
backend/chunking/chunk_tool.py::_structural_tags_from_block propagates from
block.metadata.folder, which backend/categorize/folder_router.py only populates
when deployment.corpus_root is configured -- silently returns nothing on a
deployment that doesn't use a folder-structured corpus.
"""
from __future__ import annotations

import os
from typing import Any


def _load_default_config() -> dict:
    # Same pattern as search_documents.py's own _load_default_config -- tool.run()
    # is dispatched as tool.run(**args), args coming only from the LLM's own
    # tool-call arguments (never includes "config"), so any tool needing the full
    # YAML config loads it itself rather than expecting it to be injected.
    from backend.core.config import load_config
    return load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))


class BrowseByEquipmentTool:
    """Agent-callable tool: documents/chunks tagged with a given machine/
    component/doc_category. Conforms to the AgentTool protocol in
    backend/agent_tools.py.
    """

    name = "browse_by_equipment"
    description = (
        "Browse ingested content by EQUIPMENT/COMPONENT tags derived from the "
        "corpus's own folder structure (e.g. machine='120_CYLINDRICAL GRINDER', "
        "component='Spindlehead') -- for when you already know which machine or "
        "component a question is about and want everything filed under it, "
        "across document types (manual sections, CAD drawings, parts lists). "
        "This is a metadata browse, NOT a text search -- it will find nothing "
        "for a corpus without folder-derived tags, or if your machine/component "
        "spelling doesn't exactly match how the corpus is organized. ALWAYS fall "
        "back to search_documents if this returns empty -- an empty result here "
        "means 'not tagged this way', never 'doesn't exist'. Pass at least one "
        "of machine/component/doc_category."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "machine": {
                "type": "string",
                "description": "Machine/equipment folder name, e.g. '120_CYLINDRICAL GRINDER'.",
            },
            "component": {
                "type": "string",
                "description": "Component/subsystem folder name, e.g. 'Spindlehead'.",
            },
            "doc_category": {
                "type": "string",
                "description": "Document category folder name, e.g. '3.INSTRUCTION MANUAL'.",
            },
        },
    }

    def run(self, machine: str | None = None, component: str | None = None,
            doc_category: str | None = None) -> dict[str, Any]:
        filters = {k: v for k, v in
                   {"machine": machine, "component": component, "doc_category": doc_category}.items()
                   if v}
        if not filters:
            return {"error": "pass at least one of machine/component/doc_category"}

        config = _load_default_config()
        # Config-gated, off by default (query.agent.equipment_browse.enabled) until
        # a live pass confirms folder tags are actually populated and useful for a
        # given deployment -- same "off until validated" posture as
        # query.answerer.image_ground, gated the same way (inside run(), since
        # build_agent_registry() has no config param to gate registration itself).
        eb_cfg = (config.get("query") or {}).get("agent", {}).get("equipment_browse") or {}
        if not eb_cfg.get("enabled", False):
            return {"error": "browse_by_equipment is disabled for this deployment "
                             "(query.agent.equipment_browse.enabled: false). Use "
                             "search_documents instead."}

        from backend.retrieval.vector_store import VectorStore
        chunks = VectorStore.browse_by_filter(config, filters, limit=50)
        if not chunks:
            return {
                "filters": filters,
                "documents": [],
                "note": "No indexed content tagged this way. This does NOT mean the "
                        "equipment/component doesn't exist -- try search_documents instead.",
            }

        by_doc: dict[str, dict] = {}
        for c in chunks:
            doc_id = str(c.get("document_id") or "")
            if not doc_id:
                continue
            entry = by_doc.setdefault(doc_id, {"document_id": doc_id, "chunks": []})
            if len(entry["chunks"]) < 5:   # cap per-document payload
                ref = c.get("source_ref") or {}
                entry["chunks"].append({
                    "page": ref.get("page"),
                    "snippet": (c.get("text") or "")[:200],
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
                "chunks": entry["chunks"],
            })

        return {"filters": filters, "documents": documents}

    __call__ = run
