"""Agent-callable tool: navigate a document's own chapter/section structure
(backend/pipeline/outline_builder.py) node by node, instead of semantic search
happening to retrieve the right chunk among several similarly-worded procedures.

Research basis (external, 10-Aug): PageIndex-style tree navigation -- an LLM walks
a document's own outline, evaluating each node against the query and descending
only into relevant branches -- is validated on exactly this document class (long,
hierarchical technical manuals). Implemented here as explicit, individually-visible
tool calls (one per navigation step) rather than one opaque black-box search, so
every step is inspectable in the agent trace like any other tool call.

Config-gated off by default (query.agent.document_outline.enabled) pending a live
validation pass -- the underlying outline data is always built at ingest time
(cheap, no LLM call), only this tool's exposure to the agent is gated, so turning
it on doesn't require re-ingesting anything.
"""
from __future__ import annotations

import os
from typing import Any


def _load_default_config() -> dict:
    from backend.core.config import load_config
    return load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))


class BrowseDocumentOutlineTool:
    """Agent-callable tool: children of a document outline node. Conforms to the
    AgentTool protocol in backend/agent_tools.py.
    """

    name = "browse_document_outline"
    description = (
        "Navigate a document's own chapter/section structure, one level at a "
        "time -- like browsing a table of contents instead of searching for a "
        "section. Omit node_id to see the document's top-level chapters; pass a "
        "node_id from a previous call's result to see ITS children. Each node "
        "has a title and page range. Prefer this over search_documents when you "
        "need to locate one SPECIFIC named section reliably (e.g. before "
        "starting a guided walkthrough, or when a manual has several similarly-"
        "named procedures you need to tell apart). Not every document has an "
        "outline -- an empty result means use search_documents instead, not "
        "that the content doesn't exist."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document to navigate (from search results or list_documents).",
            },
            "node_id": {
                "type": "string",
                "description": "A node_id from a previous browse_document_outline call, "
                "to see its children. Omit for the top-level chapters.",
            },
        },
        "required": ["document_id"],
    }

    def run(self, document_id: str = "", node_id: str | None = None) -> dict[str, Any]:
        if not document_id:
            return {"error": "document_id is required"}

        config = _load_default_config()
        # Config-gated, off by default, same posture/reasoning as
        # browse_by_equipment -- see module docstring.
        outline_cfg = (config.get("query") or {}).get("agent", {}).get("document_outline") or {}
        if not outline_cfg.get("enabled", False):
            return {"error": "browse_document_outline is disabled for this deployment "
                             "(query.agent.document_outline.enabled: false). Use "
                             "search_documents instead."}

        from backend.pipeline.outline_builder import get_outline_children
        children = get_outline_children(document_id, node_id)
        if not children:
            return {
                "document_id": document_id,
                "node_id": node_id,
                "children": [],
                "note": "No outline available at this level -- this does NOT mean the "
                        "content doesn't exist. Use search_documents instead.",
            }
        return {"document_id": document_id, "node_id": node_id, "children": children}

    __call__ = run
