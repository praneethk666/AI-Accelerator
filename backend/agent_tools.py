"""Agent-callable tools registry.

Agent tools are named, described, JSON-schema'd operations an LLM agent can invoke
— distinct from pipeline-step Tools (which run inside the graph on shared state).
This is the seam the agent-executor consumes: build_agent_registry() returns
{name: tool}, and each tool exposes .name, .description, .input_schema, and
run(**kwargs) -> dict (the standard tool-use shape).

Add a tool: append it in build_agent_registry() (or register it there). Connectors
(db/inventory/email/…) will land here too as they come online.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentTool(Protocol):
    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's arguments

    def run(self, **kwargs) -> dict: ...


def build_agent_registry() -> dict[str, AgentTool]:
    """Every agent-callable tool, keyed by name. The agent-executor advertises
    these to the LLM and dispatches calls by name."""
    from backend.agent.clarify_tool import RequestClarificationTool
    from backend.connectors.sql_read import SQLReadTool
    from backend.extraction.excel.excel_tool import ExcelTool
    from backend.pipeline.ingest import IngestDocumentTool
    from backend.retrieval.browse_by_equipment import BrowseByEquipmentTool
    from backend.retrieval.find_related_documents import FindRelatedDocumentsTool
    from backend.retrieval.get_page_context import GetPageContextTool
    from backend.retrieval.list_documents import ListDocumentsTool
    from backend.retrieval.search_documents import SearchDocumentsTool
    from backend.retrieval.view_page_image import ViewPageImageTool

    tools: list[AgentTool] = [
        IngestDocumentTool(),
        SearchDocumentsTool(),
        GetPageContextTool(),
        ViewPageImageTool(),
        FindRelatedDocumentsTool(),
        BrowseByEquipmentTool(),
        ListDocumentsTool(),
        SQLReadTool(),
        RequestClarificationTool(),
        ExcelTool(),
    ]
    return {t.name: t for t in tools}
