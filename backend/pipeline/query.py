"""Query-time orchestration — the read path (planner -> retrieval -> answerer).

Reuses the same config-driven graph as ingestion: the query profile is just a
different ordered list of steps (config["query"]["steps"]), no extractors. The
graph runs them in order with the same graceful-failure wrapper.

  run_query(query, registry, config, session_id=..., document_scope=...)
    -> final state with state["answer"], state["citations"], state["errors"]

Conversation history is loaded before the run (best-effort — a missing/down
store never blocks answering). The turn itself is saved by AnswererTool, which
owns the question+answer pair, so logging lives in exactly one place.
"""
from __future__ import annotations

import logging

from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState
from backend.pipeline.graph import build_pipeline

logger = logging.getLogger(__name__)

DEFAULT_QUERY_STEPS = ["query_planner", "retrieval", "answerer"]


def run_query(
    query: str,
    registry: ToolRegistry,
    config: dict,
    *,
    session_id: str = "",
    document_scope: list[str] | None = None,
    conversation_history: list | None = None,
) -> PipelineState:
    steps = config.get("query", {}).get("steps", DEFAULT_QUERY_STEPS)
    qconfig = PipelineConfig.from_dict({**config, "steps": steps, "route": "query"})
    graph = build_pipeline(registry, qconfig)

    history = conversation_history
    if history is None:
        history = _load_history(session_id, config)

    # Initialize every field the query tools read by direct index, so a tool that
    # does state["sub_questions"] (not .get) can't KeyError on a fresh run.
    state: PipelineState = {
        "query": query,
        "session_id": session_id,
        "document_scope": document_scope or [],
        "conversation_history": history or [],
        "standalone_query": "",
        "sub_questions": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "errors": [],
    }

    # AnswererTool saves the turn (it holds question + answer).
    return graph.invoke(state)


def _load_history(session_id: str, config: dict) -> list:
    if not session_id:
        return []
    try:
        from backend.storage.conversation_store import get_conversation_store

        return get_conversation_store().load_history(session_id)
    except Exception as exc:  # store down / not configured — degrade to no history
        logger.debug("conversation history unavailable: %s", exc)
        return []
