"""QueryPlannerTool — turn a raw user question into retrieval-ready sub-questions.

  run(state, config)
    READS  state["query"]                str        — the raw user question
           state["conversation_history"] list       — prior turns (for follow-ups)
    WRITES state["standalone_query"]      str        — query rewritten to stand alone
           state["sub_questions"]         list[str]  — what RetrievalTool searches
    ERRORS state["errors"]                list       — append only, never raise

Two jobs in one LLM call:
  1. Contextualize: rewrite a follow-up ("what about its cost?") into a
     self-contained question using the conversation history.
  2. Decompose: split a multi-part question into focused sub-questions; a simple
     question yields exactly one. Each sub-question is searched independently and
     the hits are merged downstream.

Always degrades to sub_questions=[query] so the query pipeline never stalls.
"""
from __future__ import annotations

import json
import logging

from backend.core import usage
from backend.core.llm_client import get_llm_for, clean_message_content, resolve_model_provider
from backend.core.tool import PipelineState

logger = logging.getLogger(__name__)

_PROMPT = """You rewrite a user's question for a document search engine.

Conversation so far (may be empty):
{history}

User question: {query}

Do two things:
1. Rewrite the question so it stands alone without the conversation (resolve "it",
   "that", "they", etc.). If it is already standalone, keep it as-is.
   If the user question is a single keyword, entity, or a name (e.g. "Manoj", "stipend"), rewrite it as a proper search question asking for information about that keyword (e.g., "who is Manoj" or "what is the information about Manoj", "details about stipend").
2. Break it into 1 to {max_subs} focused sub-questions. A simple question becomes
   exactly one sub-question. Only split genuinely multi-part questions.

Reply with ONLY a JSON object, no prose:
{{"standalone_query": "...", "sub_questions": ["...", "..."]}}"""


class QueryPlannerTool:
    name = "query_planner"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        query = (state.get("query") or "").strip()
        if not query:
            state["standalone_query"] = ""
            state["sub_questions"] = []
            return state

        max_subs = config.get("query", {}).get("planner", {}).get(
            "max_sub_questions", 4
        )
        try:
            plan = _plan(query, state.get("conversation_history") or [], config, max_subs)
            standalone = (plan.get("standalone_query") or query).strip()
            subs = [s.strip() for s in (plan.get("sub_questions") or []) if s and s.strip()]
            state["standalone_query"] = standalone
            state["sub_questions"] = subs[:max_subs] or [standalone]
        except Exception as exc:
            logger.warning("QueryPlannerTool fell back to raw query: %s", exc)
            errors = state.get("errors") or []
            errors.append({"tool": "query_planner", "error": str(exc)})
            state["errors"] = errors
            state["standalone_query"] = query
            state["sub_questions"] = [query]
        return state


def _plan(query: str, history: list, config: dict, max_subs: int) -> dict:
    # query rewriting/decomposition benefits from a stronger model. Resolution:
    # query.planner.model -> llm.answer_model -> global llm.model.
    llm = get_llm_for(config, config.get("query", {}).get("planner"),
                      default_model=config["llm"].get("answer_model"))
    prompt = _PROMPT.format(
        history=_format_history(history), query=query, max_subs=max_subs
    )
    response = llm.invoke(prompt)
    model_name, provider_name = resolve_model_provider(
        config, config.get("query", {}).get("planner"),
        default_model=config["llm"].get("answer_model"),
    )
    usage.record_from_message("query_planner", response, model=model_name, provider=provider_name)
    return json.loads(_extract_json(clean_message_content(response.content)))


def _format_history(history: list) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history[-10:]:  # last 5 QA turns (10 messages total) for context
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of an LLM reply (handles ```json fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in LLM reply: {text[:120]!r}")
    return text[start : end + 1]
