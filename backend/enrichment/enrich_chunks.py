"""EnrichChunksTool — stamp category + lightweight text tags onto each chunk.

  run(state, config)
    READS  state["chunks"]        list[Chunk dict]
           state["industry"]      str  (from categorize)
           state["document_type"] str  (from categorize)
    WRITES chunk["tags"]          dict (industry, doc_type, section, keywords)

Why it matters: retrieval filters on these tags (Qdrant flattens chunk["tags"]
to the payload), so category-scoped search and citations depend on them being
present. Kept dependency-free — topic/keyword extraction is simple and offline;
a model-based enricher can swap in later behind the same tool name.
"""
from __future__ import annotations

import re
from collections import Counter

from backend.core.tool import PipelineState

# very common words to ignore when picking keywords (kept tiny + offline)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "by", "at", "from", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "we", "you", "they", "i",
    "not", "no", "can", "will", "shall", "may", "must", "should", "would",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


class EnrichChunksTool:
    name = "enrich_chunks"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        industry = state.get("industry")
        doc_type = state.get("document_type")
        top_k = config.get("enrichment", {}).get("keyword_count", 6)

        for chunk in state.get("chunks", []):
            tags = chunk.setdefault("tags", {})
            # category tags drive retrieval filtering — only set if not already there
            if industry is not None:
                tags.setdefault("industry", industry)
            if doc_type is not None:
                tags.setdefault("doc_type", doc_type)

            ref = chunk.get("source_ref") or {}
            section = ref.get("section") if isinstance(ref, dict) else None
            if section:
                tags.setdefault("section", section)

            tags.setdefault("keywords", _keywords(chunk.get("text") or "", top_k))

        return state


def _keywords(text: str, k: int) -> list[str]:
    """Frequency-ranked content words — a cheap, offline topic signal."""
    counts = Counter(
        w.lower()
        for w in _WORD.findall(text)
        if w.lower() not in _STOPWORDS
    )
    return [word for word, _ in counts.most_common(k)]
