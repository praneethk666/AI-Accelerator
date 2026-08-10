"""backend/agent/context_manager.py

Deterministic Active Context State Manager for RAG sessions.

STATUS (as of guided-procedure design review): this module is not currently
imported or called anywhere in executor.py — verified by repo-wide grep. It
tracks a DIFFERENT kind of session context than guided-procedure state (RAG
system component follow-ups, e.g. "what's the reranker model?" -> "and its
speed?" — not manual/step tracking), so it is explicitly OUT OF SCOPE for the
guided-procedure feature and should not be reused or extended for it; its
3-turn/15-minute hybrid expiration model is also the wrong shape for a
step-by-step procedure a technician may pause on for much longer.

KNOWN BUG (separate from the above, tracked independently): update_context_from_metadata()
reads chunk.get("content") / chunk.get("metadata"), but the actual Chunk schema
(schemas.py) uses chunk["text"] / chunk["tags"] / chunk["source_ref"] throughout
the rest of the pipeline. If this module is ever wired in, that line will
silently no-op (always operating on "" + "{}") rather than error. Whoever
revives this module should fix that first.

Manages structured active entities across chat turns with:
1. Explicit entity detection & override.
2. Hybrid expiration (3 turns OR 15 minutes of inactivity).
3. Selective query rephrasing for single-entity follow-ups (e.g. 'model name').
4. Metadata-driven context updates from retrieved source chunks (ground truth).
"""
from __future__ import annotations

import time
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Known entity types and keywords in our document ecosystem
KNOWN_ENTITIES = {
    "Reranker": ["rerank", "reranker", "reranking", "bge-reranker", "cross-encoder"],
    "Categorize": ["categorize", "categorizer", "categorization"],
    "Vision Model": ["vision", "vision_ocr", "llama-3.2-11b", "vlm"],
    "Dense Embedding": ["dense embedding", "nomic-embed", "embedding model", "embeddings", "embedding"],
    "Sparse Embedding": ["sparse embedding", "bm25", "qdrant/bm25"],
    "Chunker": ["chunking", "chunker", "potion-base-32m", "chonkie"],
    "Query Planner": ["query_planner", "planner", "deepseek-chat"],
    "Answerer": ["answerer"],
}

# Queries that should NOT be rewritten (broad requests)
BROAD_QUERY_PATTERNS = [
    r"\bcompare\b", r"\blist\b", r"\bshow all\b", r"\bwhat models\b", r"\ball models\b", r"\bsummarize\b"
]

# Patterns representing ambiguous follow-ups
AMBIGUOUS_PATTERNS = [
    r"^\s*model\s*name\s*$",
    r"^\s*model\s*$",
    r"^\s*its\s+model\b",
    r"^\s*what\s+is\s+its\b",
    r"^\s*what\s+is\s+it\b",
    r"^\s*its\s+version\b",
    r"^\s*its\s+speed\b",
]

@dataclass
class ActiveContext:
    entity_name: str | None = None
    entity_type: str = "component"
    topic: str | None = None
    source_document: str | None = None
    updated_turn: int = 0
    updated_timestamp: float = field(default_factory=time.time)

    def is_valid(self, current_turn: int, max_turns: int = 3, max_seconds: float = 900.0) -> bool:
        """Check if context is still valid (not expired by turns or 15-min timeout)."""
        if not self.entity_name:
            return False
        turns_elapsed = current_turn - self.updated_turn
        if turns_elapsed > max_turns:
            return False
        if (time.time() - self.updated_timestamp) > max_seconds:
            return False
        return True


# Global in-memory session contexts store
_SESSION_CONTEXTS: dict[str, ActiveContext] = {}


def get_context(session_id: str) -> ActiveContext:
    """Retrieve or create active context for a session."""
    if session_id not in _SESSION_CONTEXTS:
        _SESSION_CONTEXTS[session_id] = ActiveContext()
    return _SESSION_CONTEXTS[session_id]


def detect_explicit_entity(text: str) -> str | None:
    """Detect if prompt explicitly names a known entity (e.g. 'embedding model')."""
    lower_text = text.lower()
    for entity, keywords in KNOWN_ENTITIES.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                return entity
    return None


def is_broad_query(text: str) -> bool:
    """Check if query is a broad/comparison request that should not be rewritten."""
    lower_text = text.lower()
    for pat in BROAD_QUERY_PATTERNS:
        if re.search(pat, lower_text):
            return True
    return False


def is_ambiguous_followup(text: str) -> bool:
    """Check if query is a short ambiguous follow-up."""
    lower_text = text.lower().strip()
    for pat in AMBIGUOUS_PATTERNS:
        if re.search(pat, lower_text):
            return True
    return len(lower_text.split()) <= 2 and not detect_explicit_entity(text)


def process_query_context(session_id: str, query: str, current_turn: int) -> tuple[str, bool]:
    """Process incoming query against session context.

    Returns:
        (processed_query, was_rewritten)
    """
    ctx = get_context(session_id)

    # 1. Check for explicit entity mention in prompt -> explicit override
    explicit_entity = detect_explicit_entity(query)
    if explicit_entity:
        logger.info("Explicit entity detected: '%s'. Updating context for session %s.", explicit_entity, session_id)
        ctx.entity_name = explicit_entity
        ctx.updated_turn = current_turn
        ctx.updated_timestamp = time.time()
        return query, False

    # 2. Check if broad request -> do not rewrite
    if is_broad_query(query):
        return query, False

    # 3. Check if ambiguous follow-up AND active context is valid
    if is_ambiguous_followup(query) and ctx.is_valid(current_turn):
        rewritten = f"{ctx.entity_name} {query}"
        logger.info("Rewrote ambiguous query '%s' -> '%s' using session context.", query, rewritten)
        return rewritten, True

    return query, False


def update_context_from_metadata(session_id: str, retrieved_chunks: list[dict], current_turn: int) -> None:
    """Update active context using retrieved chunk metadata (ground truth)."""
    if not retrieved_chunks:
        return
    ctx = get_context(session_id)

    # Scan top retrieved chunks for known entity mentions in metadata/text
    for chunk in retrieved_chunks[:3]:
        content = (chunk.get("content") or "") + " " + str(chunk.get("metadata") or {})
        detected = detect_explicit_entity(content)
        if detected:
            ctx.entity_name = detected
            ctx.updated_turn = current_turn
            ctx.updated_timestamp = time.time()
            doc = chunk.get("metadata", {}).get("document_id") or chunk.get("metadata", {}).get("file_name")
            if doc:
                ctx.source_document = doc
            logger.info("Updated active context for %s from metadata: entity='%s'", session_id, detected)
            break
