"""Postgres-read adapter — thin layer over the canonical PostgresStore.

Retrieval/answerer reach Postgres through this static interface. The schema and
connection live in backend.storage.postgres_store (one writer/reader of truth).

  PGStore.fetch_by_ids(config, chunk_ids)                       -> list[chunk dict]
  PGStore.log_conversation(config, session_id, turn, q, answer) -> None (best-effort)
"""
from __future__ import annotations

import logging

from backend.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


class PGStore:
    @staticmethod
    def fetch_by_ids(config: dict, chunk_ids: list[str]) -> list[dict]:
        pg = PostgresStore()
        try:
            return pg.get_chunks_by_ids(chunk_ids)
        finally:
            pg.close()

    @staticmethod
    def log_conversation(
        config: dict, session_id: str, turn: int, question: str, answer: str
    ) -> None:
        """Persist one Q/A turn as a user row + an assistant row, via the single
        conversation store (one schema: session_id, role, content). Best-effort —
        answerer._log already swallows exceptions, so a down DB never blocks
        answering."""
        from backend.storage.conversation_store import get_conversation_store

        store = get_conversation_store()
        store.save_turn(session_id, "user", question)
        store.save_turn(session_id, "assistant", answer)
