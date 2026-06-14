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
        """Append one turn to the conversations table. Best-effort — the caller
        (answerer._log) already swallows exceptions, so a missing table or a
        down DB never blocks answering."""
        pg = PostgresStore()
        try:
            pg.conn.execute(
                """
                INSERT INTO conversations (session_id, turn, question, answer)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, turn, question, answer),
            )
        finally:
            pg.close()
