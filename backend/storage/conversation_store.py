"""Conversation store interface + PostgreSQL-backed implementation.

QueryPlannerTool loads history (to contextualize follow-ups); AnswererTool saves
each turn. Both go through this one interface so conversation logging lives in a
single place.

Table (created by scripts/init_db.sql):
    conversations(id, session_id, role, content, created_at)

Do not change the Protocol signatures without telling the team.
"""
from __future__ import annotations

from typing import Protocol

from backend.storage.postgres_store import PostgresStore


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one turn to the conversation history."""
        ...

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        """Return the last n turns as [{"role": ..., "content": ...}, ...]."""
        ...


class PostgresConversationStore:
    """ConversationStore backed by the `conversations` table."""

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        pg = PostgresStore()
        try:
            pg.conn.execute(
                """
                INSERT INTO conversations (session_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_id, role, content),
            )
        finally:
            pg.close()

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        pg = PostgresStore()
        try:
            rows = pg.conn.execute(
                """
                SELECT role, content FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, n),
            ).fetchall()
        finally:
            pg.close()
        # newest-first from SQL -> reverse to chronological for prompt context
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    """Return the process-wide conversation store (PostgreSQL-backed)."""
    global _store
    if _store is None:
        _store = PostgresConversationStore()
    return _store
