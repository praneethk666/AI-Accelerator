"""Conversation store interface + Postgres implementation.

AnswerTool calls load_history() before answering and save_turn() after, so
multi-turn follow-ups have prior context. Same Postgres instance as
postgres_store (reuses dsn_from_env), different table: `conversations`.

Do not change the function signatures without telling the AnswerTool owner.
"""

from __future__ import annotations

from typing import Protocol


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one turn to the conversation history."""
        ...

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        """Return the last n turns as [{"role": ..., "content": ...}, ...]."""
        ...


class PostgresConversationStore:
    """Postgres-backed store. One row per turn; reads back chronological."""

    def __init__(self, dsn: str | None = None) -> None:
        import psycopg

        # lazy import: keep this module importable without postgres_store's deps
        from backend.storage.postgres_store import dsn_from_env

        self.conn = psycopg.connect(dsn or dsn_from_env(), autocommit=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # must match scripts/init_db.sql conversations table (single source of truth)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """)
        # fast lookups + stable ordering per session
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations "
            "ON conversations (session_id, id)"
        )

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content),
        )

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        # take the last n by insert order, then flip to chronological (oldest first)
        rows = self.conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = %s "
            "ORDER BY id DESC LIMIT %s",
            (session_id, n),
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def close(self) -> None:
        self.conn.close()


def get_conversation_store() -> ConversationStore:
    """Return the active conversation store (Postgres-backed)."""
    return PostgresConversationStore()
