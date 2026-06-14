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
    """ConversationStore backed by the `conversations` table.

    Per-message schema (role/content) — the standard chat shape: tolerates system/
    tool messages, non-alternating turns, and streaming. session_id is TEXT so the
    web UI's "web" session (a non-UUID) works.
    """

    def _ensure_schema(self) -> None:
        """Create the table if it's missing — keeps the store usable even if
        scripts/init_db.sql wasn't run. MUST stay in sync with init_db.sql."""
        pg = PostgresStore()
        try:
            pg.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          BIGSERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW()
                )
                """
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations "
                "ON conversations (session_id, created_at)"
            )
        finally:
            pg.close()

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
        store = PostgresConversationStore()
        try:
            store._ensure_schema()  # best-effort; harmless if the table exists
        except Exception:
            pass
        _store = store
    return _store
