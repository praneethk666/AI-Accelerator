"""Conversation store interface + Postgres implementation.

AnswerTool calls load_history() before answering and save_turn() after, so
multi-turn follow-ups have prior context. Same Postgres instance as
postgres_store (reuses dsn_from_env), different table: `conversations`.

Storage note: the shared `conversations` table is per Q&A turn
(session_id, turn, question, answer) — not per message. This store maps the
role/content Protocol onto it: a "user" turn opens a row (answer filled later),
the next "assistant" turn closes it. Keep in sync with scripts/init_db.sql.

Do not change the function signatures without telling the AnswerTool owner (Abhishek).
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
    """Postgres-backed store over the shared per-turn `conversations` table."""

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
                session_id  UUID,
                turn        INTEGER,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT now(),
                PRIMARY KEY (session_id, turn)
            )
            """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations "
            "ON conversations (session_id, turn)"
        )

    def _next_turn(self, session_id: str) -> int:
        # turns are 1-based, contiguous per session
        row = self.conn.execute(
            "SELECT COALESCE(MAX(turn), 0) + 1 FROM conversations WHERE session_id = %s",
            (session_id,),
        ).fetchone()
        return row[0]

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        # answer NOT NULL → "" is the open-turn sentinel until the assistant replies
        if role == "assistant":
            # close the latest open user turn; if none, store an answer-only row
            updated = self.conn.execute(
                "UPDATE conversations SET answer = %s "
                "WHERE session_id = %s AND turn = "
                "(SELECT MAX(turn) FROM conversations WHERE session_id = %s AND answer = '')",
                (content, session_id, session_id),
            ).rowcount
            if not updated:
                self.conn.execute(
                    "INSERT INTO conversations (session_id, turn, question, answer) "
                    "VALUES (%s, %s, '', %s)",
                    (session_id, self._next_turn(session_id), content),
                )
        else:  # "user" (and any non-assistant role) opens a new turn
            self.conn.execute(
                "INSERT INTO conversations (session_id, turn, question, answer) "
                "VALUES (%s, %s, %s, '')",
                (session_id, self._next_turn(session_id), content),
            )

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        # last n turns by order, flipped to chronological, flattened to messages
        rows = self.conn.execute(
            "SELECT question, answer FROM conversations WHERE session_id = %s "
            "ORDER BY turn DESC LIMIT %s",
            (session_id, n),
        ).fetchall()
        history: list[dict] = []
        for question, answer in reversed(rows):
            if question:
                history.append({"role": "user", "content": question})
            if answer:
                history.append({"role": "assistant", "content": answer})
        return history

    def close(self) -> None:
        self.conn.close()


def get_conversation_store() -> ConversationStore:
    """Return the active conversation store (Postgres-backed)."""
    return PostgresConversationStore()
