"""Conversation store interface + PostgreSQL-backed implementation.

QueryPlannerTool loads history (to contextualize follow-ups); AnswererTool saves
each turn; the agent-chat endpoint saves turns too (with tool-call metadata) so
the chat UI's session sidebar can list and reopen past conversations. All go
through this one interface so conversation logging lives in a single place.

Table (created by scripts/init_db.sql):
    conversations(id, session_id, role, content, metadata, created_at)

Do not change the Protocol signatures without telling the team.
"""
from __future__ import annotations

from typing import Protocol

from backend.storage.postgres_store import PostgresStore, _Json


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str,
                  metadata: dict | None = None) -> None:
        """Append one turn to the conversation history."""
        ...

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        """Return the last n turns as [{"role", "content", "metadata"}, ...]."""
        ...

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Most-recently-active sessions: [{"session_id", "title", "last_active"}]."""
        ...

    def delete_session(self, session_id: str) -> None:
        """Delete every turn for one session."""
        ...


class PostgresConversationStore:
    """ConversationStore backed by the `conversations` table.

    Per-message schema (role/content/metadata) — the standard chat shape:
    tolerates system/tool messages, non-alternating turns, and streaming.
    session_id is TEXT so the web UI's "web" session (a non-UUID) works.
    metadata is JSONB, used for e.g. an assistant turn's tool_calls (agent chat).
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
                    metadata    JSONB,
                    created_at  TIMESTAMP DEFAULT NOW()
                )
                """
            )
            # existing DBs from before `metadata` existed — additive, safe to re-run
            pg.conn.execute(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS metadata JSONB"
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations "
                "ON conversations (session_id, created_at)"
            )
        finally:
            pg.close()

    def save_turn(self, session_id: str, role: str, content: str,
                  metadata: dict | None = None) -> None:
        pg = PostgresStore()
        try:
            pg.conn.execute(
                """
                INSERT INTO conversations (session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, role, content, _Json(metadata) if metadata else None),
            )
        finally:
            pg.close()

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        pg = PostgresStore()
        try:
            rows = pg.conn.execute(
                """
                SELECT role, content, metadata FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, n),
            ).fetchall()
        finally:
            pg.close()
        # newest-first from SQL -> reverse to chronological for prompt context
        return [{"role": r[0], "content": r[1], "metadata": r[2]} for r in reversed(rows)]

    def list_sessions(self, limit: int = 50) -> list[dict]:
        pg = PostgresStore()
        try:
            rows = pg.conn.execute(
                """
                SELECT session_id, MIN(created_at) AS started_at,
                       MAX(created_at) AS last_active
                FROM conversations GROUP BY session_id
                ORDER BY last_active DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
            sessions = []
            for session_id, started_at, last_active in rows:
                title_row = pg.conn.execute(
                    """
                    SELECT content FROM conversations
                    WHERE session_id = %s AND role = 'user'
                    ORDER BY created_at ASC, id ASC LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                title = (title_row[0][:60] if title_row and title_row[0] else "New chat")
                sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "started_at": started_at.isoformat() if started_at else None,
                    "last_active": last_active.isoformat() if last_active else None,
                })
            return sessions
        finally:
            pg.close()

    def delete_session(self, session_id: str) -> None:
        pg = PostgresStore()
        try:
            pg.conn.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        finally:
            pg.close()


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
