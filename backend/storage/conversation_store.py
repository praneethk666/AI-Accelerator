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
        """Most-recently-active sessions: [{"session_id", "title", "pinned", "last_active"}]."""
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
        """Create tables if missing — keeps the store usable even if
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
            pg.conn.execute(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS metadata JSONB"
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations "
                "ON conversations (session_id, created_at)"
            )
            pg.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    title       TEXT,
                    pinned      BOOLEAN DEFAULT FALSE,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
                """
            )
        finally:
            pg.close()

    def save_turn(self, session_id: str, role: str, content: str,
                  metadata: dict | None = None) -> None:
        pg = PostgresStore()
        try:
            # ensure a sessions row exists (upsert so it's idempotent)
            pg.conn.execute(
                """
                INSERT INTO sessions (session_id, updated_at)
                VALUES (%s, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET updated_at = NOW()
                """,
                (session_id,),
            )
            pg.conn.execute(
                """
                INSERT INTO conversations (session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, role, content, _Json(metadata) if metadata else None),
            )
        finally:
            pg.close()

    def update_session(self, session_id: str, *,
                       title: str | None = None,
                       pinned: bool | None = None) -> None:
        """Update session metadata (custom title, pinned status)."""
        pg = PostgresStore()
        try:
            # ensure a row exists first
            pg.conn.execute(
                """
                INSERT INTO sessions (session_id) VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )
            sets = ["updated_at = NOW()"]
            params = []
            if title is not None:
                sets.append("title = %s")
                params.append(title)
            if pinned is not None:
                sets.append("pinned = %s")
                params.append(pinned)
            params.append(session_id)
            pg.conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = %s",
                params,
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
                SELECT c.session_id, MIN(c.created_at) AS started_at,
                       MAX(c.created_at) AS last_active,
                       s.pinned, s.title AS custom_title
                FROM conversations c
                LEFT JOIN sessions s ON s.session_id = c.session_id
                GROUP BY c.session_id, s.pinned, s.title
                ORDER BY COALESCE(s.pinned, FALSE) DESC, MAX(c.created_at) DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            sessions = []
            for session_id, started_at, last_active, pinned, custom_title in rows:
                if custom_title:
                    title = custom_title[:60]
                else:
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
                    "pinned": bool(pinned) if pinned else False,
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
            pg.conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
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
