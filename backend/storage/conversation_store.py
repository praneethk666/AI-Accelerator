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


def _get_store():
    """Get a pooled or direct PostgresStore instance."""
    try:
        from backend.api.main import _PooledPostgresStore
        return _PooledPostgresStore()
    except (ImportError, Exception):
        return PostgresStore()


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str,
                  metadata: dict | None = None) -> None:
        """Append one turn to the conversation history."""
        ...

    def update_turn_by_message_id(self, message_id: str, content: str, metadata: dict) -> None:
        """Update a specific conversation turn by its message_id in metadata."""
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

    def update_session(self, session_id: str, *,
                       title: str | None = None,
                       pinned: bool | None = None) -> None:
        """Update session metadata (custom title, pinned status)."""
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
        pg = _get_store()
        try:
            pg.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          BIGSERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    JSONB,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            # existing DBs from before `metadata` existed — additive, safe to re-run
            pg.conn.execute(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS metadata JSONB"
            )
            pg.conn.execute(
                "ALTER TABLE conversations ALTER COLUMN created_at TYPE TIMESTAMPTZ"
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations "
                "ON conversations (session_id, created_at)"
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_session_created "
                "ON conversations (session_id, created_at, id)"
            )
            pg.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_session_role_created "
                "ON conversations (session_id, role, created_at, id) "
                "WHERE role = 'user'"
            )

            # Advisory lock wrapper to serialize index check/creation across replicas.
            # pg.conn is autocommit=True by default, so CREATE INDEX CONCURRENTLY runs safely.
            pg.conn.execute("SELECT pg_advisory_lock(hashtext('idx_conversations_message_id'))")
            try:
                row = pg.conn.execute(
                    """
                    SELECT i.indisvalid FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = 'idx_conversations_message_id'
                    """
                ).fetchone()
                
                if row is None:
                    pg.conn.execute(
                        "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_message_id "
                        "ON conversations ((metadata->>'message_id'))"
                    )
                elif not row[0]: # Index exists but is INVALID
                    pg.conn.execute("DROP INDEX IF EXISTS idx_conversations_message_id")
                    pg.conn.execute(
                        "CREATE UNIQUE INDEX CONCURRENTLY idx_conversations_message_id "
                        "ON conversations ((metadata->>'message_id'))"
                    )
            finally:
                pg.conn.execute("SELECT pg_advisory_unlock(hashtext('idx_conversations_message_id'))")
            pg.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    title       TEXT,
                    pinned      BOOLEAN DEFAULT FALSE,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            # existing DBs from before TIMESTAMPTZ was adopted — safe to run if table exists
            try:
                pg.conn.execute(
                    "ALTER TABLE sessions ALTER COLUMN created_at TYPE TIMESTAMPTZ"
                )
                pg.conn.execute(
                    "ALTER TABLE sessions ALTER COLUMN updated_at TYPE TIMESTAMPTZ"
                )
            except Exception:
                pass
            # Migrations for history performance optimizations
            try:
                pg.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS active_document_id TEXT"
                )
                pg.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS first_message TEXT"
                )
                # ADDED 10-Aug: guided procedure walkthrough state, same "restart-
                # based executor, Postgres is the only thing that survives across
                # turns" reasoning as active_document_id above -- see
                # get/set_session_active_procedure.
                pg.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS active_procedure JSONB"
                )
                pg.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at "
                    "ON sessions (pinned DESC, updated_at DESC)"
                )
                # Backfill first_message for existing historical sessions
                pg.conn.execute(
                    """
                    UPDATE sessions s
                    SET first_message = LEFT(c.content, 60)
                    FROM (
                        SELECT DISTINCT ON (session_id) session_id, content
                        FROM conversations
                        WHERE role = 'user'
                        ORDER BY session_id, created_at ASC, id ASC
                    ) c
                    WHERE s.session_id = c.session_id AND s.first_message IS NULL
                    """
                )
            except Exception:
                pass
        finally:
            pg.close()

    def save_turn(self, session_id: str, role: str, content: str,
                  metadata: dict | None = None) -> None:
        pg = _get_store()
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
            # Cache the first user message on the sessions table for sidebar loading speed
            if role == "user":
                pg.conn.execute(
                    """
                    UPDATE sessions SET first_message = LEFT(%s, 60)
                    WHERE session_id = %s AND first_message IS NULL
                    """,
                    (content, session_id),
                )
        finally:
            pg.close()

    def update_turn_by_message_id(self, message_id: str, content: str, metadata: dict) -> None:
        pg = _get_store()
        try:
            pg.conn.execute(
                "UPDATE conversations SET content = %s, metadata = %s WHERE metadata->>'message_id' = %s",
                (content, _Json(metadata), message_id)
            )
        finally:
            pg.close()

    def update_session(self, session_id: str, *,
                       title: str | None = None,
                       pinned: bool | None = None) -> None:
        """Update session metadata (custom title, pinned status)."""
        pg = _get_store()
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

    def get_session_active_doc(self, session_id: str) -> str | None:
        """Fetch current active_document_id for a session from Postgres."""
        if not session_id:
            return None
        pg = _get_store()
        try:
            row = pg.conn.execute(
                "SELECT active_document_id FROM sessions WHERE session_id = %s",
                (session_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            pg.close()

    def set_session_active_doc(self, session_id: str, doc_id: str | None) -> None:
        """Atomically UPSERT current active_document_id for a session in Postgres."""
        if not session_id:
            return
        pg = _get_store()
        try:
            pg.conn.execute(
                """
                INSERT INTO sessions (session_id, active_document_id, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET active_document_id = EXCLUDED.active_document_id, updated_at = NOW()
                """,
                (session_id, doc_id),
            )
        finally:
            pg.close()

    def get_session_active_procedure(self, session_id: str) -> dict | None:
        """Fetch the in-progress guided-procedure state (backend/agent/
        procedure_tools.py) for a session, or None if no procedure is active.
        Mirrors get_session_active_doc exactly -- Postgres is the only state that
        survives across turns for this restart-based agent executor (every
        run_agent() call builds a fresh graph; conversation_history alone can't
        reliably carry "which step number are we on" for a long procedure)."""
        if not session_id:
            return None
        pg = _get_store()
        try:
            row = pg.conn.execute(
                "SELECT active_procedure FROM sessions WHERE session_id = %s",
                (session_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            pg.close()

    def set_session_active_procedure(self, session_id: str, procedure: dict | None) -> None:
        """Atomically UPSERT the active procedure state for a session. Pass None
        to clear it (procedure completed or aborted)."""
        if not session_id:
            return
        pg = _get_store()
        try:
            pg.conn.execute(
                """
                INSERT INTO sessions (session_id, active_procedure, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET active_procedure = EXCLUDED.active_procedure, updated_at = NOW()
                """,
                (session_id, _Json(procedure) if procedure is not None else None),
            )
        finally:
            pg.close()


    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        pg = _get_store()
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
        pg = _get_store()
        try:
            # High-performance index-backed query. No expensive CTE or DISTINCT ON sorting.
            rows = pg.conn.execute(
                """
                SELECT session_id, created_at AS started_at, updated_at AS last_active,
                       pinned, title AS custom_title, first_message
                FROM sessions s
                WHERE EXISTS (
                    SELECT 1 FROM conversations c WHERE c.session_id = s.session_id
                )
                ORDER BY COALESCE(pinned, FALSE) DESC, updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            sessions = []
            for session_id, started_at, last_active, pinned, custom_title, first_message in rows:
                if custom_title:
                    title = custom_title[:60]
                else:
                    title = (first_message[:60] if first_message else "New chat")
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
        pg = _get_store()
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
