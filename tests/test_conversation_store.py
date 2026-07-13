"""Conversation store tests — DB-gated (auto-skip when Postgres is down).

To exercise:  docker compose up -d postgres  (with .env POSTGRES_URL set).
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pg_up() -> bool:
    try:
        import psycopg

        from backend.storage.postgres_store import dsn_from_env

        psycopg.connect(dsn_from_env(), connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_up(), reason="Postgres not running (docker compose up -d postgres)"
)


def _store():
    from backend.storage.conversation_store import PostgresConversationStore

    store = PostgresConversationStore()
    store._ensure_schema()  # works even if init_db.sql wasn't run
    # session_id is TEXT, not UUID — the web UI uses "web". Use a unique non-UUID
    # id here too, proving non-UUID sessions work.
    return store, f"test-{uuid.uuid4()}"


def _clear(sid: str) -> None:
    from backend.storage.postgres_store import PostgresStore

    pg = PostgresStore()
    try:
        pg.conn.execute("DELETE FROM conversations WHERE session_id = %s", (sid,))
    finally:
        pg.close()


def test_save_and_load_in_chronological_order():
    store, sid = _store()
    try:
        store.save_turn(sid, "user", "what is the 5V rail?")
        store.save_turn(sid, "assistant", "It powers the op-amp.")
        store.save_turn(sid, "user", "and the tolerance?")
        assert store.load_history(sid) == [
            {"role": "user", "content": "what is the 5V rail?", "metadata": None},
            {"role": "assistant", "content": "It powers the op-amp.", "metadata": None},
            {"role": "user", "content": "and the tolerance?", "metadata": None},
        ]
    finally:
        _clear(sid)


def test_n_limit_returns_last_n_oldest_first():
    store, sid = _store()
    try:
        for i in range(6):  # 6 messages
            store.save_turn(sid, "user" if i % 2 == 0 else "assistant", f"m{i}")
        last4 = store.load_history(sid, n=4)
        assert [m["content"] for m in last4] == ["m2", "m3", "m4", "m5"]
    finally:
        _clear(sid)


def test_sessions_are_isolated():
    store, sid = _store()
    _, other = _store()
    try:
        store.save_turn(sid, "user", "mine")
        store.save_turn(other, "user", "theirs")
        assert store.load_history(sid) == [{"role": "user", "content": "mine", "metadata": None}]
    finally:
        _clear(sid)
        _clear(other)


def test_metadata_round_trips():
    store, sid = _store()
    try:
        store.save_turn(sid, "user", "ingest the report")
        store.save_turn(
            sid, "assistant", "done",
            metadata={"tool_calls": [{"name": "ingest_document", "args": {"file_path": "x.pdf"}}]},
        )
        history = store.load_history(sid)
        assert history[0]["metadata"] is None
        assert history[1]["metadata"] == {
            "tool_calls": [{"name": "ingest_document", "args": {"file_path": "x.pdf"}}]
        }
    finally:
        _clear(sid)


def test_list_sessions_orders_by_last_active_with_title():
    store, sid = _store()
    _, other = _store()
    try:
        store.save_turn(sid, "user", "first session's opening question, quite long indeed")
        store.save_turn(other, "user", "second session")
        sessions = store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert other in ids and sid in ids
        # most-recently-active first — `other` was written after `sid`
        assert ids.index(other) < ids.index(sid)
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id[sid]["title"].startswith("first session's opening question")
    finally:
        _clear(sid)
        _clear(other)


def test_delete_session_removes_all_its_turns():
    store, sid = _store()
    try:
        store.save_turn(sid, "user", "hello")
        store.save_turn(sid, "assistant", "hi")
        store.delete_session(sid)
        assert store.load_history(sid) == []
    finally:
        _clear(sid)


if __name__ == "__main__":
    if _pg_up():
        test_save_and_load_in_chronological_order()
        test_n_limit_returns_last_n_oldest_first()
        test_sessions_are_isolated()
        test_metadata_round_trips()
        test_list_sessions_orders_by_last_active_with_title()
        test_delete_session_removes_all_its_turns()
        print("conversation store tests passed")
    else:
        print("Postgres down — skipped")
