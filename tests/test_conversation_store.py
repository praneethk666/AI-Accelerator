"""Conversation store tests — DB-gated (auto-skip when Postgres is down).

To exercise them:  docker compose up -d  (with a local .env whose POSTGRES_*
creds match the running container).
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
    not _pg_up(), reason="Postgres not running (docker compose up -d)"
)


def _fresh_store():
    from backend.storage.conversation_store import get_conversation_store

    # session_id column is UUID → use a real uuid string
    return get_conversation_store(), str(uuid.uuid4())


def test_save_turn_persists_and_loads_in_order():
    store, sid = _fresh_store()
    try:
        store.save_turn(sid, "user", "what is the 5V rail?")
        store.save_turn(sid, "assistant", "It powers the op-amp.")
        store.save_turn(sid, "user", "and the tolerance?")  # open turn, no answer yet
        history = store.load_history(sid)
        assert history == [
            {"role": "user", "content": "what is the 5V rail?"},
            {"role": "assistant", "content": "It powers the op-amp."},
            {"role": "user", "content": "and the tolerance?"},
        ]  # chronological, oldest first; open turn shows the user message only
    finally:
        store.conn.execute("DELETE FROM conversations WHERE session_id = %s", (sid,))
        store.close()


def test_n_limit_returns_last_n_turns_oldest_first():
    store, sid = _fresh_store()
    try:
        for i in range(3):  # 3 complete Q&A turns
            store.save_turn(sid, "user", f"q{i}")
            store.save_turn(sid, "assistant", f"a{i}")
        history = store.load_history(sid, n=2)  # last 2 turns
        assert history == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]  # last 2 turns, still oldest-first
    finally:
        store.conn.execute("DELETE FROM conversations WHERE session_id = %s", (sid,))
        store.close()


def test_unknown_session_returns_empty():
    store, sid = _fresh_store()
    try:
        assert store.load_history(sid) == []
    finally:
        store.close()
