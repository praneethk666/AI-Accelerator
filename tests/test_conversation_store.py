import os
import sys
import uuid

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


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


def test_list_sessions_single_query(monkeypatch):
    from backend.storage.postgres_store import PostgresStore
    store, sid = _store()
    try:
        store.save_turn(sid, "user", "Opening query")
        call_count = {"n": 0}
        original_init = PostgresStore.__init__
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            orig_execute = self.conn.execute
            def counting_execute(*a, **kw):
                call_count["n"] += 1
                return orig_execute(*a, **kw)
            self.conn.execute = counting_execute
        monkeypatch.setattr(PostgresStore, "__init__", wrapped_init)
        
        sessions = store.list_sessions()
        # Ensure only 1 query is executed inside list_sessions
        assert call_count["n"] == 1
    finally:
        _clear(sid)


def test_list_sessions_pinned_ordering():
    store, sid1 = _store()
    _, sid2 = _store()
    try:
        store.save_turn(sid1, "user", "query 1")
        store.save_turn(sid2, "user", "query 2")
        store.update_session(sid1, pinned=True)
        
        sessions = store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert ids.index(sid1) < ids.index(sid2)
    finally:
        _clear(sid1)
        _clear(sid2)


def test_list_sessions_no_user_message_falls_back_to_new_chat():
    store, sid = _store()
    try:
        store.save_turn(sid, "assistant", "system status ok")
        sessions = store.list_sessions()
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id[sid]["title"] == "New chat"
    finally:
        _clear(sid)


def test_list_sessions_custom_title_overrides_first_message():
    store, sid = _store()
    try:
        store.save_turn(sid, "user", "first question")
        store.update_session(sid, title="Custom Overridden Title")
        sessions = store.list_sessions()
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id[sid]["title"] == "Custom Overridden Title"
    finally:
        _clear(sid)


def test_save_stream_turn_upsert():
    store, sid = _store()
    msg_id = str(uuid.uuid4())
    try:
        # Save once
        store.save_stream_turn(sid, msg_id, "assistant", "partial text", {"complete": False, "message_id": msg_id})
        history1 = store.load_history(sid)
        assert len(history1) == 1
        assert history1[0]["content"] == "partial text"
        assert history1[0]["metadata"].get("complete") is False
        
        # Save again with same message_id (upsert check)
        store.save_stream_turn(sid, msg_id, "assistant", "final text", {"complete": True, "message_id": msg_id})
        history2 = store.load_history(sid)
        assert len(history2) == 1
        assert history2[0]["content"] == "final text"
        assert history2[0]["metadata"].get("complete") is True
    finally:
        _clear(sid)


def test_concurrent_message_id_upserts():
    import concurrent.futures
    store, sid = _store()
    msg_id = str(uuid.uuid4())
    
    def run_upsert(text, complete_val):
        store.save_stream_turn(sid, msg_id, "assistant", text, {"complete": complete_val, "message_id": msg_id})
        
    try:
        # Fire 10 concurrent requests with the same message_id
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(run_upsert, f"text {i}", i % 2 == 0)
                for i in range(10)
            ]
            concurrent.futures.wait(futures)
            # Ensure no exceptions were raised
            for f in futures:
                f.result()
                
        # Assert exactly one row exists in DB
        history = store.load_history(sid)
        assert len(history) == 1
    finally:
        _clear(sid)


def test_self_healing_index_repair():
    from backend.storage.postgres_store import PostgresStore
    store, sid = _store()
    pg = PostgresStore()
    try:
        # 1. Drop index manually to simulate missing/invalid state
        pg.conn.execute("DROP INDEX IF EXISTS idx_conversations_message_id")
    finally:
        pg.close()  # Close connection to allow CONCURRENT index creation without deadlocking

    # 2. Trigger schema check/repair
    store._ensure_schema()

    # 3. Assert index exists and is valid
    pg = PostgresStore()
    try:
        row = pg.conn.execute(
            """
            SELECT i.indisvalid FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname = 'idx_conversations_message_id'
            """
        ).fetchone()
        assert row is not None
        assert row[0] is True
    finally:
        pg.close()
        _clear(sid)


if __name__ == "__main__":
    if _pg_up():
        print("Starting test_save_and_load_in_chronological_order...")
        test_save_and_load_in_chronological_order()
        print("Starting test_n_limit_returns_last_n_oldest_first...")
        test_n_limit_returns_last_n_oldest_first()
        print("Starting test_sessions_are_isolated...")
        test_sessions_are_isolated()
        print("Starting test_metadata_round_trips...")
        test_metadata_round_trips()
        print("Starting test_list_sessions_orders_by_last_active_with_title...")
        test_list_sessions_orders_by_last_active_with_title()
        print("Starting test_delete_session_removes_all_its_turns...")
        test_delete_session_removes_all_its_turns()
        print("Starting test_save_stream_turn_upsert...")
        test_save_stream_turn_upsert()
        print("Starting test_concurrent_message_id_upserts...")
        test_concurrent_message_id_upserts()
        print("Starting test_self_healing_index_repair...")
        test_self_healing_index_repair()
        print("conversation store tests passed")
    else:
        print("Postgres down — skipped")
