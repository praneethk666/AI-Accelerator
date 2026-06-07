"""Storage round-trip tests — DB/MinIO-gated.

Auto-SKIP when the stack is down, so `pytest` stays green without Docker.
To actually exercise them:  docker compose up -d  (and a local .env).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.embeddings.local_embedder import embed_text  # noqa: E402

DIM = 256


def _pg_up() -> bool:
    try:
        import psycopg

        from backend.storage.postgres_store import dsn_from_env

        psycopg.connect(dsn_from_env(), connect_timeout=2).close()
        return True
    except Exception:
        return False


def _minio_up() -> bool:
    try:
        from backend.storage.object_store import ObjectStore

        ObjectStore()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_up(), reason="Postgres not running (docker compose up -d)")
def test_chunk_round_trip_with_tag_filter():
    from backend.storage.postgres_store import PostgresStore

    store = PostgresStore(dim=DIM)
    chunk = {
        "chunk_id": "t_rt_1",
        "document_id": "d1",
        "text": "5V power rail",
        "tags": {"industry": "automotive", "topic": "power"},
        "source_ref": {"filename": "x.pdf", "page": 12},
    }
    try:
        store.write_chunk(chunk, embed_text(chunk["text"]))
        # found via matching tag filter, and tags survive the trip
        hits = store.search(
            embed_text("5V power rail"), {"industry": "automotive"}, top_n=3
        )
        assert any(h["chunk_id"] == "t_rt_1" for h in hits)
        assert (
            next(h for h in hits if h["chunk_id"] == "t_rt_1")["tags"]["topic"]
            == "power"
        )
        # excluded by a non-matching tag filter (category isolation, one schema)
        miss = store.search(
            embed_text("5V power rail"), {"industry": "finance"}, top_n=3
        )
        assert all(h["chunk_id"] != "t_rt_1" for h in miss)
    finally:
        store.conn.execute("DELETE FROM chunks WHERE chunk_id = %s", ["t_rt_1"])
        store.close()


@pytest.mark.skipif(not _minio_up(), reason="MinIO not running (docker compose up -d)")
def test_object_store_put_get():
    from backend.storage.object_store import ObjectStore

    store = ObjectStore()
    store.put("t_obj_1", b"original file bytes")
    assert store.get("t_obj_1") == b"original file bytes"


if __name__ == "__main__":
    if _pg_up():
        test_chunk_round_trip_with_tag_filter()
    if _minio_up():
        test_object_store_put_get()
    print("storage tests passed (or skipped if stack down)")
