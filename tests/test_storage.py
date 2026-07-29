"""Storage round-trip tests — DB/Qdrant/MinIO-gated.

Auto-SKIP when the stack is down, so `pytest` stays green without Docker.
To actually exercise them:  docker compose up -d  (and a local .env).
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM = 256  # test vectors are fixed-dim; storage tests only need round-trip fidelity


def embed_text(text: str, dim: int = DIM) -> list[float]:
    """Deterministic fake embedding for storage round-trip tests (no model needed).

    Storage tests exercise persistence + filtering, not semantic quality, so a
    stable hash-derived unit vector is enough."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dim // len(digest)) + 1))[:dim]
    return [b / 255.0 for b in raw]


def _pg_up() -> bool:
    try:
        import psycopg

        from backend.storage.postgres_store import dsn_from_env

        psycopg.connect(dsn_from_env(), connect_timeout=2).close()
        return True
    except Exception:
        return False


def _qdrant_up() -> bool:
    try:
        from qdrant_client import QdrantClient

        from backend.storage.qdrant_store import url_from_env

        QdrantClient(url=url_from_env(), api_key=os.getenv("QDRANT_API_KEY")).get_collections()
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


@pytest.mark.skipif(
    not (_pg_up() and _qdrant_up()),
    reason="Postgres+Qdrant not running (docker compose up -d)",
)
def test_chunk_round_trips_through_both_stores():
    # the §7.5 "Done when": a tagged chunk writes to + reads back from both stores,
    # joined by the same chunk_id — text/tags from Postgres, vector + tags from Qdrant
    from backend.storage.postgres_store import PostgresStore
    from backend.storage.qdrant_store import QdrantStore

    # init_db.sql owns the schema: chunk_id/document_id are UUIDs and chunks has a
    # FK to documents, so use real UUIDs and seed the parent row first.
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    pg = PostgresStore()
    vectors = QdrantStore(DIM, collection="chunks_test")  # throwaway collection
    # Create payload index for 'industry' (required by strict Qdrant Cloud setups for filtering)
    from qdrant_client.models import PayloadSchemaType
    vectors.client.create_payload_index(
        collection_name="chunks_test",
        field_name="industry",
        field_schema=PayloadSchemaType.KEYWORD
    )
    chunk = {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "text": "5V power rail",
        "vector": embed_text("5V power rail"),
        "tags": {"industry": "automotive", "topic": "power"},
        "source_ref": {"filename": "x.pdf", "page": 12},
    }
    try:
        pg.conn.execute(
            "INSERT INTO documents (document_id, filename) VALUES (%s, %s)",
            [doc_id, "x.pdf"],
        )
        pg.write_chunk(chunk)
        vectors.write_chunk(chunk)

        # Postgres (source of truth): text + tags read back by id
        rows = pg.get_chunks_by_ids([chunk_id])
        assert len(rows) == 1
        assert rows[0]["text"] == "5V power rail"
        assert rows[0]["tags"]["topic"] == "power"

        # Qdrant: vector search finds the chunk, and its tags rode along in the payload
        hits = vectors.search(
            embed_text("5V power rail"), {"industry": "automotive"}, top_n=3
        )
        hit = next(h for h in hits if h["chunk_id"] == chunk_id)
        assert hit["tags"]["topic"] == "power"

        # excluded by a non-matching tag filter (category isolation, one schema)
        miss = vectors.search(
            embed_text("5V power rail"), {"industry": "finance"}, top_n=3
        )
        assert all(h["chunk_id"] != chunk_id for h in miss)
    finally:
        # ON DELETE CASCADE drops the chunk row along with its document
        pg.conn.execute("DELETE FROM documents WHERE document_id::text = %s", [doc_id])
        pg.close()
        vectors.client.delete_collection(vectors.collection)
        vectors.close()


@pytest.mark.skipif(not _minio_up(), reason="MinIO not running (docker compose up -d)")
def test_object_store_put_get():
    from backend.storage.object_store import ObjectStore

    store = ObjectStore()
    store.put("t_obj_1", b"original file bytes")
    assert store.get("t_obj_1") == b"original file bytes"


if __name__ == "__main__":
    if _pg_up() and _qdrant_up():
        test_chunk_round_trips_through_both_stores()
    if _minio_up():
        test_object_store_put_get()
    print("storage tests passed (or skipped if stack down)")
