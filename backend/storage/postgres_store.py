"""Postgres store.

- the `chunks` table (text + tags + source_ref) is created by scripts/init_db.sql,
  the single source of truth for the relational schema — this store does no DDL
- one schema, category lives in tags, never a separate DB per client
- write_chunk upserts; vector search lives in Qdrant
- connection comes from env (.env); never hardcode creds
"""

from __future__ import annotations

import os

import psycopg
from psycopg.types.json import Json


def dsn_from_env() -> str:
    """Build a Postgres connection string from POSTGRES_* env vars."""
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'accelerator')} "
        f"user={os.getenv('POSTGRES_USER', 'accel')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'accel_local_pw')}"
    )


class PostgresStore:
    """Thin wrapper over a Postgres connection (text + tags; no vectors)."""

    def __init__(self, dsn: str | None = None) -> None:
        # schema is owned by scripts/init_db.sql (run at DB init); no DDL here
        self.conn = psycopg.connect(dsn or dsn_from_env(), autocommit=True)

    def write_chunk(self, chunk: dict) -> None:
        """Upsert one chunk row (full record), keyed by chunk_id.

        Persists table_data / image_path too — table and image_caption chunks
        carry these and retrieval/citations need them back (Qdrant holds only the
        vectors + tag payload; Postgres is the source of truth for content)."""
        self.conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, text, token_count, tags, source_ref,
                 table_data, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                text        = EXCLUDED.text,
                token_count = EXCLUDED.token_count,
                tags        = EXCLUDED.tags,
                source_ref  = EXCLUDED.source_ref,
                table_data  = EXCLUDED.table_data,
                image_path  = EXCLUDED.image_path
            """,
            (
                chunk["chunk_id"],
                chunk.get("document_id"),
                chunk.get("text"),
                chunk.get("token_count", 0),
                Json(chunk.get("tags", {})),
                Json(chunk.get("source_ref")),
                Json(chunk.get("table_data")) if chunk.get("table_data") else None,
                chunk.get("image_path"),
            ),
        )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch full chunk rows by chunk_id (hydrate Qdrant search hits)."""
        if not chunk_ids:
            return []
        rows = self.conn.execute(
            # ::text cast keeps this agnostic to the chunk_id column type (uuid)
            """
            SELECT chunk_id, document_id, text, token_count, tags, source_ref,
                   table_data, image_path
            FROM chunks WHERE chunk_id::text = ANY(%s)
            """,
            (chunk_ids,),
        ).fetchall()
        return [
            {
                "chunk_id": r[0],
                "document_id": r[1],
                "text": r[2],
                "token_count": r[3],
                "tags": r[4],
                "source_ref": r[5],
                "table_data": r[6],
                "image_path": r[7],
            }
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()
