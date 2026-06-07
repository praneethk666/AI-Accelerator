"""Postgres + pgvector store.

- one `chunks` table holds text + tags(JSONB) + embedding(vector) — one schema,
  category lives in tags, never a separate DB per client
- write_chunk upserts; search does similarity + tag filter in one query
- connection comes from env (.env); never hardcode creds
"""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector
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
    """Thin wrapper over a pgvector-enabled Postgres connection."""

    def __init__(self, dim: int, dsn: str | None = None) -> None:
        self.dim = dim
        self.conn = psycopg.connect(dsn or dsn_from_env(), autocommit=True)
        register_vector(self.conn)  # adapt python lists <-> vector
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # dim baked into the column here -> kept config-driven, not in init.sql
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                document_id TEXT,
                text        TEXT,
                tags        JSONB,
                source_ref  JSONB,
                embedding   VECTOR({self.dim})
            )
            """)

    def write_chunk(self, chunk: dict, embedding: list[float]) -> None:
        """Upsert one chunk row (text + tags + embedding), keyed by chunk_id."""
        self.conn.execute(
            """
            INSERT INTO chunks (chunk_id, document_id, text, tags, source_ref, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                text        = EXCLUDED.text,
                tags        = EXCLUDED.tags,
                source_ref  = EXCLUDED.source_ref,
                embedding   = EXCLUDED.embedding
            """,
            (
                chunk["chunk_id"],
                chunk.get("document_id"),
                chunk.get("text"),
                Json(chunk.get("tags", {})),
                Json(chunk.get("source_ref")),
                embedding,
            ),
        )

    def search(
        self, query_embedding: list[float], filters: dict | None = None, top_n: int = 5
    ) -> list[dict]:
        """Nearest chunks by cosine distance, narrowed by exact tag matches."""
        filters = filters or {}
        where, params = "", []
        if filters:
            params = [x for k, v in filters.items() for x in (k, str(v))]
            where = "WHERE " + " AND ".join(["tags->>%s = %s"] * len(filters))
        rows = self.conn.execute(
            # ::vector cast — a list param arrives as double precision[], but <=> needs vector
            f"SELECT chunk_id, text, tags FROM chunks {where} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            [*params, query_embedding, top_n],
        ).fetchall()
        return [{"chunk_id": r[0], "text": r[1], "tags": r[2]} for r in rows]

    def close(self) -> None:
        self.conn.close()
