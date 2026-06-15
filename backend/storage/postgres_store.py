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
    """Postgres connection string. Prefer POSTGRES_URL (matches .env.example +
    config + docker-compose); otherwise assemble from POSTGRES_* parts."""
    url = os.getenv("POSTGRES_URL")
    if url:
        return url
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'accelerator')} "
        f"user={os.getenv('POSTGRES_USER', 'postgres')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'postgres')}"
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

    # ── documents (upload metadata; backs the API's /files) ────────────────────

    def insert_document(
        self, document_id: str, filename: str, file_type: str, file_path: str
    ) -> None:
        """Create a document row in 'processing' state at upload time."""
        self.conn.execute(
            """
            INSERT INTO documents (document_id, filename, file_type, file_path, status)
            VALUES (%s, %s, %s, %s, 'processing')
            ON CONFLICT (document_id) DO NOTHING
            """,
            (document_id, filename, file_type, file_path),
        )

    def finalize_document(
        self, document_id: str, *, document_type, industry, route, confidence,
        status, errors
    ) -> None:
        """Record categorization results + final status after the pipeline runs."""
        self.conn.execute(
            """
            UPDATE documents SET document_type = %s, industry = %s, route = %s,
                                 confidence = %s, status = %s, errors = %s
            WHERE document_id = %s
            """,
            (document_type, industry, route, confidence, status,
             Json(errors or []), document_id),
        )

    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT document_id, filename, file_type, document_type, industry,
                   route, confidence, status, created_at
            FROM documents ORDER BY created_at DESC
            """
        ).fetchall()
        return [_document_row(r) for r in rows]

    def get_document(self, document_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT document_id, filename, file_type, document_type, industry,
                   route, confidence, status, created_at
            FROM documents WHERE document_id::text = %s
            """,
            (document_id,),
        ).fetchone()
        return _document_row(row) if row else None

    def delete_document(self, document_id: str) -> None:
        # chunks cascade via the FK ON DELETE CASCADE
        self.conn.execute(
            "DELETE FROM documents WHERE document_id::text = %s", (document_id,)
        )

    def close(self) -> None:
        self.conn.close()


def _document_row(r) -> dict:
    # `id` mirrors document_id so the frontend can treat it as an opaque key
    return {
        "id": str(r[0]),
        "document_id": str(r[0]),
        "filename": r[1],
        "file_type": r[2],
        "document_type": r[3],
        "industry": r[4],
        "route": r[5],
        "confidence": r[6],
        "status": r[7],
        "created_at": r[8].isoformat() if r[8] else None,
    }
