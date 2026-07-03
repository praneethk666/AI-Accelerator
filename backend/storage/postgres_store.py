"""Postgres store.

- the `chunks` table (text + tags + source_ref) is created by scripts/init_db.sql,
  the single source of truth for the relational schema — this store does no DDL
- one schema, category lives in tags, never a separate DB per client
- write_chunk upserts; vector search lives in Qdrant
- connection comes from env (.env); never hardcode creds
"""

from __future__ import annotations

import datetime
import decimal
import json
import os

import psycopg
from psycopg.types.json import Json


def _json_default(o):
    """Make pandas/Excel cell values JSON-safe.

    Excel table_data rows come from pandas, so they can hold Timestamp/datetime,
    Decimal, or numpy scalars — none of which the stdlib JSON encoder handles.
    Without this, writing a table chunk raises and the whole row fails to store.
    """
    if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    if hasattr(o, "item"):      # numpy scalar (int64/float64/bool_)
        return o.item()
    if hasattr(o, "tolist"):    # numpy array
        return o.tolist()
    return str(o)


def _Json(value):
    """psycopg Json wrapper with the pandas/datetime-safe encoder."""
    return Json(value, dumps=lambda o: json.dumps(o, default=_json_default))


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
                _Json(chunk.get("tags", {})),
                _Json(chunk.get("source_ref")),
                _Json(chunk.get("table_data")) if chunk.get("table_data") else None,
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

    def update_progress(
        self, document_id: str, *, metrics, current_step, progress,
        total_steps=None, route=None, confidence=None,
        document_type=None, industry=None,
    ) -> None:
        """Persist live per-step progress mid-pipeline. Called after each step so the
        DB is the single source of truth the API serves (no in-memory state). COALESCE
        keeps already-known fields (e.g. route set by categorize) from being nulled by
        later steps that don't carry them."""
        self.conn.execute(
            """
            UPDATE documents SET
                metrics       = %s,
                current_step  = %s,
                progress      = %s,
                total_steps   = COALESCE(%s, total_steps),
                route         = COALESCE(%s, route),
                confidence    = COALESCE(%s, confidence),
                document_type = COALESCE(%s, document_type),
                industry      = COALESCE(%s, industry),
                updated_at    = NOW()
            WHERE document_id::text = %s
            """,
            (_Json(metrics or []), current_step, progress, total_steps, route,
             confidence, document_type, industry, document_id),
        )

    def finalize_document(
        self, document_id: str, *, document_type, industry, route, confidence,
        status, errors, metrics=None, token_usage=None, indexed_tokens=None,
        chunk_count=None,
    ) -> None:
        """Record categorization results, final aggregates, and terminal status after
        the pipeline runs — the durable end-state the API reports."""
        self.conn.execute(
            """
            UPDATE documents SET
                document_type  = %s,
                industry       = %s,
                route          = %s,
                confidence     = %s,
                status         = %s,
                errors         = %s,
                metrics        = COALESCE(%s::jsonb, metrics),
                token_usage    = %s,
                indexed_tokens = %s,
                chunk_count    = %s,
                current_step   = 'done',
                progress       = 1.0,
                updated_at     = NOW()
            WHERE document_id::text = %s
            """,
            (document_type, industry, route, confidence, status,
             _Json(errors or []),
             _Json(metrics) if metrics is not None else None,
             _Json(token_usage) if token_usage is not None else None,
             indexed_tokens, chunk_count, document_id),
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
        """Full document + live progress (the API's progress source of truth)."""
        row = self.conn.execute(
            """
            SELECT document_id, filename, file_type, document_type, industry,
                   route, confidence, status, created_at,
                   current_step, metrics, token_usage, indexed_tokens,
                   chunk_count, progress, total_steps, updated_at, errors
            FROM documents WHERE document_id::text = %s
            """,
            (document_id,),
        ).fetchone()
        if not row:
            return None
        doc = _document_row(row)
        doc.update({
            "current_step": row[9],
            "metrics": row[10] or [],
            "token_usage": row[11],
            "indexed_tokens": row[12],
            "chunk_count": row[13],
            "chunks": row[13],            # alias the UI/older callers expect
            "progress": row[14],
            "total_steps": row[15],
            "updated_at": row[16].isoformat() if row[16] else None,
            "errors": row[17] or [],
        })
        return doc

    # ── document pages (rendered full-page images for visual grounding) ─────────

    def insert_page_image(
        self, document_id: str, page: int, image_path: str,
        width=None, height=None,
    ) -> None:
        """Record one rendered page image (upsert by document_id+page)."""
        self.conn.execute(
            """
            INSERT INTO document_pages (document_id, page, image_path, width, height)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, page) DO UPDATE SET
                image_path = EXCLUDED.image_path,
                width = EXCLUDED.width, height = EXCLUDED.height
            """,
            (document_id, page, image_path, width, height),
        )

    def list_page_images(self, document_id: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT page, image_path, width, height FROM document_pages
            WHERE document_id::text = %s ORDER BY page
            """,
            (document_id,),
        ).fetchall()
        return [{"page": r[0], "image_path": r[1], "width": r[2], "height": r[3]}
                for r in rows]

    def get_page_image(self, document_id: str, page: int) -> dict | None:
        row = self.conn.execute(
            """
            SELECT page, image_path, width, height FROM document_pages
            WHERE document_id::text = %s AND page = %s
            """,
            (document_id, page),
        ).fetchone()
        return ({"page": row[0], "image_path": row[1], "width": row[2], "height": row[3]}
                if row else None)

    def delete_document(self, document_id: str) -> None:
        # chunks + document_pages cascade via the FK ON DELETE CASCADE
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
