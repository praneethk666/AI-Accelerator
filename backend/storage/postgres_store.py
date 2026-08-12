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


def _strip_nul(obj):
    """Recursively strip NUL bytes (\x00) from dicts/lists/strings, keeping types intact.
    Postgres text fields strictly reject NUL bytes, which often sneak in from PDFs.
    """
    if isinstance(obj, str):
        return obj.replace("\x00", "").replace("\u0000", "")
    if isinstance(obj, dict):
        return {k: _strip_nul(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nul(v) for v in obj]
    return obj


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

    _migration_done: bool = False  # class-level flag — runs DDL only once per process

    def __init__(self, dsn: str | None = None, config: dict | None = None) -> None:
        if not dsn and config and isinstance(config, dict):
            from backend.core.config import get_db_url
            dsn = get_db_url(config)
        # schema is owned by scripts/init_db.sql (run at DB init); no DDL here
        self.conn = psycopg.connect(dsn or dsn_from_env(), autocommit=True, prepare_threshold=None)
        # Auto-migration: add columns/tables introduced after initial schema.
        # Guard with a class-level flag so this runs ONCE per process, not per request.
        if not PostgresStore._migration_done:
            # Helper to run a DDL statement safely without aborting on individual error
            def _safe_exec(stmt: str):
                try:
                    self.conn.execute(stmt)
                except Exception:
                    pass

            # Existing migrations
            _safe_exec("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_path TEXT")
            _safe_exec("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS session_id TEXT")
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls (session_id, created_at)")
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls (created_at DESC)")
            _safe_exec(
                """
                CREATE TABLE IF NOT EXISTS terminal_logs (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT NOW(),
                    level TEXT,
                    logger_name TEXT,
                    message TEXT,
                    exception TEXT
                )
                """
            )
            # ── P0: Document revision tracking & RBAC ─────────────────────────
            for col_ddl in [
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS revision_id UUID",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_hash TEXT",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS effective_date TIMESTAMPTZ",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_by UUID",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingestion_quality_score REAL",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS previous_chunk_count INTEGER",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS allowed_roles TEXT[]",
            ]:
                _safe_exec(col_ddl)

            # ── P0: Query audit table ──────────────────────────────────────────
            _safe_exec(
                """
                CREATE TABLE IF NOT EXISTS query_audit (
                    audit_id             UUID PRIMARY KEY,
                    session_id           TEXT,
                    query_hash           TEXT NOT NULL,
                    query_text           TEXT NOT NULL,
                    retrieved_chunk_ids  TEXT[],
                    answer_excerpt       TEXT,
                    latency_ms           REAL,
                    index_version        TEXT,
                    user_roles           TEXT[],
                    created_at           TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_query_audit_session ON query_audit (session_id, created_at DESC)")
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_query_audit_created ON query_audit (created_at DESC)")

            # ── P0: Index version registry ─────────────────────────────────────
            _safe_exec(
                """
                CREATE TABLE IF NOT EXISTS index_versions (
                    index_version   TEXT PRIMARY KEY,
                    model_name      TEXT NOT NULL,
                    config_hash     TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )

            # ── Indexes for new columns ────────────────────────────────────────
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (document_hash)")
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_documents_revision ON documents (revision_id)")
            _safe_exec("CREATE INDEX IF NOT EXISTS idx_chunks_roles ON chunks USING gin(allowed_roles) WHERE allowed_roles IS NOT NULL")

            PostgresStore._migration_done = True

    def write_chunk(self, chunk: dict) -> None:
        """Upsert one chunk row (full record), keyed by chunk_id.

        Persists table_data / image_path / allowed_roles too — table and image_caption chunks
        carry these and retrieval/citations need them back (Qdrant holds only the
        vectors + tag payload; Postgres is the source of truth for content)."""
        chunk = _strip_nul(chunk)
        self.conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, text, token_count, tags, source_ref,
                 table_data, image_path, allowed_roles)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id   = EXCLUDED.document_id,
                text          = EXCLUDED.text,
                token_count   = EXCLUDED.token_count,
                tags          = EXCLUDED.tags,
                source_ref    = EXCLUDED.source_ref,
                table_data    = EXCLUDED.table_data,
                image_path    = EXCLUDED.image_path,
                allowed_roles = EXCLUDED.allowed_roles
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
                chunk.get("allowed_roles") or None,   # None = public (no ACL)
            ),
        )

    def write_chunks(self, chunks: list[dict]) -> None:
        """Upsert multiple chunks in bulk (batch optimization)."""
        if not chunks:
            return
        chunks = _strip_nul(chunks)
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, text, token_count, tags, source_ref,
                     table_data, image_path, allowed_roles)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id   = EXCLUDED.document_id,
                    text          = EXCLUDED.text,
                    token_count   = EXCLUDED.token_count,
                    tags          = EXCLUDED.tags,
                    source_ref    = EXCLUDED.source_ref,
                    table_data    = EXCLUDED.table_data,
                    image_path    = EXCLUDED.image_path,
                    allowed_roles = EXCLUDED.allowed_roles
                """,
                [
                    (
                        c["chunk_id"],
                        c.get("document_id"),
                        c.get("text"),
                        c.get("token_count", 0),
                        _Json(c.get("tags", {})),
                        _Json(c.get("source_ref")),
                        _Json(c.get("table_data")) if c.get("table_data") else None,
                        c.get("image_path"),
                        c.get("allowed_roles") or None,
                    )
                    for c in chunks
                ],
            )

    def write_blocks(self, document_id: str, blocks: list[dict]) -> None:
        """Persist the raw extracted blocks (extractor output BEFORE chunking), in
        reading order. Lets chunking be re-run later without re-extracting, and
        gives full visibility into what was actually pulled from the document."""
        if not blocks:
            return
            
        blocks = _strip_nul(blocks)
        # psycopg3 returns a uuid.UUID object for this column elsewhere (e.g. a
        # chunk's document_id from get_chunks_by_ids) — str() defensively so a
        # caller passing that value straight through doesn't hit "operator does
        # not exist: text = uuid" (validated live, 21-Jul).
        document_id = str(document_id)
        self.conn.execute("DELETE FROM document_blocks WHERE document_id::text = %s",
                          (document_id,))
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_blocks
                    (block_id, document_id, block_order, type, text, table_data,
                     source_ref, metadata, confidence, language)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (block_id) DO NOTHING
                """,
                [
                    (
                        b.get("block_id"), document_id, i, b.get("type"), b.get("text"),
                        _Json(b.get("table_data")) if b.get("table_data") else None,
                        _Json(b.get("source_ref")),
                        _Json(b.get("metadata")) if b.get("metadata") else None,
                        b.get("confidence"), b.get("language"),
                    )
                    for i, b in enumerate(blocks)
                ],
            )

    def write_page_blocks(self, document_id: str, page_no: int, blocks: list[dict]) -> None:
        """Upsert raw extracted blocks for a specific page/slide/sheet."""
        if not blocks:
            return
            
        import json
        blocks = json.loads(json.dumps(blocks).replace("\\u0000", ""))
        document_id = str(document_id)
        # Delete existing blocks for this document and page
        self.conn.execute(
            """
            DELETE FROM document_blocks 
            WHERE document_id::text = %s 
              AND (
                source_ref->>'page' = %s 
                OR source_ref->>'slide' = %s 
                OR source_ref->>'sheet' = %s
              )
            """,
            (document_id, str(page_no), str(page_no), str(page_no)),
        )
        
        # Get the current maximum block_order for this document to append
        row = self.conn.execute(
            "SELECT COALESCE(MAX(block_order), -1) FROM document_blocks WHERE document_id::text = %s",
            (document_id,),
        ).fetchone()
        max_order = row[0] if row else -1
        
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_blocks
                    (block_id, document_id, block_order, type, text, table_data,
                     source_ref, metadata, confidence, language)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (block_id) DO NOTHING
                """,
                [
                    (
                        b.get("block_id"), document_id, max_order + 1 + i, b.get("type"), b.get("text"),
                        _Json(b.get("table_data")) if b.get("table_data") else None,
                        _Json(b.get("source_ref")),
                        _Json(b.get("metadata")) if b.get("metadata") else None,
                        b.get("confidence"), b.get("language"),
                    )
                    for i, b in enumerate(blocks)
                ],
            )

    def get_blocks(self, document_id: str) -> list[dict]:
        """Fetch a document's raw extracted blocks, in reading order."""
        document_id = str(document_id)  # see write_blocks — caller may pass a uuid.UUID
        rows = self.conn.execute(
            """
            SELECT block_id, type, text, table_data, source_ref, metadata,
                   confidence, language
            FROM document_blocks WHERE document_id::text = %s
            ORDER BY block_order
            """,
            (document_id,),
        ).fetchall()
        return [
            {
                "block_id": str(r[0]), "document_id": document_id, "type": r[1],
                "text": r[2], "table_data": r[3], "source_ref": r[4],
                "metadata": r[5], "confidence": r[6], "language": r[7],
            }
            for r in rows
        ]

    def write_llm_calls(self, document_id: str | None, calls: list[dict], session_id: str | None = None) -> None:
        """Persist the raw prompt + raw response for every LLM/vision call made
        during ingestion or agent chat into the llm_calls table."""
        if not calls:
            return
        calls = _strip_nul(calls)
        import uuid

        # Validate document_id as a valid UUID, or None
        doc_uuid = None
        if document_id:
            try:
                doc_uuid = str(uuid.UUID(str(document_id)))
            except (ValueError, AttributeError):
                doc_uuid = None

        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO llm_calls
                    (call_id, document_id, session_id, kind, provider, model, prompt,
                     raw_response, input_tokens, output_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        doc_uuid,
                        session_id or c.get("session_id"),
                        str(c.get("kind") or "unknown"),
                        c.get("provider"),
                        c.get("model"),
                        c.get("prompt") if (c.get("prompt") is None or isinstance(c.get("prompt"), str)) else json.dumps(c.get("prompt"), default=str),
                        c.get("raw_response") if (c.get("raw_response") is None or isinstance(c.get("raw_response"), str)) else json.dumps(c.get("raw_response"), default=str),
                        int(c.get("input_tokens") or 0),
                        int(c.get("output_tokens") or 0),
                    )
                    for c in calls
                ],
            )

    def get_llm_calls(
        self,
        document_id: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """Retrieve LLM/vision calls with optional filtering by document_id, session_id, or kind."""
        query = (
            "SELECT call_id, document_id, session_id, kind, provider, model, "
            "prompt, raw_response, input_tokens, output_tokens, created_at "
            "FROM llm_calls WHERE 1=1"
        )
        params: list[Any] = []
        if document_id:
            query += " AND document_id::text = %s"
            params.append(str(document_id))
        if session_id:
            query += " AND session_id = %s"
            params.append(str(session_id))
        if kind:
            query += " AND kind = %s"
            params.append(str(kind))
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "call_id": str(r[0]),
                "document_id": str(r[1]) if r[1] else None,
                "session_id": r[2],
                "kind": r[3],
                "provider": r[4],
                "model": r[5],
                "prompt": r[6],
                "raw_response": r[7],
                "input_tokens": r[8],
                "output_tokens": r[9],
                "created_at": r[10].isoformat() if r[10] else None,
            }
            for r in rows
        ]

    def write_log_batch(self, logs: list[dict]) -> None:
        """Persist a batch of application terminal logs to the database."""
        if not logs:
            return
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO terminal_logs (level, logger_name, message, exception)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (
                        r.get("level"),
                        r.get("logger_name"),
                        r.get("message"),
                        r.get("exception"),
                    )
                    for r in logs
                ]
            )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch full chunk rows by chunk_id (hydrate Qdrant search hits)."""
        if not chunk_ids:
            return []
        rows = self.conn.execute(
            # ::text cast keeps this agnostic to the chunk_id column type (uuid)
            """
            SELECT chunk_id, document_id, text, token_count, tags, source_ref,
                   table_data, image_path, allowed_roles
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
                "allowed_roles": r[8] or [],
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
        chunk_count=None, revision_id=None, document_hash=None, effective_date=None,
        ingestion_quality_score=None, previous_chunk_count=None,
    ) -> None:
        """Record categorization results, final aggregates, and terminal status after
        the pipeline runs — the durable end-state the API reports."""
        self.conn.execute(
            """
            UPDATE documents SET
                document_type          = %s,
                industry               = %s,
                route                  = %s,
                confidence             = %s,
                status                 = %s,
                errors                 = %s,
                metrics                = COALESCE(%s::jsonb, metrics),
                token_usage            = %s,
                indexed_tokens         = %s,
                chunk_count            = %s,
                current_step           = 'done',
                progress               = 1.0,
                updated_at             = NOW(),
                revision_id            = COALESCE(%s::uuid, revision_id),
                document_hash          = COALESCE(%s, document_hash),
                effective_date         = COALESCE(%s, effective_date),
                ingestion_quality_score = COALESCE(%s, ingestion_quality_score),
                previous_chunk_count   = COALESCE(%s, previous_chunk_count)
            WHERE document_id::text = %s
            """,
            (document_type, industry, route, confidence, status,
             _Json(errors or []),
             _Json(metrics) if metrics is not None else None,
             _Json(token_usage) if token_usage is not None else None,
             indexed_tokens, chunk_count,
             str(revision_id) if revision_id else None,
             document_hash,
             effective_date,
             ingestion_quality_score,
             previous_chunk_count,
             document_id),
        )

    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT document_id, filename, file_type, document_type, industry,
                   route, confidence, status, created_at, file_path,
                   current_step, metrics, token_usage, indexed_tokens,
                   chunk_count, progress, total_steps, updated_at, errors
            FROM documents ORDER BY created_at DESC
            """
        ).fetchall()
        out = []
        for row in rows:
            doc = _document_row(row)
            doc.update({
                "current_step": row[10],
                "metrics": row[11] or [],
                "token_usage": row[12],
                "indexed_tokens": row[13],
                "chunk_count": row[14],
                "chunks": row[14],            # alias the UI/older callers expect
                "progress": row[15],
                "total_steps": row[16],
                "updated_at": row[17].isoformat() if row[17] else None,
                "errors": row[18] or [],
            })
            out.append(doc)
        return out

    _doc_meta_cache: tuple[float, list[dict]] | None = None

    def list_documents_with_metadata(self, ttl_sec: int = 60) -> list[dict]:
        """List documents joined with page 1 header text for scope matching (60s TTL cache)."""
        import time
        now = time.time()
        if PostgresStore._doc_meta_cache and (now - PostgresStore._doc_meta_cache[0]) < ttl_sec:
            return PostgresStore._doc_meta_cache[1]

        rows = self.conn.execute(
            """
            SELECT d.document_id, d.filename, d.document_type, d.industry,
                   COALESCE(string_agg(b.text, ' '), '') AS page1_text
            FROM documents d
            LEFT JOIN document_blocks b ON b.document_id::text = d.document_id::text
              AND (b.source_ref->>'page' = '1' OR b.source_ref->>'slide' = '1' OR b.source_ref->>'sheet' = '1' OR b.block_order < 3)
            WHERE d.status = 'ready'
            GROUP BY d.document_id, d.filename, d.document_type, d.industry, d.created_at
            ORDER BY d.created_at DESC
            """
        ).fetchall()
        result = [
            {
                "document_id": str(r[0]),
                "filename": r[1],
                "document_type": r[2],
                "industry": r[3],
                "page1_text": r[4],
            }
            for r in rows
        ]
        PostgresStore._doc_meta_cache = (now, result)
        return result


    def get_document(self, document_id: str) -> dict | None:
        """Full document + live progress (the API's progress source of truth)."""
        row = self.conn.execute(
            """
            SELECT document_id, filename, file_type, document_type, industry,
                   route, confidence, status, created_at, file_path,
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
            "current_step": row[10],
            "metrics": row[11] or [],
            "token_usage": row[12],
            "indexed_tokens": row[13],
            "chunk_count": row[14],
            "chunks": row[14],            # alias the UI/older callers expect
            "progress": row[15],
            "total_steps": row[16],
            "updated_at": row[17].isoformat() if row[17] else None,
            "errors": row[18] or [],
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

    def delete_chunks(self, document_id: str) -> None:
        # clear a document's chunk rows without touching the document itself —
        # used to make re-ingestion idempotent (chunk_ids are regenerated per run,
        # so old rows must go before re-indexing or they'd accumulate).
        self.conn.execute(
            "DELETE FROM chunks WHERE document_id::text = %s", (document_id,)
        )

    def delete_document(self, document_id: str) -> None:
        # chunks + document_pages cascade via the FK ON DELETE CASCADE
        self.conn.execute(
            "DELETE FROM documents WHERE document_id::text = %s", (document_id,)
        )

    def document_exists(self, document_id: str) -> bool:
        if not document_id:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE document_id::text = %s", (document_id,)
        ).fetchone()
        return bool(row)

    # ── Revision tracking ───────────────────────────────────────────────────────

    def mark_superseded(self, document_id: str, superseded_by: str) -> None:
        """Mark a document revision as superseded by a newer one.

        Keeps the document row (for audit history / 'as-of' queries) but
        updates its status to 'superseded' and records who replaced it.
        The chunks are deleted separately by _preclean() in the new ingest run.
        """
        self.conn.execute(
            """
            UPDATE documents SET
                status        = 'superseded',
                superseded_at = NOW(),
                superseded_by = %s::uuid,
                updated_at    = NOW()
            WHERE document_id::text = %s
              AND status NOT IN ('superseded')
            """,
            (superseded_by, document_id),
        )

    def get_previous_chunk_count(self, document_id: str) -> int | None:
        """Return the current chunk_count before re-ingestion (for quality gate comparison)."""
        row = self.conn.execute(
            "SELECT chunk_count FROM documents WHERE document_id::text = %s",
            (document_id,),
        ).fetchone()
        return row[0] if row else None

    # ── Query audit ─────────────────────────────────────────────────────────────

    def write_query_audit(
        self,
        *,
        session_id: str | None,
        query_text: str,
        retrieved_chunk_ids: list[str],
        answer_excerpt: str | None = None,
        latency_ms: float | None = None,
        index_version: str | None = None,
        user_roles: list[str] | None = None,
    ) -> None:
        """Append one query audit record. Call after answer generation completes.

        This is the compliance record: session, query hash, which chunks were
        used, and the first 500 chars of the answer. Append-only — never updated.
        """
        import hashlib as _hashlib
        import uuid as _uuid

        query_hash = _hashlib.sha256(query_text.encode()).hexdigest()
        self.conn.execute(
            """
            INSERT INTO query_audit
                (audit_id, session_id, query_hash, query_text, retrieved_chunk_ids,
                 answer_excerpt, latency_ms, index_version, user_roles)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(_uuid.uuid4()),
                session_id,
                query_hash,
                query_text,
                [str(c) for c in (retrieved_chunk_ids or [])],
                (answer_excerpt or "")[:500],
                latency_ms,
                index_version,
                user_roles or None,
            ),
        )

    def list_query_audits(
        self,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch query audit records for compliance inspection."""
        query = (
            "SELECT audit_id, session_id, query_hash, query_text, retrieved_chunk_ids, "
            "answer_excerpt, latency_ms, index_version, user_roles, created_at "
            "FROM query_audit WHERE 1=1"
        )
        params: list = []
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "audit_id": str(r[0]),
                "session_id": r[1],
                "query_hash": r[2],
                "query_text": r[3],
                "retrieved_chunk_ids": r[4] or [],
                "answer_excerpt": r[5],
                "latency_ms": r[6],
                "index_version": r[7],
                "user_roles": r[8] or [],
                "created_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ]

    # ── Index version registry ───────────────────────────────────────────────────

    def register_index_version(
        self, index_version: str, model_name: str, config_hash: str | None = None
    ) -> None:
        """Record a new index version (model + config). Idempotent."""
        self.conn.execute(
            """
            INSERT INTO index_versions (index_version, model_name, config_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (index_version) DO NOTHING
            """,
            (index_version, model_name, config_hash),
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
        "file_path": r[9] if len(r) > 9 else None,
    }
