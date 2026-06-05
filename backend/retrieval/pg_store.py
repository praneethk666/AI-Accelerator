"""
backend/retrieval/pg_store.py
──────────────────────────────
PostgreSQL adapter — reads chunk metadata and full text from the
`chunks` and `documents` tables defined in scripts/init_db.sql.

Schema (from init_db.sql):
─────────────────────────────────────────────────────────────────
  chunks
    chunk_id    UUID PRIMARY KEY
    document_id UUID  → documents(document_id)
    text        TEXT NOT NULL
    token_count INTEGER
    tags        JSONB          ← GIN-indexed; holds topic/section/keywords/industry/doc_type
    source_ref  JSONB          ← filename, page, sheet, slide, bbox
    table_data  JSONB          ← non-null for table chunks
    image_path  TEXT           ← non-null for image_caption chunks
    created_at  TIMESTAMP

  documents
    document_id  UUID PRIMARY KEY
    filename     TEXT
    file_type    TEXT
    document_type TEXT
    industry     TEXT
    route        TEXT
    status       TEXT           ← "processing" | "ready" | "failed"
─────────────────────────────────────────────────────────────────

WHERE this is used in the retrieval pipeline
─────────────────────────────────────────────
1. KeywordIndex.build_from_pg()
     Loads all `ready` chunks from Postgres into BM25 at startup.
     Called once by the graph initialiser before serving queries.

2. PGStore.fetch_by_ids(chunk_ids)
     After Qdrant returns chunk UUIDs, fetches full text + metadata
     from Postgres. Qdrant stores vectors; Postgres stores payload.
     (Avoids duplicating large text blobs inside Qdrant payload.)

3. PGStore.fetch_chunks_for_scope(document_ids)
     Loads all chunks for a document_scope filter — used when
     skip_retrieval=True (adaptive_router puts the whole doc in context).

4. PGStore.log_conversation(session_id, turn, question, answer)
     Writes to the `conversations` table after the answer is generated.

Connection settings from environment (.env / docker-compose):
    POSTGRES_HOST     default "localhost"
    POSTGRES_PORT     default 5432
    POSTGRES_DB       default "accelerator"
    POSTGRES_USER     default "postgres"
    POSTGRES_PASSWORD required
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class PGStore:
    """
    Singleton Postgres connection.

    First call:  PGStore.get(config)  — reads config, opens connection
    After that:  PGStore.get()        — returns cached instance

    Config path (config/global.yaml):
        database:
          postgres_url: postgresql://username:password@localhost:5432/accelerator
          # OR individual fields (used when postgres_url is absent):
          host:     localhost
          port:     5432
          dbname:   accelerator
          username: postgres
          password: secret
    """

    _instance: Optional["PGStore"] = None

    def __init__(self, config: dict) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as e:
            raise ImportError("psycopg2 required: pip install psycopg2-binary") from e

        db_cfg = config.get("database", {})

        postgres_url = db_cfg.get("postgres_url")
        if postgres_url:
            self._conn = psycopg2.connect(postgres_url)
            logger.info("PGStore connected via postgres_url")
        else:
            host     = db_cfg.get("host",     "localhost")
            port     = int(db_cfg.get("port", 5432))
            dbname   = db_cfg.get("dbname",   "accelerator")
            username = db_cfg.get("username", "postgres")
            password = db_cfg.get("password", "")

            self._conn = psycopg2.connect(
                host    =host,
                port    =port,
                dbname  =dbname,
                user    =username,
                password=password,
            )
            logger.info("PGStore connected: %s:%s/%s", host, port, dbname)

        self._conn.autocommit = True

    @classmethod
    def get(cls, config: dict = None) -> "PGStore":
        if cls._instance is None:
            if config is None:
                raise RuntimeError(
                    "PGStore.get() called before initialisation — "
                    "pass config on the first call: PGStore.get(config)"
                )
            cls._instance = cls(config)
        return cls._instance

    # ── read operations ───────────────────────────────────────────────────────

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Fetch full chunk rows from Postgres by UUID list.

        Used after Qdrant search: Qdrant gives back chunk_ids (UUIDs),
        then we pull text + metadata from here instead of storing
        large text blobs in Qdrant payload.

        Columns read: chunk_id, document_id, text, tags, source_ref,
                      table_data, image_path
        """
        if not chunk_ids:
            return []

        import psycopg2.extras
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT chunk_id::text,
                   document_id::text,
                   text,
                   tags,
                   source_ref,
                   table_data,
                   image_path
            FROM   chunks
            WHERE  chunk_id = ANY(%s::uuid[])
            """,
            (chunk_ids,),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    def fetch_all_ready_chunks(
        self,
        document_scope: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """
        Load all chunks whose document status = 'ready'.
        Used by KeywordIndex.build_from_pg() at startup.

        document_scope: optional list of document_ids to restrict to.
        Joins `documents` to filter by status='ready'.
        """
        import psycopg2.extras
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if document_scope:
            cur.execute(
                """
                SELECT c.chunk_id::text,
                       c.document_id::text,
                       c.text,
                       c.tags,
                       c.source_ref,
                       c.table_data,
                       c.image_path
                FROM   chunks c
                JOIN   documents d ON d.document_id = c.document_id
                WHERE  d.status = 'ready'
                  AND  c.document_id = ANY(%s::uuid[])
                ORDER BY c.document_id, c.created_at
                """,
                (document_scope,),
            )
        else:
            cur.execute(
                """
                SELECT c.chunk_id::text,
                       c.document_id::text,
                       c.text,
                       c.tags,
                       c.source_ref,
                       c.table_data,
                       c.image_path
                FROM   chunks c
                JOIN   documents d ON d.document_id = c.document_id
                WHERE  d.status = 'ready'
                ORDER BY c.document_id, c.created_at
                """
            )
        rows = cur.fetchall()
        cur.close()
        logger.info("PGStore.fetch_all_ready_chunks: loaded %d chunks", len(rows))
        return [dict(row) for row in rows]

    def fetch_chunks_for_scope(self, document_ids: list[str]) -> list[Chunk]:
        """
        Fetch ALL chunks for a set of document_ids.
        Used when adaptive_router sets skip_retrieval=True — the whole
        document corpus for those docs goes directly into the LLM context.
        """
        if not document_ids:
            return []
        import psycopg2.extras
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT chunk_id::text,
                   document_id::text,
                   text,
                   tags,
                   source_ref,
                   table_data,
                   image_path
            FROM   chunks
            WHERE  document_id = ANY(%s::uuid[])
            ORDER BY created_at
            """,
            (document_ids,),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    def get_document_industry(self, document_id: str) -> Optional[str]:
        """
        Look up the industry tag for a document.
        Used by retrieval to auto-populate the `industry` filter when
        document_scope is a single document.

        Column read: documents.industry
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT industry FROM documents WHERE document_id = %s::uuid",
            (document_id,),
        )
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

    # ── write operations ──────────────────────────────────────────────────────

    def log_conversation(
        self,
        session_id: str,
        turn: int,
        question: str,
        answer: str,
    ) -> None:
        """
        Write one Q&A turn to the `conversations` table.
        Schema: (session_id UUID, turn INTEGER, question TEXT, answer TEXT)

        Called by the answer step after state["answer"] is set.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO conversations (session_id, turn, question, answer)
            VALUES (%s::uuid, %s, %s, %s)
            ON CONFLICT (session_id, turn) DO UPDATE
                SET question = EXCLUDED.question,
                    answer   = EXCLUDED.answer
            """,
            (session_id, turn, question, answer),
        )
        cur.close()
        logger.debug("PGStore.log_conversation session=%s turn=%d", session_id, turn)

