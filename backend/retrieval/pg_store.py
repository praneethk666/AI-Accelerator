"""
backend/retrieval/pg_store.py
──────────────────────────────
PostgreSQL adapter — reads chunk metadata and full text from the
`chunks` and `documents` tables defined in scripts/init_db.sql.

config is always passed at call time — never stored during init.

Config path (config/global.yaml):
    database:
      postgres_url: postgresql://username:password@localhost:5432/accelerator
      # OR individual fields:
      host:     localhost
      port:     5432
      dbname:   accelerator
      username: postgres
      password: secret
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.schemas import Chunk

logger = logging.getLogger(__name__)


class PGStore:
    """
    Singleton Postgres connection.
    config is passed on every call — connection is established once and cached.

    First call:  PGStore.get(config)  — reads config, opens connection
    After that:  PGStore.get(config)  — returns cached connection (config ignored)
    """

    _conn = None

    @classmethod
    def get(cls, config: dict) -> "type[PGStore]":
        if cls._conn is None:
            try:
                import psycopg2
            except ImportError as e:
                raise ImportError("psycopg2 required: pip install psycopg2-binary") from e

            db_cfg = config["database"]
            postgres_url = db_cfg["postgres_url"]

            if postgres_url:
                cls._conn = psycopg2.connect(postgres_url)
                logger.info("PGStore connected via postgres_url")
            else:
                host     = db_cfg["host"]
                port     = int(db_cfg["port"])
                dbname   = db_cfg["dbname"]
                username = db_cfg["username"]
                password = db_cfg["password"]

                cls._conn = psycopg2.connect(
                    host=host,
                    port=port,
                    dbname=dbname,
                    user=username,
                    password=password,
                )
                logger.info("PGStore connected: %s:%s/%s", host, port, dbname)

            cls._conn.autocommit = True

        return cls

    # ── read operations ───────────────────────────────────────────────────────

    @classmethod
    def fetch_by_ids(cls, config: dict, chunk_ids: list[str]) -> list[Chunk]:
        """
        Fetch full chunk rows from Postgres by UUID list.
        Qdrant gives back chunk_ids; we pull text + metadata from here.
        """
        if not chunk_ids:
            return []

        cls.get(config)
        import psycopg2.extras
        cur = cls._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
        WHERE  c.chunk_id = ANY(%s::uuid[])
          AND  d.status = 'ready'
        """,
        (chunk_ids,),
    )
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    @classmethod
    def fetch_all_ready_chunks(
        cls,
        config: dict,
        document_scope: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """
        Load all chunks whose document status = 'ready'.
        Used by KeywordIndex.build_from_pg() at startup.
        """
        cls.get(config)
        import psycopg2.extras
        cur = cls._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

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

    @classmethod
    def fetch_chunks_for_scope(cls, config: dict, document_ids: list[str]) -> list[Chunk]:
        """Fetch ALL chunks for a set of document_ids."""
        if not document_ids:
            return []

        cls.get(config)
        import psycopg2.extras
        cur = cls._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
        WHERE  c.document_id = ANY(%s::uuid[])
          AND  d.status = 'ready'
        ORDER BY c.created_at
        """,
            (document_ids,),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    @classmethod
    def get_document_industry(cls, config: dict, document_id: str) -> Optional[str]:
        """Look up the industry tag for a document."""
        cls.get(config)
        cur = cls._conn.cursor()
        cur.execute(
            "SELECT industry FROM documents WHERE document_id = %s::uuid",
            (document_id,),
        )
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

    # ── write operations ──────────────────────────────────────────────────────

    @classmethod
    def log_conversation(
        cls,
        config: dict,
        session_id: str,
        turn: int,
        question: str,
        answer: str,
    ) -> None:
        """Write one Q&A turn to the `conversations` table."""
        cls.get(config)
        cur = cls._conn.cursor()
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