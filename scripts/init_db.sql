-- Run once on first startup (mounted via docker-entrypoint-initdb.d).
-- Creates the 'langfuse' database and the app schema in 'accelerator'.

-- ── Langfuse database ──────────────────────────────────────────────────────
-- Langfuse manages its own schema; we just need the database to exist.
CREATE DATABASE langfuse;


-- ── App schema (accelerator database) ─────────────────────────────────────
-- All tables below are created in the default 'accelerator' database.

CREATE TABLE IF NOT EXISTS documents (
    document_id   UUID PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_type     TEXT,
    file_path     TEXT,
    document_type TEXT,
    industry      TEXT,
    route         TEXT,
    confidence    REAL,                         -- categorize confidence (UI shows a bar)
    status        TEXT DEFAULT 'processing',   -- processing | ready | failed
    errors        JSONB DEFAULT '[]',
    -- live ingestion progress (DB is the single source of truth; the API reads
    -- these, survives restarts + works across workers — no in-memory state).
    current_step    TEXT,                       -- step running now (or 'done')
    metrics         JSONB DEFAULT '[]',         -- [{step, ms, status}] per step, accumulated
    token_usage     JSONB,                      -- {input_tokens, output_tokens, by_kind, ...}
    indexed_tokens  INTEGER,                     -- tokens of text indexed
    chunk_count     INTEGER,                     -- chunks produced
    progress        REAL DEFAULT 0,             -- 0..1 (completed_steps / total_steps)
    total_steps     INTEGER,
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      UUID PRIMARY KEY,
    document_id   UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    token_count   INTEGER,
    tags          JSONB,
    source_ref    JSONB,
    table_data    JSONB,        -- non-null only for table chunks
    image_path    TEXT,         -- non-null only for image_caption chunks
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Rendered full-page images, one per PDF page that produced chunks. Lets the
-- answerer/agent pull up the whole page and hand it to the vision model when a
-- retrieved chunk is ambiguous (visual grounding / "show me the source").
CREATE TABLE IF NOT EXISTS document_pages (
    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    image_path  TEXT NOT NULL,    -- served at /pages/<doc_id>/p{N}.jpg
    width       INTEGER,
    height      INTEGER,
    PRIMARY KEY (document_id, page)
);

CREATE TABLE IF NOT EXISTS conversations (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,        -- 'user' | 'assistant'
    content       TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_tags      ON chunks USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_chunks_doc       ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_img       ON chunks (image_path) WHERE image_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_tokens    ON chunks (document_id, token_count);
CREATE INDEX IF NOT EXISTS idx_conversations    ON conversations (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
