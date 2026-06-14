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
    status        TEXT DEFAULT 'processing',   -- processing | ready | failed
    errors        JSONB DEFAULT '[]',
    created_at    TIMESTAMP DEFAULT NOW()
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
