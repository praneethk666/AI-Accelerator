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

-- Raw extracted blocks (extractor output BEFORE chunking — text/heading/table/
-- image_caption in reading order). Two reasons to keep these: (1) lets chunking
-- be re-run later without re-extracting (extraction, not chunking, is the
-- expensive/slow step — docling + vision calls); (2) full visibility into what
-- was actually pulled from the document, independent of how it got chunked.
CREATE TABLE IF NOT EXISTS document_blocks (
    block_id      UUID PRIMARY KEY,
    document_id   UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    block_order   INTEGER NOT NULL,   -- position in the extracted stream (reading order)
    type          TEXT NOT NULL,      -- text | heading | table | image_caption
    text          TEXT,
    table_data    JSONB,
    source_ref    JSONB,
    metadata      JSONB,
    confidence    REAL,
    language      TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_document_blocks_doc ON document_blocks (document_id, block_order);

-- Full audit trail of every LLM/vision call made during ingestion: the exact
-- prompt sent and the RAW, unparsed response. categorize/vision/enrichment only
-- keep the fields they parsed out (document_type, caption, summary, ...) in
-- their normal tables — this is the complete record, so nothing is lost even if
-- a parser is wrong or a future need shows up that wasn't anticipated.
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id       UUID PRIMARY KEY,
    document_id   UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,      -- categorize | vision | enrichment
    provider      TEXT,
    model         TEXT,
    prompt        TEXT,
    raw_response  TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_doc ON llm_calls (document_id, kind);

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

-- Navigable chapter/section outline (backend/pipeline/outline_builder.py), from
-- either real PDF bookmarks or a detected numbered-heading stack -- lets the
-- agent locate an exact section by structure instead of trusting semantic
-- search among several similarly-worded procedures. Not every document has one
-- (a CAD sheet, a short flat document) -- that's a normal, valid, empty state.
CREATE TABLE IF NOT EXISTS document_outline (
    id           BIGSERIAL PRIMARY KEY,
    document_id  UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    node_id      TEXT NOT NULL,
    parent_id    TEXT,
    title        TEXT,
    level        INTEGER,
    page_start   INTEGER,
    page_end     INTEGER,
    source       TEXT,        -- pdf_bookmark | heading_detect
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_document_outline_doc ON document_outline (document_id, parent_id);

CREATE TABLE IF NOT EXISTS conversations (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,        -- 'user' | 'assistant'
    content       TEXT NOT NULL,
    metadata      JSONB,                -- e.g. an assistant turn's tool_calls (agent chat)
    created_at    TIMESTAMP DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_tags      ON chunks USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_chunks_doc       ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_img       ON chunks (image_path) WHERE image_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_tokens    ON chunks (document_id, token_count);
CREATE INDEX IF NOT EXISTS idx_conversations    ON conversations (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_session_created ON conversations (session_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_role_created ON conversations (session_id, role, created_at, id) WHERE role = 'user';
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);

-- ── Guardrails (v5 Tables) ──────────────────────────────────────────────────
-- Operational guardrail events log (all events)
CREATE TABLE IF NOT EXISTS guardrail_events (
    id                    BIGSERIAL PRIMARY KEY,
    ts                    TIMESTAMPTZ DEFAULT NOW(),
    session_id            TEXT,
    stage                 TEXT NOT NULL,
    event_type            TEXT NOT NULL,
    policy                TEXT NOT NULL,
    risk_score            INTEGER,
    session_cumulative    INTEGER,
    allowed               BOOLEAN NOT NULL,
    bypassed              BOOLEAN DEFAULT FALSE,
    hard_block            BOOLEAN DEFAULT FALSE,
    rule_id               TEXT,
    guardrail_version     TEXT,
    latency_ms            REAL,
    detail                JSONB
);

-- Security audit log (BLOCK and injection events only)
-- TODO (before production): REVOKE UPDATE, DELETE ON security_audit_log FROM app role
CREATE TABLE IF NOT EXISTS security_audit_log (
    id                    BIGSERIAL PRIMARY KEY,
    ts                    TIMESTAMPTZ DEFAULT NOW(),
    session_id            TEXT,
    event_type            TEXT NOT NULL,
    risk_score            INTEGER,
    rule_id               TEXT,
    guardrail_version     TEXT,
    detail                JSONB
);

CREATE INDEX IF NOT EXISTS idx_guardrail_events_session ON guardrail_events (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_guardrail_events_type    ON guardrail_events (event_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_guardrail_bypass         ON guardrail_events (bypassed, ts DESC) WHERE bypassed = TRUE;
CREATE INDEX IF NOT EXISTS idx_security_audit_session   ON security_audit_log (session_id, ts DESC);

