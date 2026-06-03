-- Runs once on first DB init (empty data dir), via docker-entrypoint-initdb.d.
-- Only the extension here (dimension-independent); tables are created from code
-- in Phase 3 so the embedding dimension stays config-driven.
CREATE EXTENSION IF NOT EXISTS vector;
