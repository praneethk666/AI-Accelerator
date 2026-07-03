#!/usr/bin/env bash
# Reset all ingestion state for a CLEAN TEST run: Postgres rows, the Qdrant vector
# collection, and the local uploads (figure crops + page images).
#
# NOTE: this is a TEST convenience, not production behavior. In production every upload
# gets its own document_id and they coexist — you do NOT wipe between uploads. Use this
# only when you want to re-ingest the same file from scratch and inspect the result.
#
#   scripts/reset_state.sh          # wipe everything
#   scripts/reset_state.sh --keep-uploads   # DB only, leave crops/pages on disk
set -euo pipefail
cd "$(dirname "$0")/.."

PG_CONTAINER=${PG_CONTAINER:-ai-accelerator-postgres-1}
QDRANT_URL=${QDRANT_URL:-localhost:6333}

echo "→ Postgres: truncating chunks, document_pages, documents, conversations"
docker exec "$PG_CONTAINER" psql -U postgres -d accelerator -c \
  "TRUNCATE chunks, document_pages, documents, conversations RESTART IDENTITY CASCADE;" >/dev/null

echo "→ Qdrant: dropping 'chunks' collection"
curl -s -X DELETE "http://$QDRANT_URL/collections/chunks" >/dev/null || true

if [[ "${1:-}" != "--keep-uploads" ]]; then
  echo "→ uploads: clearing images/ and pages/"
  rm -rf uploads/images/* uploads/pages/* 2>/dev/null || true
fi

echo "✓ state reset"
