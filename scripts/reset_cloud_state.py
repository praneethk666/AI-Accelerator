"""Reset all ingestion state for a CLEAN TEST run against whichever Postgres/
Qdrant .env actually points at — unlike reset_state.sh (which hardcodes a local
docker container name + localhost:6333), this reads POSTGRES_URL/QDRANT_URL/
QDRANT_API_KEY the same way the app itself does, so it works correctly against
the project's own cloud Supabase + Qdrant Cloud instances (set up 27-Jul,
separate from any shared/team instance).

NOTE: this is a TEST convenience, not production behavior — see reset_state.sh's
own docstring for the same caveat (every real upload gets its own document_id
and coexists; you do NOT normally wipe between uploads).

    python scripts/reset_cloud_state.py            # wipe everything
    python scripts/reset_cloud_state.py --keep-uploads   # DB only, leave local crops/pages
"""
from __future__ import annotations

import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()

# Every table but `documents`/`conversations` cascades from documents via FK —
# see scripts/init_db.sql. conversations has no FK to documents at all.
_TABLES = ["conversations", "documents"]


def reset_postgres() -> None:
    url = os.environ["POSTGRES_URL"]
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE;")
        conn.commit()
    print(f"→ Postgres: truncated {', '.join(_TABLES)} (chunks/document_blocks/"
          "llm_calls/document_pages cascade from documents)")


def reset_qdrant() -> None:
    from qdrant_client import QdrantClient
    url = os.environ["QDRANT_URL"]
    api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key)
    collection = "chunks"
    if client.collection_exists(collection):
        client.delete_collection(collection)
        print(f"→ Qdrant: dropped '{collection}' collection")
    else:
        print(f"→ Qdrant: '{collection}' collection did not exist, nothing to drop")


def clear_uploads() -> None:
    import shutil
    for sub in ("uploads/images", "uploads/pages"):
        if os.path.isdir(sub):
            for name in os.listdir(sub):
                path = os.path.join(sub, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
    print("→ uploads: cleared images/ and pages/")


if __name__ == "__main__":
    reset_postgres()
    reset_qdrant()
    if "--keep-uploads" not in sys.argv:
        clear_uploads()
    print("✓ cloud state reset")
