"""Drive the FULL ingestion pipeline on one file (same entry point the API's /upload
uses), synchronously and with the extraction report + per-step metrics printed inline.

    python scripts/run_ingest.py <file_path> [document_id]

Persists to Postgres/Qdrant exactly like the API (so the UI can show it afterward).
Thin CLI over backend.pipeline.ingest.ingest_document — no recipe duplicated here.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.pipeline.ingest import ingest_document  # noqa: E402
from backend.pipeline.page_images import pages_with_chunks, save_page_images  # noqa: E402
from backend.storage.postgres_store import PostgresStore  # noqa: E402


def main(path: str, document_id: str | None) -> None:
    filename = os.path.basename(path)
    file_type = "pdf" if path.lower().endswith(".pdf") else None
    t0 = time.time()
    print(f"=== ingesting {filename}  (doc {document_id or 'auto content-hash'}) ===",
          flush=True)

    def on_step(entry, snapshot):
        ms = entry.get("ms")
        dur = f"{ms / 1000:.1f}s" if isinstance(ms, (int, float)) else "?"
        print(f"  · {entry.get('step'):<24} {entry.get('status'):<8} {dur}", flush=True)

    def on_complete(result):
        metrics = result.get("metrics", []) or []
        chunks = result.get("chunks", []) or []
        print(f"\nroute={result.get('route')}  type={result.get('document_type')}  "
              f"industry={result.get('industry')}  pdf_kind={result.get('pdf_kind')}  "
              f"chunks={len(chunks)}")

        for m in metrics:  # extraction-decision report (folded into a step's metric)
            if m.get("report"):
                print(f"\n--- extraction report ({m.get('step')}) ---")
                print(json.dumps(m["report"], indent=2, default=str))

        print("\n--- per-step timing ---")
        for m in metrics:
            ms = m.get("ms")
            d = f"{ms / 1000:6.1f}s" if isinstance(ms, (int, float)) else "    ?  "
            print(f"  {m.get('step'):<24} {m.get('status'):<8} {d}")

        tu = result.get("token_usage")
        if tu:
            print("\n--- token usage ---")
            print(json.dumps(tu, indent=2, default=str))

        img_dir = os.path.join("uploads", "images", result.get("document_id", ""))
        if os.path.isdir(img_dir):
            imgs = sorted(os.listdir(img_dir))
            print(f"\n--- {len(imgs)} figure crops in {img_dir} ---")
            for f in imgs[:60]:
                print(f"  {f}")

        # Mirrors the API's /upload on_complete tail (backend/api/main.py) -- real
        # finding, 28-Jul: this script used to skip it entirely, so document_pages
        # stayed empty (0 rows) after a real full ingestion even though the API
        # path always populates it. Same shared helper, same behavior either way.
        did = result.get("document_id")
        if did and file_type == "pdf" and result.get("status") not in ("failed", "deleted"):
            try:
                pages = pages_with_chunks(chunks)
                pg = PostgresStore()
                try:
                    for p, web, w, h in save_page_images(did, path, pages):
                        pg.insert_page_image(did, p, web, w, h)
                finally:
                    pg.close()
                print(f"\n--- saved {len(pages)} page images ---")
            except Exception as e:
                print(f"\n(page-image save failed: {e})")

    out = ingest_document(path, document_id, on_step=on_step, on_complete=on_complete)
    elapsed = time.time() - t0
    print(f"\n=== DONE in {elapsed:.1f}s — doc={out['document_id']} status={out['status']} ===")
    if out["errors"]:
        print(f"errors: {out['errors']}")


if __name__ == "__main__":
    pdf = sys.argv[1]
    did = sys.argv[2] if len(sys.argv) > 2 else None
    main(pdf, did)
