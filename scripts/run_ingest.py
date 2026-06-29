"""Drive the FULL ingestion pipeline on one file (same path the API's /upload uses),
but synchronously and with the extraction report + per-step metrics printed inline.

    python scripts/run_ingest.py <pdf_path> [document_id]

Persists to Postgres/Qdrant exactly like the API (so the UI can show it afterward).
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.core.config import PipelineConfig, load_config
from backend.pipeline.default_registry import build_default_registry
from backend.pipeline.graph import run_pipeline
from backend.storage.postgres_store import PostgresStore


def _ingestion_cfg(cfg: dict) -> PipelineConfig:
    return PipelineConfig.from_dict({
        **cfg,
        "steps": cfg.get("ingestion", {}).get("steps", []),
        "route_gates": cfg.get("ingestion", {}).get("route_gates", {}),
    })


def main(pdf_path: str, document_id: str) -> None:
    cfg = load_config(os.getenv("CONFIG_PATH", "config/global.yaml"))
    registry = build_default_registry()
    icfg = _ingestion_cfg(cfg)

    filename = os.path.basename(pdf_path)
    state = {"document_id": document_id, "file_path": pdf_path,
             "file_type": "pdf", "errors": []}

    pg = PostgresStore()
    try:
        pg.insert_document(document_id, filename, "pdf", pdf_path)
    except Exception as e:
        print(f"[warn] insert_document: {e}")

    def on_step(entry, snapshot):
        dur = entry.get("duration_s")
        dur = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "?"
        print(f"  · {entry.get('step'):<24} {entry.get('status'):<8} {dur}", flush=True)

    t0 = time.time()
    print(f"=== ingesting {filename}  (doc {document_id}) ===", flush=True)
    result = run_pipeline(registry, state, icfg, on_step=on_step)
    elapsed = time.time() - t0

    metrics = result.get("metrics", []) or []
    chunks = result.get("chunks", []) or []
    step_failed = any(m.get("status") == "error" for m in metrics)
    status = "failed" if step_failed else "ready"

    print(f"\n=== DONE in {elapsed:.1f}s — status={status}, chunks={len(chunks)} ===")
    print(f"route={result.get('route')}  type={result.get('document_type')}  "
          f"industry={result.get('industry')}  pdf_kind={result.get('pdf_kind')}")

    # The extraction-decision report (folded into the docling_pdf step in metrics)
    report = None
    for m in metrics:
        if m.get("report"):
            report = m["report"]
            print(f"\n--- extraction report ({m.get('step')}) ---")
            print(json.dumps(report, indent=2, default=str))

    print("\n--- per-step timing ---")
    for m in metrics:
        d = m.get("duration_s")
        d = f"{d:6.1f}s" if isinstance(d, (int, float)) else "    ?  "
        print(f"  {m.get('step'):<24} {m.get('status'):<8} {d}")

    tu = result.get("token_usage")
    if tu:
        print("\n--- token usage ---")
        print(json.dumps(tu, indent=2, default=str))

    # Finalize like the API so the UI reflects it.
    indexed_tokens = sum(int(c.get("token_count") or 0) for c in chunks)
    try:
        pg.finalize_document(
            document_id,
            document_type=result.get("document_type"),
            industry=result.get("industry"),
            route=result.get("route"),
            confidence=result.get("confidence"),
            status=status,
            errors=result.get("errors"),
            metrics=metrics,
            token_usage=tu,
            indexed_tokens=indexed_tokens,
            chunk_count=len(chunks),
        )
    except Exception as e:
        print(f"[warn] finalize_document: {e}")
    finally:
        pg.close()

    # Show what figures got cropped.
    img_dir = os.path.join("uploads", "images", document_id)
    if os.path.isdir(img_dir):
        imgs = sorted(os.listdir(img_dir))
        print(f"\n--- {len(imgs)} figure crops in {img_dir} ---")
        for f in imgs[:60]:
            print(f"  {f}")


if __name__ == "__main__":
    pdf = sys.argv[1]
    did = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())
    main(pdf, did)
