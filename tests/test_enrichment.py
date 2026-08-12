"""EnrichChunksTool unit tests (no infra)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.enrichment.enrich_chunks import EnrichChunksTool


def test_stamps_category_tags():
    state = {
        "industry": "automotive",
        "document_type": "datasheet",
        "chunks": [{"chunk_id": "c1", "text": "torque torque spec for the engine"}],
    }
    out = EnrichChunksTool().run(state, {})
    tags = out["chunks"][0]["tags"]
    assert tags["industry"] == "automotive"
    assert tags["doc_type"] == "datasheet"
    assert "torque" in tags["keywords"]  # frequency-ranked content word


def test_does_not_overwrite_existing_tags():
    state = {
        "industry": "finance",
        "document_type": "invoice",
        "chunks": [{"chunk_id": "c1", "text": "x", "tags": {"industry": "legal"}}],
    }
    out = EnrichChunksTool().run(state, {})
    assert out["chunks"][0]["tags"]["industry"] == "legal"  # preserved


def test_handles_missing_category_and_empty_text():
    state = {"chunks": [{"chunk_id": "c1", "text": ""}]}
    out = EnrichChunksTool().run(state, {})
    assert out["chunks"][0]["tags"]["keywords"] == []


def test_run_actual_retrieval():
    import os
    import psycopg
    from dotenv import load_dotenv
    load_dotenv()
    
    out_lines = []
    def log(msg):
        print(msg)
        out_lines.append(str(msg))
        
    log("=== START TOYOPUC DIAGNOSTICS ===")
    
    postgres_url = os.getenv("POSTGRES_URL")
    log(f"POSTGRES_URL: {postgres_url.split('@')[-1] if postgres_url else None}")
    
    try:
        with psycopg.connect(postgres_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                # Query matches for TOYOPUC
                cur.execute(
                    "SELECT document_id, filename, document_type, route, status, token_usage, metrics, errors "
                    "FROM documents WHERE filename ILIKE '%TOYOPUC%'"
                )
                docs = cur.fetchall()
                log(f"Found matches: {len(docs)}")
                for d in docs:
                    log("="*60)
                    log(f"Document ID: {d[0]}")
                    log(f"Filename: {d[1]}")
                    log(f"Document Type: {d[2]}")
                    log(f"Route: {d[3]}")
                    log(f"Status: {d[4]}")
                    log(f"Token Usage: {d[5]}")
                    log(f"Metrics: {d[6]}")
                    log(f"Errors: {d[7]}")
                    
                    # Check page count or chunk count
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (d[0],))
                    chunk_cnt = cur.fetchone()[0]
                    log(f"Chunk count: {chunk_cnt}")
                    
    except Exception as e:
        log(f"Postgres Error: {e}")
        
    log("=== END TOYOPUC DIAGNOSTICS ===")
    
    # Save to file
    with open(r"c:\Users\visha\OneDrive\Desktop\AI-Accelerator-vishal-new\diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("Wrote diagnostics.txt successfully")


if __name__ == "__main__":
    test_stamps_category_tags()
    test_does_not_overwrite_existing_tags()
    test_handles_missing_category_and_empty_text()
    print("enrichment tests passed")
