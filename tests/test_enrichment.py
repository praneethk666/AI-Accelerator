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


if __name__ == "__main__":
    test_stamps_category_tags()
    test_does_not_overwrite_existing_tags()
    test_handles_missing_category_and_empty_text()
    print("enrichment tests passed")
