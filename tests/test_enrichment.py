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


def test_repaired_chunk_summary_preserved_not_overwritten():
    # chunk_tool._repair_table_with_llm already wrote a real summary — enrich_chunks
    # must never spend a second LLM call re-summarizing text an LLM just wrote.
    state = {
        "chunks": [{
            "chunk_id": "c1",
            "text": "Factor 3: Motor failure. Action: Replace motor.",
            "tags": {"repaired": True, "chunk_type": "llm_repaired_table_row",
                     "summary": "Factor 3: Motor failure. Action: Replace motor."},
        }],
    }
    out = EnrichChunksTool().run(state, {})
    tags = out["chunks"][0]["tags"]
    assert tags["summary"] == "Factor 3: Motor failure. Action: Replace motor."
    assert "motor" in tags["keywords"]  # still gets local keywords


def test_repaired_chunks_excluded_from_llm_batch():
    from unittest.mock import MagicMock, patch

    chunks = [
        {"chunk_id": "c1", "text": "a real paragraph with enough words to be eligible for llm",
         "tags": {}},
        {"chunk_id": "c2", "text": "Factor 3: Motor failure. Action: Replace motor.",
         "tags": {"repaired": True, "summary": "Factor 3: Motor failure. Action: Replace motor."}},
    ]
    state = {"chunks": chunks}
    seen_groups = []

    def _fake_enrich_group(llm, instruction, group, usage, results):
        seen_groups.extend(i for i, _ in group)

    with patch("backend.core.llm_client.get_llm_for", return_value=MagicMock()), \
         patch("backend.enrichment.enrich_chunks._enrich_group", side_effect=_fake_enrich_group):
        EnrichChunksTool().run(state, {"enrichment": {}})

    assert seen_groups == [0]  # only the non-repaired chunk (index 0) was sent to the LLM


if __name__ == "__main__":
    test_stamps_category_tags()
    test_does_not_overwrite_existing_tags()
    test_handles_missing_category_and_empty_text()
    test_repaired_chunk_summary_preserved_not_overwritten()
    test_repaired_chunks_excluded_from_llm_batch()
    print("enrichment tests passed")
