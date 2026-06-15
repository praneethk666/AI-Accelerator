"""Tests for EmbedTool using sample_chunks from fixtures."""

import pytest
from backend.embeddings.embed_tool import EmbedTool
from tests.fixtures import sample_chunks


@pytest.fixture
def embed_tool():
    return EmbedTool()


def test_embed_tool_adds_vectors(embed_tool):
    config = {
        "embeddings": {
            "dense_model": "BAAI/bge-large-en-v1.5",
            "sparse_model": "Qdrant/bm25",
        }
    }
    chunks = [c.__dict__ for c in sample_chunks()]
    state = {"chunks": chunks}
    result = embed_tool.run(state, config)
    updated_chunks = result["chunks"]

    for chunk in updated_chunks:
        assert "vector" in chunk
        assert isinstance(chunk["vector"], list)
        assert len(chunk["vector"]) == 1024
        assert "sparse_vector" in chunk
        assert isinstance(chunk["sparse_vector"], dict)
        assert "indices" in chunk["sparse_vector"]
        assert "values" in chunk["sparse_vector"]


def test_embed_tool_does_nothing_if_no_chunks(embed_tool):
    config = {"embeddings": {"dense_model": "dummy", "sparse_model": "dummy"}}
    state = {"chunks": []}
    result = embed_tool.run(state, config)
    assert result["chunks"] == []