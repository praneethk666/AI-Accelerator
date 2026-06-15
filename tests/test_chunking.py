"""chunk_tool tests (no infra).  Run: python tests/test_chunking.py  (or pytest)

Asserts the contract: size-splitting + overlap, heading merge, atomic
tables/captions, skipped non-content, source_ref carried, and that every
emitted chunk is a valid Chunk (schema conformance).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chunking.chunk_tool import chunk_blocks
from backend.core.schemas import Chunk


def _text_block(text, **kw):
    b = {"block_id": "b", "document_id": "d1", "type": "text", "text": text}
    b.update(kw)
    return b


def test_text_splits_by_size_with_overlap():
    # one long sentence (no punctuation) -> hard word-split into size windows
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_blocks([_text_block(text)], size=10, overlap=3)
    assert len(chunks) > 1  # actually split
    assert all(c["token_count"] <= 10 for c in chunks)  # respects size
    # consecutive chunks share words -> overlap working
    first_words = set(chunks[0]["text"].split())
    second_words = set(chunks[1]["text"].split())
    assert first_words & second_words


def test_heading_merges_into_next_text():
    blocks = [
        {"type": "heading", "text": "Section 1", "document_id": "d1"},
        _text_block("body text here"),
    ]
    chunks = chunk_blocks(blocks, size=400)
    assert len(chunks) == 1  # heading did NOT become its own chunk
    assert chunks[0]["text"].startswith("Section 1")
    assert "body text here" in chunks[0]["text"]


def test_table_is_atomic_and_keeps_table_data():
    td = {"headers": ["a"], "rows": [["1"]]}
    blocks = [{"type": "table", "text": "tbl", "table_data": td, "document_id": "d1"}]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["table_data"] == td


def test_image_caption_atomic_with_image_path():
    blocks = [
        {
            "type": "image_caption",
            "text": "a bar chart",
            "metadata": {"image_path": "uploads/x.jpg"},
            "document_id": "d1",
        }
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["image_path"] == "uploads/x.jpg"


def test_non_content_blocks_skipped():
    blocks = [{"type": "page_metrics", "text": "ignore me", "document_id": "d1"}]
    assert chunk_blocks(blocks) == []


def test_empty_text_produces_no_chunk():
    assert chunk_blocks([_text_block("   ")]) == []


def test_source_ref_carried_through():
    src = {"filename": "x.pdf", "page": 12}
    chunks = chunk_blocks([_text_block("hello world", source_ref=src)])
    assert chunks[0]["source_ref"] == src


def test_every_chunk_is_a_valid_chunk_schema():
    blocks = [
        {"type": "heading", "text": "H", "document_id": "d1"},
        _text_block("some body text", source_ref={"filename": "x.pdf", "page": 1}),
        {"type": "table", "text": "t", "table_data": {"headers": [], "rows": []}, "document_id": "d1"},
    ]
    for c in chunk_blocks(blocks):
        Chunk(**c)  # raises TypeError if any key is not a valid Chunk field


if __name__ == "__main__":
    test_text_splits_by_size_with_overlap()
    test_heading_merges_into_next_text()
    test_table_is_atomic_and_keeps_table_data()
    test_image_caption_atomic_with_image_path()
    test_non_content_blocks_skipped()
    test_empty_text_produces_no_chunk()
    test_source_ref_carried_through()
    test_every_chunk_is_a_valid_chunk_schema()
    print("chunking tests passed")
