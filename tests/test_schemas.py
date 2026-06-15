"""Contract tests: blocks flow through state as plain dicts (not dataclasses).

Guards the bug where extractors emitted NormalizedBlock dataclasses but chunk/
vision/storage all access blocks as dicts (block.get(...), Json(source_ref)).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.schemas import NormalizedBlock, SourceRef, as_dicts
from backend.chunking.chunk_tool import chunk_blocks


def test_as_dicts_recurses_into_source_ref():
    block = NormalizedBlock(
        block_id="b1", document_id="d1", type="text", text="hello",
        source_ref=SourceRef(filename="x.pdf", page=3),
    )
    out = as_dicts([block])
    assert isinstance(out[0], dict)
    # nested dataclass became a plain dict too (so Json(source_ref) can serialize it)
    assert isinstance(out[0]["source_ref"], dict)
    assert out[0]["source_ref"]["page"] == 3


def test_chunk_blocks_consumes_converted_dataclasses():
    blocks = as_dicts([
        NormalizedBlock(block_id="b1", document_id="d1", type="text",
                        text="the torque spec is 50 nm for the bolt"),
        NormalizedBlock(block_id="b2", document_id="d1", type="table",
                        text="| a | b |", table_data={"headers": ["a", "b"], "rows": []}),
        # un-enriched image with a PLACEHOLDER text -> must NOT become a chunk
        # (this is the digital-PDF case: "[Image - awaiting vision enrichment]")
        NormalizedBlock(block_id="b3", document_id="d1", type="image_caption",
                        text="[Image - awaiting vision enrichment]",
                        metadata={"pending_vision": True}),
    ])
    chunks = chunk_blocks(blocks, size=400, overlap=50, document_id="d1")
    types_text = [c["text"] for c in chunks]
    assert any("torque" in t for t in types_text)          # text chunked
    assert any(c.get("table_data") for c in chunks)        # table carried as atomic chunk
    assert not any("awaiting vision" in t for t in types_text)  # placeholder NOT indexed
    assert len(chunks) == 2                                 # only text + table


def test_enriched_image_caption_is_chunked():
    # once vision clears pending_vision and writes a caption, it IS indexed
    blocks = as_dicts([
        NormalizedBlock(block_id="b1", document_id="d1", type="image_caption",
                        text="A wiring schematic showing a 5V rail and ground.",
                        metadata={"pending_vision": False,
                                  "image_path": "/images/d1/b1.png"}),
    ])
    chunks = chunk_blocks(blocks, document_id="d1")
    assert len(chunks) == 1
    assert "schematic" in chunks[0]["text"]
    assert chunks[0]["image_path"] == "/images/d1/b1.png"  # carried from metadata


if __name__ == "__main__":
    test_as_dicts_recurses_into_source_ref()
    test_chunk_blocks_consumes_converted_dataclasses()
    print("schema/contract tests passed")
