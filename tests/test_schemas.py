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
        # un-enriched image: empty text, no table_data -> must NOT become a chunk
        NormalizedBlock(block_id="b3", document_id="d1", type="image_caption",
                        text="", metadata={"pending_vision": True}),
    ])
    chunks = chunk_blocks(blocks, size=400, overlap=50, document_id="d1")
    types_text = [c["text"] for c in chunks]
    assert any("torque" in t for t in types_text)          # text chunked
    assert any(c.get("table_data") for c in chunks)        # table carried as atomic chunk
    assert len(chunks) == 2                                 # empty image dropped


if __name__ == "__main__":
    test_as_dicts_recurses_into_source_ref()
    test_chunk_blocks_consumes_converted_dataclasses()
    print("schema/contract tests passed")
