"""chunk tool — NormalizedBlock[] -> Chunk[].

- the "chunk" pipeline step (config: chunking.size / overlap / strategy)
- sliding token-window split for text; every chunk <= size, sharing overlap
- headings merge into the next text block (per schemas.py)
- tables + image_captions are ATOMIC — one chunk each, never split (citations)
- skips non-content blocks (e.g. page_metrics)
- carries source_ref; sets token_count

NOTE: token count uses tiktoken if installed, else a word-count approximation.
A semantic strategy (chonkie) can swap in later via config — same in/out shape.

Run standalone on a NormalizedBlock JSON (e.g. Vishal's output):
    python -m backend.chunking.chunk_tool <blocks.json> [chunks.json]
"""

from __future__ import annotations

import uuid

from backend.core.tool import PipelineState

CONTENT_TYPES = {"text", "heading", "table", "image_caption"}
ATOMIC_TYPES = {"table", "image_caption"}  # one chunk each, never split

DEFAULT_SIZE = 400  # tokens
DEFAULT_OVERLAP = 50  # tokens


def _ntok(text: str) -> int:
    # real tokens if tiktoken present; else ~word count (documented approximation)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def _units(text: str):
    # tokenize into (encode/decode) units so windows match _ntok's counting:
    # real tokens if tiktoken present, else words. Returns (unit_list, join_fn).
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text), enc.decode
    except Exception:
        return text.split(), " ".join


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding token-window split: every chunk <= `size`, consecutive chunks
    share `overlap` units of context. (Semantic/sentence-aware via chonkie later.)"""
    text = text.strip()
    if not text:
        return []
    units, join = _units(text)
    if len(units) <= size:
        return [text]

    step = max(1, size - overlap)
    chunks: list[str] = []
    i = 0
    while i < len(units):
        piece = join(units[i : i + size])
        if isinstance(piece, str) and piece.strip():
            chunks.append(piece.strip())
        if i + size >= len(units):
            break
        i += step
    return chunks


def _make_chunk(block: dict, text: str, document_id: str | None) -> dict:
    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": block.get("document_id") or document_id,
        "text": text,
        "token_count": _ntok(text),
        "tags": {},  # categorize/enrich_chunks fill these later
        "source_ref": block.get("source_ref"),
    }
    if block.get("type") == "table":
        chunk["table_data"] = block.get("table_data")
    if block.get("type") == "image_caption":
        chunk["image_path"] = (block.get("metadata") or {}).get("image_path")
    return chunk


def chunk_blocks(
    blocks: list[dict],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    document_id: str | None = None,
) -> list[dict]:
    """Turn NormalizedBlock dicts into Chunk dicts."""
    chunks: list[dict] = []
    pending_heading = ""  # merged into the next text block
    for block in blocks:
        btype = block.get("type")
        if btype not in CONTENT_TYPES:
            continue
        text = (block.get("text") or "").strip()

        if btype == "heading":
            pending_heading = (
                (pending_heading + "\n\n" + text).strip() if pending_heading else text
            )
            continue

        if btype in ATOMIC_TYPES:  # table / image_caption -> atomic
            # an image still pending vision was never described — its text is a
            # placeholder ("[Image - awaiting vision enrichment]"), not content.
            # Don't index placeholders; vision_enrichment clears pending_vision
            # and writes the real caption on the blocks it enriches.
            if btype == "image_caption" and (block.get("metadata") or {}).get(
                "pending_vision"
            ):
                continue
            if text or block.get("table_data"):
                chunks.append(_make_chunk(block, text, document_id))
            continue

        # text block: prepend any pending heading, then split
        if pending_heading:
            text = f"{pending_heading}\n\n{text}".strip()
            pending_heading = ""
        for piece in _split_text(text, size, overlap):
            chunks.append(_make_chunk(block, piece, document_id))

    if pending_heading:  # trailing heading with no following text
        chunks.append(_make_chunk({"type": "heading"}, pending_heading, document_id))
    return chunks


class ChunkTool:
    name = "chunk"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        cfg = config.get("chunking", {})
        chunks = chunk_blocks(
            state.get("blocks", []),
            size=cfg.get("size", DEFAULT_SIZE),
            overlap=cfg.get("overlap", DEFAULT_OVERLAP),
            document_id=state.get("document_id"),
        )
        state.setdefault("chunks", []).extend(chunks)
        return state


if __name__ == "__main__":
    import json
    import sys

    blocks = json.load(open(sys.argv[1]))
    out = chunk_blocks(blocks)
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    if dest:
        json.dump(out, open(dest, "w"), indent=2)
        print(f"wrote {len(out)} chunks -> {dest}")
    else:
        print(json.dumps(out[:3], indent=2))
        print(f"... {len(out)} chunks total")
