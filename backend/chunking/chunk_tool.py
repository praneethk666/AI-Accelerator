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


_SEMANTIC_CHUNKER = None
_SEMANTIC_DISABLED = False


def _get_semantic_chunker(size: int, model: str):
    """Cache one chonkie SemanticChunker. Returns None if chonkie/model unavailable
    (caller falls back to token windowing)."""
    global _SEMANTIC_CHUNKER, _SEMANTIC_DISABLED
    if _SEMANTIC_DISABLED:
        return None
    if _SEMANTIC_CHUNKER is None:
        try:
            from chonkie import SemanticChunker
            _SEMANTIC_CHUNKER = SemanticChunker(embedding_model=model, chunk_size=size)
        except Exception:
            _SEMANTIC_DISABLED = True
            return None
    return _SEMANTIC_CHUNKER


def _split(text: str, size: int, overlap: int, strategy: str, model: str) -> list[str]:
    """Split a text stream into chunks. 'semantic' uses chonkie (boundaries at topic
    shifts); anything else (or any failure) uses the token sliding window."""
    text = (text or "").strip()
    if not text:
        return []
    if strategy == "semantic":
        chunker = _get_semantic_chunker(size, model)
        if chunker is not None:
            try:
                pieces = [c.text.strip() for c in chunker(text) if getattr(c, "text", "").strip()]
                if pieces:
                    return pieces
            except Exception:
                pass  # fall through to token windowing
    return _split_text(text, size, overlap)


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
    strategy: str = "recursive",
    semantic_model: str = "minishlab/potion-base-32M",
    section_aware: bool = True,
) -> list[dict]:
    """Turn NormalizedBlock dicts into Chunk dicts.

    Consecutive text blocks are merged into one stream BEFORE splitting, so a
    procedure that flows across a page break (or a stripped header/footer) stays
    continuous instead of fragmenting at block boundaries. Tables and image captions
    are atomic — they flush the stream and emit their own chunk (citations intact).

    Layout-aware (section_aware): a HEADING starts a new section — it flushes the
    previous section and leads the next chunk's text — so chunks align to document
    sections instead of spanning several. Each chunk carries source_ref["section"]
    (the active heading), which enrich_chunks promotes to a searchable tag. When
    section_aware is False, headings are merged inline as soft markers (legacy)."""
    chunks: list[dict] = []
    buf_parts: list[str] = []     # accumulated consecutive text, across pages
    buf_ref = None                # source_ref of the first buffered block (cite start)
    buf_page = None               # page number of the first buffered block (cite start)
    current_section = None        # active heading -> tags the section's chunks

    def _block_page(block):
        ref = block.get("source_ref") or {}
        return ref.get("page") if isinstance(ref, dict) else None

    def _ref(ref):
        # attach the active section without clobbering one an extractor already set
        if not section_aware or not current_section:
            return ref
        r = dict(ref or {})
        r.setdefault("section", current_section)
        return r

    def flush():
        nonlocal buf_parts, buf_ref, buf_page
        stream = "\n".join(p for p in buf_parts if p).strip()
        if stream:
            for piece in _split(stream, size, overlap, strategy, semantic_model):
                chunks.append(_make_chunk({"type": "text", "source_ref": _ref(buf_ref)},
                                          piece, document_id))
        buf_parts, buf_ref, buf_page = [], None, None

    for block in blocks:
        btype = block.get("type")
        if btype not in CONTENT_TYPES:
            continue
        text = (block.get("text") or "").strip()
        pg = _block_page(block)

        if buf_page is not None and pg is not None and pg != buf_page:
            flush()  # page break flushes the stream; next block starts a new chunk

        if section_aware and btype == "heading" and text:
            flush()                                  # close the previous section
            current_section = text
            buf_ref = block.get("source_ref")
            buf_page = pg
            buf_parts.append(text)                   # lead the section with its title
            continue

        if btype in ATOMIC_TYPES:  # table / image_caption -> atomic, breaks the stream
            flush()
            # an image still pending vision was never described — its placeholder
            # text isn't content; vision_enrichment writes the real caption.
            if btype == "image_caption" and (block.get("metadata") or {}).get(
                "pending_vision"
            ):
                continue
            if text or block.get("table_data"):
                b = dict(block)
                b["source_ref"] = _ref(block.get("source_ref"))
                chunks.append(_make_chunk(b, text, document_id))
            continue

        # text (and headings when not section_aware) -> accumulate into the stream
        if text:
            if buf_ref is None:
                buf_ref = block.get("source_ref")
            buf_parts.append(text)

    flush()
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
            strategy=cfg.get("strategy", "recursive"),
            semantic_model=config.get("embeddings", {}).get(
                "chunking_model", "minishlab/potion-base-32M"),
            section_aware=cfg.get("section_aware", True),
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
