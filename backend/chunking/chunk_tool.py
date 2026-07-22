"""chunk tool — NormalizedBlock[] -> Chunk[].

- the "chunk" pipeline step (config: chunking.size / overlap / strategy)
- sliding token-window split for text; every chunk <= size, sharing overlap
- headings merge into the next text block (per schemas.py)
- tables + image_captions are ATOMIC — one chunk each, never split (citations)
- skips non-content blocks (e.g. page_metrics)
- carries source_ref; sets token_count

NOTE: token count uses tiktoken if installed, else a word-count approximation.
A semantic strategy (chonkie) can swap in later via config — same in/out shape.

Run standalone on a NormalizedBlock JSON (e.g. an extractor's output):
    python -m backend.chunking.chunk_tool <blocks.json> [chunks.json]
"""

from __future__ import annotations

import re
import uuid

from backend.core.tool import PipelineState

CONTENT_TYPES = {"text", "heading", "table", "image_caption"}
ATOMIC_TYPES = {"table", "image_caption"}  # one chunk each, never split

DEFAULT_SIZE = 400  # tokens
DEFAULT_OVERLAP = 50  # tokens

# Dot-leader TOC entries ("2. Specifications . . . . . . . 2-1") and dash/underscore
# fill-lines are common in real-world manuals' table of contents — extracted
# verbatim, a single cell can balloon to thousands of characters. The pattern is a
# short unit (1-3 chars, e.g. ". " or "-") REPEATED many times, not one character
# repeated (dot-leaders alternate "." and " ", so a naive same-char regex misses
# them — validated live catching this exact case). They carry zero semantic
# content but can turn one table row into an unsplittable oversized chunk:
# validated live, one such cell hit 8229 chars / 4098 tokens (a single row —
# already the smallest splittable unit) and crashed the embedder with a 64 GiB
# self-attention allocation (memory scales quadratically with sequence length).
# Real prose never repeats a 1-3 char unit 10+ times in a row.
_REPEATED_UNIT = re.compile(r"(.{1,3}?)\1{9,}")


def _collapse_repeated_chars(text: str) -> str:
    return _REPEATED_UNIT.sub(lambda m: m.group(1) * 3, text)


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


def _table_markdown_rows(headers: list, rows: list) -> str:
    """headers/rows -> a markdown pipe table (header repeated in every render)."""
    def _fmt(cells):
        return "| " + " | ".join(str(c).replace("\n", " ").strip() for c in cells) + " |"
    lines = [_fmt(headers), "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += [_fmt(r) for r in rows]
    return "\n".join(lines)


def _split_table_rows(headers: list, rows: list, size: int) -> list[list]:
    """Greedily pack table rows into row-groups whose rendered markdown (header +
    group) stays <= size tokens. One group (the whole table) if it already fits."""
    if not rows:
        return [rows]
    groups: list[list] = []
    current: list = []
    for row in rows:
        trial = current + [row]
        if current and _ntok(_table_markdown_rows(headers, trial)) > size:
            groups.append(current)
            current = [row]
        else:
            current = trial
    if current:
        groups.append(current)
    return groups or [rows]


def _split_table_block(block: dict, size: int, document_id: str | None,
                       heading_lead: str = "") -> list[dict]:
    """Table blocks are atomic (one chunk) UNLESS the table is too big for one
    chunk — then split by ROW GROUP with the header repeated in every sub-chunk.

    Why: a large spec table as a single chunk means a query about ONE row (e.g.
    "wheelhead traverse rapid feedrate") retrieves the entire table, diluting
    relevance — validated on a 37-row Toyota machine-spec table that would
    otherwise be one ~800-token chunk. Row-group chunks keep each retrievable
    unit small; the repeated header keeps every sub-chunk self-describing on its
    own (a bare row like "20 | m/min" is meaningless without its header).

    heading_lead: an orphaned heading (e.g. "2.1 Servo motor") that introduced
    this table directly, with no body text between them — prepended to the
    FIRST chunk only, so the table itself carries its own section context in
    the embedded text instead of that heading becoming its own near-empty chunk."""
    td = block.get("table_data") or {}
    raw_headers, raw_rows = td.get("headers") or [], td.get("rows") or []
    # Clean BEFORE anything else — a dot-leader cell can make a single row look
    # "large" and trigger a split that then can't actually shrink it further
    # (one row is the smallest unit); cleaning first means the token-count check
    # below sees the TRUE size, not an artifact of repeated filler characters.
    headers = [_collapse_repeated_chars(str(h)) for h in raw_headers]
    rows = [[_collapse_repeated_chars(str(c)) for c in row] for row in raw_rows]
    if headers and rows:
        text = _table_markdown_rows(headers, rows)
        block = {**block, "table_data": {"headers": headers, "rows": rows}}
    else:
        text = _collapse_repeated_chars((block.get("text") or "").strip())
    lead_text = f"{heading_lead}\n{text}".strip() if heading_lead else text

    if not headers or not rows or _ntok(text) <= size:
        return [_make_chunk(block, lead_text, document_id)] if (lead_text or td) else []

    groups = _split_table_rows(headers, rows, size)
    if len(groups) <= 1:
        return [_make_chunk(block, lead_text, document_id)] if (lead_text or td) else []

    base_ref = block.get("source_ref") or {}
    out: list[dict] = []
    row_cursor = 0
    for i, group in enumerate(groups):
        sub_ref = dict(base_ref)
        sub_ref["table_part"] = i + 1
        sub_ref["table_parts"] = len(groups)
        sub_ref["table_row_range"] = f"{row_cursor + 1}-{row_cursor + len(group)}"
        row_cursor += len(group)
        sub_block = dict(block)
        sub_block["source_ref"] = sub_ref
        sub_block["table_data"] = {"headers": headers, "rows": group}
        md = _table_markdown_rows(headers, group)
        # Defense in depth: even after collapsing repeated chars, ONE row could in
        # principle still be pathologically large (some other cause entirely) —
        # a single row can't be split further, so hard-cap it rather than risk
        # another embedder OOM. 6x the target size is generous headroom for a
        # legitimately dense row; real rows never get anywhere near this.
        if _ntok(md) > size * 6:
            units, join = _units(md)
            md = join(units[: size * 6]) + " …[truncated, oversized row]"
        if i == 0 and heading_lead:
            md = f"{heading_lead}\n{md}"
        out.append(_make_chunk(sub_block, md, document_id))
    return out


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
    split_large_tables: bool = True,
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
    section_aware is False, headings are merged inline as soft markers (legacy).

    Orphaned headings (a heading immediately followed by a table/image, ANOTHER
    heading, or nothing at all — no body text ever accumulates under it) do NOT
    become their own near-empty chunk. Validated on a real 105-page table-heavy
    manual (256 tables): headings constantly introduce a table directly with no
    intro paragraph, producing standalone one-line chunks like "3 Wiring" or
    "2.1 Servo motor" that answer nothing on their own. Instead, a chain of
    orphaned headings keeps accumulating in the SAME buffer (so "2." + "2.1 Servo
    motor" + "2.1.1 General specifications" merge into one) and gets prepended as
    LEADING TEXT to whatever real content follows. A heading with truly nothing
    after it anywhere in the document (end of blocks) is dropped.

    KNOWN REMAINING GAP (deliberately NOT covered — see PLAN.md): a short LABEL of
    real text (not a heading) immediately followed by a heading — e.g. an
    alarm-code reference page where "Alarm code\n11H" sits between "Power device
    failure" above and "State when alarm occurred" (heading) below — still gets
    flushed as its own disconnected chunk, severing the code from the table that
    explains what to do about it. A broader "any short buffer carries forward"
    rule was tried and reverted: it fixed that case but broke two others in
    testing — dropped genuinely short-but-complete standalone text ("hello world"
    with nothing after it vanished entirely), and wrongly merged unrelated
    headings into preceding short paragraphs (a real paragraph followed by an
    "Appendix" heading absorbed "Appendix" into the wrong chunk). A correct fix
    needs LOOKAHEAD — only carry short text forward if the upcoming heading
    itself turns out to be atomic-adjacent, which isn't knowable without peeking
    ahead in the block stream — bigger scope than a same-day fix."""
    chunks: list[dict] = []
    buf_parts: list[str] = []     # accumulated consecutive text, across pages
    buf_ref = None                # source_ref of the first buffered block (cite start)
    buf_page = None               # page number of the first buffered block (cite start)
    current_section = None        # active heading -> tags the section's chunks
    buf_heading_only = True       # True until real (non-heading) text is appended;
                                  # a heading-only buffer is carried forward, not emitted

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

    def pending_lead_text() -> str:
        return "\n".join(p for p in buf_parts if p).strip() if buf_heading_only else ""

    def flush():
        nonlocal buf_parts, buf_ref, buf_page, buf_heading_only
        stream = "\n".join(p for p in buf_parts if p).strip()
        if stream and not buf_heading_only:
            for piece in _split(stream, size, overlap, strategy, semantic_model):
                chunks.append(_make_chunk({"type": "text", "source_ref": _ref(buf_ref)},
                                          piece, document_id))
        buf_parts, buf_ref, buf_page, buf_heading_only = [], None, None, True

    for block in blocks:
        btype = block.get("type")
        if btype not in CONTENT_TYPES:
            continue
        text = (block.get("text") or "").strip()
        pg = _block_page(block)

        if buf_page is not None and pg is not None and pg != buf_page:
            if buf_heading_only:
                buf_page = pg  # orphaned heading carries across the page break
            else:
                flush()  # page break flushes real content; next block starts fresh

        if section_aware and btype == "heading" and text:
            if not buf_heading_only:
                flush()                              # close the previous section
            current_section = text
            if buf_ref is None:
                buf_ref = block.get("source_ref")
            buf_page = pg
            buf_parts.append(text)                   # lead the section with its title
            buf_heading_only = True
            continue

        if btype in ATOMIC_TYPES:  # table / image_caption -> atomic, breaks the stream
            heading_lead = pending_lead_text()
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
                if btype == "table" and split_large_tables:
                    chunks.extend(_split_table_block(b, size, document_id, heading_lead))
                else:
                    lead_text = f"{heading_lead}\n{text}".strip() if heading_lead else text
                    chunks.append(_make_chunk(b, lead_text, document_id))
            continue

        # text (and headings when not section_aware) -> accumulate into the stream
        if text:
            if buf_ref is None:
                buf_ref = block.get("source_ref")
            buf_parts.append(text)
            buf_heading_only = False

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
            split_large_tables=cfg.get("split_large_tables", True),
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
