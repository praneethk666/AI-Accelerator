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

import os
import sys
import re
import uuid
import json
import logging
from typing import Any

from backend.core.tool import PipelineState
from backend.core.llm_client import get_llm_for
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

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


import re


def _clean_str(val: str) -> str:
    if not val:
        return ""
    return " ".join(str(val).replace("\n", " ").split())


def _extract_grid_from_block(block: dict) -> list[list[str]]:
    # 1. table_data dict
    td = block.get("table_data")
    if isinstance(td, dict):
        if "grid" in td and isinstance(td["grid"], list) and td["grid"]:
            return [[_clean_str(c) for c in r] for r in td["grid"]]
        if "headers" in td and "rows" in td and td["headers"] and td["rows"]:
            headers = [_clean_str(c) for c in td["headers"]]
            rows = [[_clean_str(c) for c in r] for r in td["rows"]]
            return [headers] + rows
        if "rows" in td and isinstance(td["rows"], list) and td["rows"]:
            return [[_clean_str(c) for c in r] for r in td["rows"]]

    # 2. direct grid key
    grid = block.get("grid")
    if isinstance(grid, list) and grid:
        return [[_clean_str(c) for c in r] for r in grid]

    # 3. markdown text parsing
    text = block.get("text") or ""
    if "|" in text:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        parsed_grid = []
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                if re.match(r"^\|[\s\:\-]+\|$", line):
                    continue
                cells = [_clean_str(c) for c in line.split("|")[1:-1]]
                if any(cells):
                    parsed_grid.append(cells)
        if parsed_grid:
            return parsed_grid
    return []


def _is_diagram_grid(grid: list[list[str]]) -> bool:
    """Detect if a table grid is a pinout/wiring diagram schematic misdetected as a table.
    Diagram grids contain mostly single-character or single-digit cell noise."""
    if not grid or len(grid) < 2:
        return False

    all_cells = [str(c).strip() for row in grid for c in row if str(c).strip()]
    if len(all_cells) < 6:
        return False

    single_char_count = sum(1 for c in all_cells if len(c) <= 2 or re.match(r"^[\d\W]+$", c))
    return (single_char_count / len(all_cells)) > 0.45


def _try_extract_troubleshooting_table_chunks(block: dict, document_id: str | None, ref_fn) -> list[dict] | None:
    """If a table represents a Troubleshooting / Fault Cause & Action list,
    extract 1 self-contained row-wise chunk per fault linking Cause and Action together."""
    grid = _extract_grid_from_block(block)
    if not grid or len(grid) < 2 or _is_diagram_grid(grid):
        return None

    headers = [str(h).lower() for h in grid[0]]
    trouble_keywords = ["cause", "action", "factor", "remedy", "countermeasure", "symptom", "corrective action", "investigation"]
    if not any(kw in h for h in headers for kw in trouble_keywords):
        return None

    source_ref = ref_fn(block.get("source_ref"))
    trouble_chunks = []

    for row in grid[1:]:
        if not row or not any(row):
            continue
        row_id = str(row[0]).strip()

        if not row_id:
            # Continuation row (wrapped cell text) — merge into the previous troubleshooting chunk
            # instead of dropping it.
            if not trouble_chunks:
                continue
            extra_details = []
            for i in range(1, min(len(row), len(grid[0]))):
                hdr = grid[0][i]
                val = row[i]
                if val:
                    extra_details.append(f"{hdr}: {val}")
            if extra_details:
                prev = trouble_chunks[-1]
                prev["text"] = prev["text"] + " | " + " | ".join(extra_details)
                prev["token_count"] = _ntok(prev["text"])
            continue

        row_details = []
        for i in range(1, min(len(row), len(grid[0]))):
            hdr = grid[0][i]
            val = row[i]
            if val:
                row_details.append(f"{hdr}: {val}")

        details_str = " | ".join(row_details)
        row_text = f"[Troubleshooting: {row_id}] {details_str}"

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id") or document_id,
            "text": row_text,
            "token_count": _ntok(row_text),
            "table_data": {"headers": grid[0], "row": row},
            "source_ref": source_ref,
            "tags": {
                "document_type": "manual",
                "chunk_type": "troubleshooting_row",
                "fault_name": row_id,
                "has_table": True,
            },
        }
        trouble_chunks.append(chunk)

    return trouble_chunks if trouble_chunks else None


def _try_extract_alarm_table_chunks(block: dict, document_id: str | None, ref_fn) -> list[dict] | None:
    """If a block or table represents an Alarm/Error list (Section 5.2 or alarm codes),
    extract 1 self-contained chunk per row with explicit row-identity anchors."""
    grid = _extract_grid_from_block(block)
    if not grid or len(grid) < 2:
        return None

    headers = [str(h).lower() for h in grid[0]]
    if not any("alarm" in h or "error" in h or "code" in h or "display" in h for h in headers):
        return None

    source_ref = ref_fn(block.get("source_ref"))
    alarm_chunks = []

    for row in grid[1:]:
        if not row or not any(row):
            continue
        row_id = str(row[0]).strip()

        if not row_id:
            # Continuation row (wrapped cell text) — merge into the previous alarm chunk
            # instead of dropping it.
            if not alarm_chunks:
                continue
            extra_details = []
            for i in range(1, min(len(row), len(grid[0]))):
                hdr = grid[0][i]
                val = row[i]
                if val:
                    extra_details.append(f"{hdr}: {val}")
            if extra_details:
                prev = alarm_chunks[-1]
                prev["text"] = prev["text"] + " | " + " | ".join(extra_details)
                prev["token_count"] = _ntok(prev["text"])
            continue

        row_details = []
        for i in range(1, min(len(row), len(grid[0]))):
            hdr = grid[0][i]
            val = row[i]
            if val:
                row_details.append(f"{hdr}: {val}")

        details_str = " | ".join(row_details)
        row_text = f"[Alarm: {row_id}] {details_str}"

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id") or document_id,
            "text": row_text,
            "token_count": _ntok(row_text),
            "table_data": {"headers": grid[0], "row": row},
            "source_ref": source_ref,
            "tags": {
                "document_type": "manual",
                "chunk_type": "alarm_row",
                "alarm_name": row_id,
                "has_table": True,
            },
        }
        alarm_chunks.append(chunk)

    return alarm_chunks if alarm_chunks else None


def _try_extract_warning_chunk(block: dict, document_id: str | None, ref_fn) -> list[dict] | dict | None:
    """If a block or table represents a safety warning/danger callout, format it
    into a high-priority warning chunk with Severity, Instruction, and Consequence."""
    text = (block.get("text") or "").strip()
    grid = _extract_grid_from_block(block)
    grid_str = " ".join(" ".join(r) for r in grid) if grid else ""
    full_text = f"{text} {grid_str}"

    warning_match = re.search(r"\b(DANGER|WARNING|CAUTION|NOTICE|MANDATORY|PROHIBITED)\b", full_text, re.IGNORECASE)
    if not warning_match:
        return None

    severity = warning_match.group(1).capitalize()
    source_ref = ref_fn(block.get("source_ref"))

    if grid and len(grid) >= 2:
        # Emit 1 chunk per legend row instead of smashing all rows into one run-on string
        warning_chunks = []
        for r in grid[1:]:
            r_str = " | ".join(c for c in r if c).strip()
            if not r_str:
                continue
            formatted_text = f"Warning Type: {severity}\nText: {r_str}"
            warning_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": block.get("document_id") or document_id,
                "text": formatted_text,
                "token_count": _ntok(formatted_text),
                "source_ref": source_ref,
                "tags": {
                    "document_type": "manual",
                    "chunk_type": "warning",
                    "severity": severity,
                },
            })
        if warning_chunks:
            return warning_chunks

    formatted_text = f"Warning Type: {severity}\nText: {text}"
    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": block.get("document_id") or document_id,
        "text": formatted_text,
        "token_count": _ntok(formatted_text),
        "source_ref": source_ref,
        "tags": {
            "document_type": "manual",
            "chunk_type": "warning",
            "severity": severity,
        },
    }
    return chunk


def _try_extract_cad_title_chunk(block: dict, document_id: str | None, ref_fn) -> dict | None:
    """If a block represents a CAD title block, unify drawing number, company, approvals,
    and metadata into 1 high-density CAD Title Block Chunk."""
    text = (block.get("text") or "").strip()
    if not ("drawing no" in text.lower() or "title block" in text.lower()):
        return None

    source_ref = ref_fn(block.get("source_ref"))
    chunk_text = f"# CAD Drawing Title Block & Specifications\n\n{text}"

    # Extract drawing number and company for metadata filtering. Both are read from
    # the block's own text — never hardcode a specific company name here, the same
    # extractor runs over drawings from any client.
    drawing_no = ""
    drawing_match = re.search(r"Drawing No\*?:\s*([A-Za-z0-9\-\_]+)", text, re.IGNORECASE)
    if drawing_match:
        drawing_no = drawing_match.group(1)

    company = ""
    company_match = re.search(
        r"([A-Z][A-Za-z0-9&.,\- ]*(?:CORPORATION|CORP\.?|CO\.,?\s*LTD\.?|INC\.?))",
        text,
    )
    if company_match:
        company = company_match.group(1).strip()

    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": block.get("document_id") or document_id,
        "text": chunk_text,
        "token_count": _ntok(chunk_text),
        "source_ref": source_ref,
        "tags": {
            "document_type": "cad",
            "chunk_type": "cad_title_block",
            "drawing_number": drawing_no,
            "company": company,
            "has_table": False,
        },
    }
    return chunk


def _try_extract_cad_component_chunks(block: dict, document_id: str | None, ref_fn) -> list[dict] | None:
    """If a block or table represents CAD component specifications (Steel Pipe, Hose, Copper Pipe, Tube),
    unify the matrix into 1 complete self-contained table chunk per component type."""
    text = (block.get("text") or "").strip()
    grid = _extract_grid_from_block(block)
    full_str = f"{text} " + (" ".join(" ".join(r) for r in grid) if grid else "")

    component_type = None
    if re.search(r"\b(steel pipe|pipe specifications)\b", full_str, re.IGNORECASE) or re.search(r"\bS\d+ø\b", full_str):
        component_type = "steel_pipe"
    elif re.search(r"\b(hose|hose specifications)\b", full_str, re.IGNORECASE) or re.search(r"\bH\d+\(\d+/\d+\)\b", full_str):
        component_type = "hose"
    elif re.search(r"\b(copper pipe)\b", full_str, re.IGNORECASE) or re.search(r"\bC\d+ø\b", full_str):
        component_type = "copper_pipe"
    elif re.search(r"\b(tube|vinyl tube)\b", full_str, re.IGNORECASE):
        component_type = "tube"

    if not component_type:
        return None

    source_ref = ref_fn(block.get("source_ref"))
    comp_title = component_type.replace("_", " ").title()

    if grid and len(grid) >= 2:
        headers = grid[0]
        rows = grid[1:]
        md_table = _rebuild_markdown(headers, rows)
        chunk_text = f"# Material Specifications - {comp_title}\n\n{md_table}"
    else:
        chunk_text = f"# Material Specifications - {comp_title}\n\n{text}"

    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": block.get("document_id") or document_id,
        "text": chunk_text,
        "token_count": _ntok(chunk_text),
        "source_ref": source_ref,
        "tags": {
            "document_type": "cad",
            "chunk_type": "cad_component_table",
            "component": component_type,
            "has_table": True,
        },
    }
    return [chunk]


def _rebuild_markdown(headers: list, rows: list[list]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _try_extract_model_column_chunks(block: dict, document_id: str | None, ref_fn) -> list[dict] | None:
    """If a table block contains a multi-column model specification matrix, split it
    into one self-contained chunk per model column (e.g. Q2AA 07040D, 08100D, etc.).
    Returns None if the table is not a model column specification matrix.
    """
    grid = _extract_grid_from_block(block)
    if not grid or len(grid) < 2 or _is_diagram_grid(grid):
        return None

    header_row = grid[0]
    if len(header_row) < 3:
        return None

    # Gate 1: Ensure headers don't contain troubleshooting keywords
    headers_lower = [str(h).lower() for h in header_row]
    if any(kw in h for h in headers_lower for kw in ["cause", "action", "factor", "remedy"]):
        return None

    # Gate 2: Check if header columns contain real model codes or alphanumeric series
    model_headers_count = sum(1 for h in header_row[1:] if re.search(r"[A-Z0-9]{4,}", str(h), re.IGNORECASE) or re.search(r"\d{3,}", str(h)))
    if model_headers_count == 0:
        return None

    # Gate 3: Check if table contains technical units
    full_str = " ".join(" ".join(str(c) for c in r) for r in grid).lower()
    units_count = sum(1 for u in ["kw", "n・m", "n.m", "min-1", "rpm", "v", "a", "kg", "mm", "hz"] if u in full_str)
    if units_count == 0:
        return None

    model_start_col = -1
    for col_i in range(1, len(header_row)):
        cell_val = _clean_str(header_row[col_i]).lower()
        if cell_val and cell_val not in ["symbol", "unit", "pr", "nr"]:
            model_start_col = col_i
            break

    if model_start_col == -1 or model_start_col >= len(header_row):
        return None

    series_info = _clean_str(header_row[0]) if header_row[0] else "Specification Table"
    param_rows = grid[1:]
    source_ref = ref_fn(block.get("source_ref"))
    filename = (source_ref.get("filename") if isinstance(source_ref, dict) else None) or "document"

    col_chunks = []
    for col_idx in range(model_start_col, len(header_row)):
        model_name = _clean_str(header_row[col_idx])
        if not model_name and len(grid) > 1 and len(grid[1]) > col_idx:
            model_name = _clean_str(grid[1][col_idx])
        if not model_name:
            model_name = f"Model_Col_{col_idx}"

        text_lines = [
            f"Document Source: {filename}",
            f"Series: {series_info}",
            f"Model: {model_name}",
            "--- Specifications ---"
        ]

        for row in param_rows:
            if not row:
                continue
            param_name = _clean_str(row[0]) if len(row) > 0 else ""
            symbol = _clean_str(row[1]) if len(row) > 1 else ""
            unit = _clean_str(row[2]) if len(row) > 2 else ""
            val = _clean_str(row[col_idx]) if len(row) > col_idx else ""

            if not param_name and not val:
                continue

            key_label = param_name
            if symbol and symbol != param_name:
                key_label += f" ({symbol})"

            val_str = val
            if unit and val_str and not val_str.endswith(unit):
                val_str += f" {unit}"

            text_lines.append(f"- {key_label}: {val_str}")

        chunk_text = "\n".join(text_lines)

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id") or document_id,
            "text": chunk_text,
            "token_count": _ntok(chunk_text),
            "table_data": block.get("table_data"),
            "source_ref": source_ref,
            "tags": {
                "doc_type": "specifications",
                "model_name": model_name,
                "series": series_info,
            },
        }
        col_chunks.append(chunk)

    return col_chunks if col_chunks else None


def normalize_text(text: str) -> str:
    # Convert fullwidth unicode alphanumeric characters to normal ASCII
    return text.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    ))

# Programmatic table check helpers
def has_header_anomaly(headers: list[str]) -> bool:
    non_empty = [h for h in headers if h.strip()]
    has_dupes = len(non_empty) != len(set(non_empty))
    has_blanks = any(h.strip() == "" for h in headers)
    return has_dupes or has_blanks

def has_ragged_rows(rows: list[list[str]]) -> bool:
    col_counts = {len(r) for r in rows}
    return len(col_counts) > 1

def is_orphan_table(rows: list[list[str]]) -> bool:
    return len(rows) <= 2

def lacks_context_anchor(table_data: dict, section_lead: str, preceding_text: str) -> bool:
    headers = table_data.get("headers") or []
    rows = table_data.get("rows") or []
    combined_text = normalize_text(" ".join(
        [str(h) for h in headers] + [str(c) for r in rows for c in r]
    )).lower()
    
    # Normalize context and extract candidates
    context = normalize_text(preceding_text + " " + section_lead)
    candidates = re.findall(r"\b\d+[a-zA-Z]\b", context)
    if not candidates:
        return False
    return not any(c.lower() in combined_text for c in candidates)

def classify_table(table_block: dict, section_lead: str, preceding_text: str) -> dict:
    td = table_block.get("table_data") or {}
    headers = td.get("headers") or []
    rows = td.get("rows") or []
    
    reasons = []
    if has_header_anomaly(headers):
        reasons.append("header_anomaly")
    if has_ragged_rows(rows):
        reasons.append("ragged_rows")
    if is_orphan_table(rows):
        reasons.append("orphan_table")
    if lacks_context_anchor(td, section_lead, preceding_text):
        reasons.append("missing_context_anchor")
        
    return {"needs_llm": len(reasons) > 0, "reasons": reasons}

def get_heading_level(text: str, bbox: list[float] | None) -> int:
    """Infer heading level dynamically using regex numbering or visual bbox characteristics."""
    # 1. Check for standard numbering pattern (e.g. 5.2, 5.3.1)
    num_match = re.match(r"^(\d+(?:\.\d+)*)\b", text.strip())
    if num_match:
        dots = num_match.group(1).count(".")
        return dots + 1  # 5 -> level 1, 5.3 -> level 2, 5.3.1 -> level 3
        
    # 2. Bounding box fallback
    if bbox and len(bbox) == 4:
        height = bbox[3] - bbox[1]
        indent = bbox[0]
        # Larger font height -> higher level
        if height >= 12.0:
            return 1
        # Smaller headers: separate by indentation
        if indent >= 75.0:
            return 3
        return 2
        
    return 2  # default level

def validate_repaired_chunks(parsed_chunks: list[dict], original_rows: list[list[str]], headers: list[str], section_lead: str, preceding_text: str) -> str | None:
    """Validate repaired chunks for row count matching and content traceability. Returns error string if invalid."""
    # 1. Row count validation
    if len(parsed_chunks) != len(original_rows):
        return f"Row count mismatch: LLM returned {len(parsed_chunks)} rows, expected {len(original_rows)}"
        
    # Normalize inputs
    norm_rows = [[normalize_text(c).lower().strip() for c in r] for r in original_rows]
    norm_section_lead = normalize_text(section_lead).lower()
    norm_preceding_text = normalize_text(preceding_text).lower()
    
    # 2. Content traceability validation
    source_words = set(re.findall(r"\w+", " ".join(" ".join(r) for r in norm_rows)))
    context_words = set(re.findall(r"\w+", norm_section_lead + " " + norm_preceding_text))
    allowed_words = source_words | context_words
    
    original_cell_texts = [c for r in norm_rows for c in r if c]
    
    # Comprehensive stop words
    stops = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
        "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", 
        "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", 
        "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", 
        "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", 
        "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", 
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
        "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", 
        "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", 
        "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", 
        "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", 
        "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", 
        "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", 
        "your", "yours", "yourself", "yourselves",
        # Document/Table domain stops
        "warning", "voltage", "current", "factor", "alarm", "code", "replace", "check", "state", 
        "occurred", "cause", "action", "failure", "device", "circuit", "unit", "fault",
        # Filler/Null representations
        "blank", "none", "empty", "na", "null", "nil", "n/a", "yes", "no", "true", "false"
    }

    for idx, item in enumerate(parsed_chunks):
        chunk_text = normalize_text(item.get("chunk_text") or "").strip()
        if not chunk_text:
            return f"Row {idx} chunk_text is empty"
            
        # Verify no hallucinated facts in text: check if output words are traceable
        out_words = re.findall(r"\w+", chunk_text.lower())
        new_words = [w for w in out_words if w not in allowed_words and w not in stops and not w.isdigit()]
        if len(new_words) > 3: # Allow small grammar/connection words
            return f"Row {idx} failed traceability validation. Untraceable words: {new_words}"
            
        # Verify individual structured values
        structured = item.get("structured") or {}
        for k, v in structured.items():
            v_clean = normalize_text(str(v)).lower().strip()
            if not v_clean:
                continue
            # Check if value matches any original cell or context words
            if not any(c in v_clean or v_clean in c for c in original_cell_texts):
                val_words = [w for w in re.findall(r"\w+", v_clean) if w not in stops and not w.isdigit()]
                untraceable_val_words = [w for w in val_words if w not in allowed_words]
                if untraceable_val_words:
                    return f"Row {idx} value {repr(v)} for key {repr(k)} has untraceable words: {untraceable_val_words}"
            
    return None

def repair_table_with_llm(block: dict, config: dict, section_lead: str, preceding_text: str) -> list[dict] | None:
    block_id = block.get("block_id", "unknown")
    try:
        # Build LLM client for chunking step
        llm = get_llm_for(config, config.get("chunking"))
        
        td = block.get("table_data") or {}
        headers = td.get("headers") or []
        rows = td.get("rows") or []
        raw_markdown = block.get("text") or ""
        document_id = block.get("document_id")
        source_ref = block.get("source_ref") or {}
        filename = source_ref.get("filename") or "document"
        
        system_prompt = (
            "You are an expert document parser. Your task is to correct errors in a poorly extracted "
            "PDF table and generate context-enriched row-level chunks in strict JSON format.\n\n"
            "Guidelines:\n"
            "1. Combine surrounding context and section path into EACH row-level chunk so it is self-contained "
            "for vector search (e.g. inject the alarm code, parent headers, and component name directly into the text).\n"
            "2. Resolve merged columns/cells (e.g. if 'Factor 1 Servo unit failure' is combined in one cell, split them "
            "logically based on column headers).\n"
            "3. Return exactly one JSON object per row containing:\n"
            "   - 'chunk_text': A complete, self-contained description of the row integrating context.\n"
            "   - 'structured': A dictionary of key-value pairs matching headers to cell values.\n"
            "4. Return ONLY a raw JSON array matching this schema: `[{\"chunk_text\": \"...\", \"structured\": {\"...\": \"...\"}}]`. "
            "Do not return any markdown code block fences (like ```json) or explanation text."
        )
        
        user_prompt = (
            f"Document Filename: {filename}\n"
            f"Parent Section/Context Path: {section_lead}\n"
            f"Preceding Context: {preceding_text}\n\n"
            f"Raw Table Headers:\n{json.dumps(headers)}\n\n"
            f"Raw Table Rows:\n{json.dumps(rows)}\n\n"
            f"Raw Markdown Representation:\n{raw_markdown}\n\n"
            "Please return the JSON array now:"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        from backend.core import usage
        usage.record_from_message("chunking", response)
        res_text = response.content.strip()
        
        # Strip code block fences if returned by the LLM
        if res_text.startswith("```"):
            lines = res_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            res_text = "\n".join(lines).strip()
            
        parsed_chunks = json.loads(res_text)
        if not isinstance(parsed_chunks, list):
            logger.info("Table %s: LLM response is not a JSON list: %s", block_id, res_text)
            return None
            
        # Run validations
        err = validate_repaired_chunks(parsed_chunks, rows, headers, section_lead, preceding_text)
        if err:
            logger.info("Table %s: Validation failed: %s — discarding LLM repair", block_id, err)
            return None
            
        # Convert parsed JSON objects to our standard Chunk schema
        chunks = []
        for item in parsed_chunks:
            chunk_text = item.get("chunk_text")
            if not chunk_text:
                continue
            
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "text": chunk_text,
                "token_count": _ntok(chunk_text),
                "table_data": item.get("structured"),
                "source_ref": {**source_ref, "section": section_lead},
                "tags": {
                    "document_type": "manual",
                    "chunk_type": "llm_repaired_table_row",
                    "repaired": True,
                }
            }
            chunks.append(chunk)
            
        return chunks
    except Exception as exc:
        logger.info("Table %s: LLM repair failed: %s — discarding LLM repair", block_id, exc)
        return None

def validate_spatial_conflation(block: dict, blocks: list[dict], caption_text: str) -> bool:
    """Check if any referenced alphanumeric tokens/codes in the caption are only present in distant page blocks."""
    norm_caption = normalize_text(caption_text)
    tokens = re.findall(r"\b\d+[a-zA-Z]\b", norm_caption)
    if not tokens:
        return False
        
    ref = block.get("source_ref") or {}
    page = ref.get("page")
    bbox = ref.get("bbox") or []
    if not page or len(bbox) != 4:
        return False
        
    # Exclude page-level summary diagrams (Refinement #4)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width > 300.0 or height > 300.0:
        logger.info("Diagram block %s: Spans page summary region (%.1fx%.1f) - skipping conflation validator", block.get("block_id"), width, height)
        return False
        
    # Analyze blocks on the same page
    for token in tokens:
        token_lower = token.lower()
        occurrences = []
        for b in blocks:
            bref = b.get("source_ref") or {}
            if bref.get("page") == page:
                b_text = normalize_text(b.get("text") or "").lower()
                if token_lower in b_text:
                    b_bbox = bref.get("bbox") or []
                    if len(b_bbox) == 4:
                        occurrences.append(b_bbox)
                        
        if occurrences:
            min_dist = float("inf")
            for obbox in occurrences:
                dx = max(0, bbox[0] - obbox[2], obbox[0] - bbox[2])
                dy = max(0, bbox[1] - obbox[3], obbox[1] - bbox[3])
                dist = (dx**2 + dy**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    
            # Log spatial distance (Refinement #1)
            logger.info("Diagram block %s: Token '%s' found at minimum distance of %.1f pt from figure bbox", block.get("block_id"), token, min_dist)
            
            # Distance threshold check (>150pt indicates distant conflation/reference hallucination)
            if min_dist > 150.0:
                return True
                
    return False


def chunk_blocks(
    blocks: list[dict],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    document_id: str | None = None,
    strategy: str = "recursive",
    semantic_model: str = "minishlab/potion-base-32M",
    section_aware: bool = True,
    split_large_tables: bool = True,
    config: dict | None = None,
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
    heading_stack: list[str] = [] # active headings path stack
    buf_heading_only = True       # True until real (non-heading) text is appended

    # Keep a rolling queue of the last 5 text/heading blocks to supply local preceding context
    preceding_blocks: list[str] = []

    def _block_page(block):
        ref = block.get("source_ref") or {}
        return ref.get("page") if isinstance(ref, dict) else None

    def _get_active_section() -> str:
        return " > ".join(heading_stack)

    def _ref(ref):
        sec = _get_active_section()
        if not section_aware or not sec:
            return ref
        r = dict(ref or {})
        r.setdefault("section", sec)
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
        ref = block.get("source_ref") or {}

        # 0. CAD Title Block & Component Specification Handlers
        cad_title = _try_extract_cad_title_chunk(block, document_id, _ref)
        if cad_title:
            flush()
            chunks.append(cad_title)
            continue

        cad_comp = _try_extract_cad_component_chunks(block, document_id, _ref)
        if cad_comp:
            flush()
            chunks.extend(cad_comp)
            continue

        if buf_page is not None and pg is not None and pg != buf_page:
            if buf_heading_only:
                buf_page = pg
            else:
                flush()

        if section_aware and btype == "heading" and text:
            if not buf_heading_only:
                flush()
            
            # Update heading stack based on inferred level
            level = get_heading_level(text, ref.get("bbox"))
            # Keep stack truncated up to parent level (level - 1)
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(text)
            
            if buf_ref is None:
                buf_ref = block.get("source_ref")
            buf_page = pg
            buf_parts.append(text)
            buf_heading_only = True
            
            # Add to preceding queue
            preceding_blocks.append(text)
            if len(preceding_blocks) > 5:
                preceding_blocks.pop(0)
            continue

        if btype in ATOMIC_TYPES:
            heading_lead = pending_lead_text()
            flush()
            
            if btype == "image_caption" and (block.get("metadata") or {}).get(
                "pending_vision"
            ):
                continue

            if btype == "table":
                active_sec = _get_active_section()
                prec_context = " ".join(preceding_blocks)
                
                # Check if it needs LLM repair first
                classification = classify_table(block, active_sec, prec_context)
                if classification["needs_llm"] and config is not None:
                    logger.info("Table needs LLM repair: %s (reasons: %s)", block.get("block_id"), classification["reasons"])
                    llm_chunks = repair_table_with_llm(block, config, active_sec, prec_context)
                    if llm_chunks is not None:
                        logger.info("Successfully repaired table with LLM, generated %d chunks", len(llm_chunks))
                        chunks.extend(llm_chunks)
                        continue
                    # Else fall back automatically to standard splitters

                trouble_chunks = _try_extract_troubleshooting_table_chunks(block, document_id, _ref)
                if trouble_chunks:
                    chunks.extend(trouble_chunks)
                    continue

                alarm_chunks = _try_extract_alarm_table_chunks(block, document_id, _ref)
                if alarm_chunks:
                    chunks.extend(alarm_chunks)
                    continue

                warning_chunks = _try_extract_warning_chunk(block, document_id, _ref)
                if warning_chunks:
                    if isinstance(warning_chunks, list):
                        chunks.extend(warning_chunks)
                    else:
                        chunks.append(warning_chunks)
                    continue

                model_chunks = _try_extract_model_column_chunks(block, document_id, _ref)
                if model_chunks:
                    chunks.extend(model_chunks)
                    continue

            if btype == "image_caption":
                # Compute dynamic confidence rating
                base_score = 1.0
                metadata = block.get("metadata") or {}
                fig_kind = metadata.get("figure_kind")
                if fig_kind == "unknown":
                    base_score -= 0.3
                if fig_kind in {"text", "blank", "decoration"}:
                    base_score -= 0.1
                    
                # Conflation check
                conflated = validate_spatial_conflation(block, blocks, text)
                if conflated:
                    logger.warning("Caption block %s: Spatial conflation detected - flagging for exclusion", block.get("block_id"))
                    base_score -= 0.3
                    
                # Clamp score
                confidence_score = max(0.0, base_score)
                
                exclude = False
                if confidence_score < 0.7 or fig_kind in {"logo", "banner", "header_footer"}:
                    exclude = True
                    logger.info("Caption block %s: Excluded from index (Confidence: %.2f, Kind: %s)", block.get("block_id"), confidence_score, fig_kind)
                    
                b = dict(block)
                b["source_ref"] = _ref(block.get("source_ref"))
                lead_text = f"{heading_lead}\n{text}".strip() if heading_lead else text
                
                c = _make_chunk(b, lead_text, document_id)
                c.setdefault("tags", {})
                c["tags"]["exclude_from_index"] = exclude
                c["tags"]["confidence"] = confidence_score
                c["tags"]["figure_kind"] = fig_kind
                chunks.append(c)
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

        if text:
            if buf_ref is None:
                buf_ref = block.get("source_ref")
            buf_parts.append(text)
            buf_heading_only = False
            preceding_blocks.append(text)
            if len(preceding_blocks) > 5:
                preceding_blocks.pop(0)

    flush()
    seen_texts: set[tuple] = set()
    deduped_chunks = []
    for c in chunks:
        t_raw = (c.get("text") or "").strip()
        tok_count = c.get("token_count", _ntok(t_raw))
        has_structured_data = bool(c.get("table_data") or c.get("image_path"))

        if tok_count < 5 and not has_structured_data:
            if re.match(r"^\d+(-\d+)?$", t_raw) or re.match(r"^[iIvVxXlLcCdDmM]+$", t_raw) or len(t_raw) <= 3:
                continue

        norm_t = " ".join(t_raw.split())
        page = (c.get("source_ref") or {}).get("page")
        key = (page, norm_t)
        if norm_t and key not in seen_texts:
            seen_texts.add(key)
            deduped_chunks.append(c)
        elif not norm_t:
            deduped_chunks.append(c)
    return deduped_chunks


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
            config=config,
        )
        state.setdefault("chunks", []).extend(chunks)
        return state


if __name__ == "__main__":
    import json
    import sys

    raw_data = json.load(open(sys.argv[1], encoding="utf-8"))
    blocks = raw_data.get("blocks", raw_data) if isinstance(raw_data, dict) else raw_data
    if isinstance(blocks, dict) and "pages" in blocks:
        # handle PyMuPDF page-nested format
        page_blocks = []
        for page in blocks.get("pages", []):
            page_no = page.get("page_number", 1)
            for tab in page.get("tables", []):
                tab["type"] = "table"
                tab["page_number"] = page_no
                page_blocks.append(tab)
            for blk in page.get("text_blocks", []):
                blk["type"] = "text"
                page_blocks.append(blk)
        blocks = page_blocks

    out = chunk_blocks(blocks if isinstance(blocks, list) else [])
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    if dest:
        json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)
        print(f"wrote {len(out)} chunks -> {dest}")
    else:
        print(json.dumps(out[:3], indent=2))
        print(f"... {len(out)} chunks total")
