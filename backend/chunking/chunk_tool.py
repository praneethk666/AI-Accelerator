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
    if not ("drawing no" in text.lower() or "title block" in text.lower() or "jtekt" in text.lower()):
        return None

    source_ref = ref_fn(block.get("source_ref"))
    chunk_text = f"# CAD Drawing Title Block & Specifications\n\n{text}"

    # Extract drawing number and company for metadata filtering
    drawing_no = ""
    drawing_match = re.search(r"Drawing No\*?:\s*([A-Za-z0-9\-\_]+)", text, re.IGNORECASE)
    if drawing_match:
        drawing_no = drawing_match.group(1)

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
            "company": "JTEKT CORPORATION",
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


def _try_extract_excel_invoice_chunks(block: dict, document_id: str | None, ref_fn) -> list[dict] | None:
    """If a table represents an invoice line item spreadsheet (has PDF_Name or Invoice_Number),
    group rows by invoice and create 1 self-contained chunk per invoice with explicit headers
    and rich metadata for Qdrant filtering."""
    grid = _extract_grid_from_block(block)
    if not grid or len(grid) < 2:
        return None

    headers = grid[0]
    headers_lower = [str(h).lower() for h in headers]

    pdf_name_idx = -1
    inv_num_idx = -1
    total_amt_idx = -1

    for idx, h in enumerate(headers_lower):
        if "pdf_name" in h or "filename" in h:
            pdf_name_idx = idx
        elif "invoice_number" in h or "inv_num" in h:
            inv_num_idx = idx
        elif "total_payable" in h or "total_amount" in h or "grand_total" in h:
            total_amt_idx = idx

    if pdf_name_idx == -1 and inv_num_idx == -1:
        return None

    data_rows = grid[1:]
    from collections import defaultdict
    invoice_groups = defaultdict(list)

    for row in data_rows:
        if pdf_name_idx != -1 and pdf_name_idx < len(row) and row[pdf_name_idx]:
            key = str(row[pdf_name_idx]).strip()
        elif inv_num_idx != -1 and inv_num_idx < len(row) and row[inv_num_idx]:
            key = str(row[inv_num_idx]).strip()
        else:
            key = "general_invoice"
        invoice_groups[key].append(row)

    source_ref = ref_fn(block.get("source_ref"))
    invoice_chunks = []

    for invoice_key, rows in invoice_groups.items():
        sample_row = rows[0]
        inv_num = sample_row[inv_num_idx] if inv_num_idx != -1 and inv_num_idx < len(sample_row) else ""
        tot_amt = sample_row[total_amt_idx] if total_amt_idx != -1 and total_amt_idx < len(sample_row) else ""

        summary_lines = [f"Invoice File: {invoice_key}"]
        if inv_num:
            summary_lines.append(f"Invoice Number: {inv_num}")
        if tot_amt:
            summary_lines.append(f"Total Amount: {tot_amt}")
        summary_header = "\n".join(summary_lines)

        md_table = _rebuild_markdown(headers, rows)
        chunk_text = f"{summary_header}\n\n{md_table}"

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id") or document_id,
            "text": chunk_text,
            "token_count": _ntok(chunk_text),
            "table_data": {"headers": headers, "rows": rows},
            "source_ref": source_ref,
            "tags": {
                "document_type": "excel",
                "chunk_type": "excel_invoice",
                "pdf_name": invoice_key,
                "invoice_number": inv_num,
                "total_amount": tot_amt,
                "item_count": len(rows),
                "has_table": True,
            },
        }
        invoice_chunks.append(chunk)

    return invoice_chunks if invoice_chunks else None


def _try_extract_large_table_chunks(block: dict, document_id: str | None, ref_fn, max_rows: int = 25) -> list[dict] | None:
    """If a table exceeds 30 rows (e.g. large Excel sheet or catalog), paginate it into
    sub-chunks of `max_rows` with the column header repeated at the top of every chunk."""
    grid = _extract_grid_from_block(block)
    if not grid or len(grid) <= 30:
        return None

    headers = grid[0]
    data_rows = grid[1:]
    source_ref = ref_fn(block.get("source_ref"))

    table_chunks = []
    total_parts = (len(data_rows) + max_rows - 1) // max_rows

    for i in range(0, len(data_rows), max_rows):
        part_idx = (i // max_rows) + 1
        sub_rows = data_rows[i : i + max_rows]
        sub_grid = [headers] + sub_rows

        md_text = _rebuild_markdown(headers, sub_rows)
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id") or document_id,
            "text": md_text,
            "token_count": _ntok(md_text),
            "table_data": {"grid": sub_grid, "headers": headers, "rows": sub_rows},
            "source_ref": source_ref,
            "tags": {
                "chunk_type": "large_table_part",
                "part": part_idx,
                "total_parts": total_parts,
            },
        }
        table_chunks.append(chunk)

    return table_chunks


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

            if btype == "table":
                # 0. Check if table represents a Troubleshooting Cause & Action list
                trouble_chunks = _try_extract_troubleshooting_table_chunks(block, document_id, _ref)
                if trouble_chunks:
                    chunks.extend(trouble_chunks)
                    continue

                # 1. Check if table represents an Alarm/Error list
                alarm_chunks = _try_extract_alarm_table_chunks(block, document_id, _ref)
                if alarm_chunks:
                    chunks.extend(alarm_chunks)
                    continue

                # 1. Check if table represents an Excel invoice line item spreadsheet
                excel_invoice_chunks = _try_extract_excel_invoice_chunks(block, document_id, _ref)
                if excel_invoice_chunks:
                    chunks.extend(excel_invoice_chunks)
                    continue

                # 2. Check if table is a safety warning callout
                warning_chunks = _try_extract_warning_chunk(block, document_id, _ref)
                if warning_chunks:
                    if isinstance(warning_chunks, list):
                        chunks.extend(warning_chunks)
                    else:
                        chunks.append(warning_chunks)
                    continue

                # 2. Check if table exceeds 30 rows (large table header-repeated pagination)
                large_chunks = _try_extract_large_table_chunks(block, document_id, _ref)
                if large_chunks:
                    chunks.extend(large_chunks)
                    continue

                # 3. Check if table is a multi-column model specification matrix
                model_chunks = _try_extract_model_column_chunks(block, document_id, _ref)
                if model_chunks:
                    chunks.extend(model_chunks)
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
    # Deduplicate exact repeat chunks & filter near-empty noise (<5 tokens)
    seen_texts = set()
    deduped_chunks = []
    for c in chunks:
        t_raw = (c.get("text") or "").strip()
        tok_count = c.get("token_count", _ntok(t_raw))

        # Filter near-empty noise chunks (<5 tokens matching page/section numbers or stray fragments)
        if tok_count < 5:
            if re.match(r"^\d+(-\d+)?$", t_raw) or re.match(r"^[iIvVxXlLcCdDmM]+$", t_raw) or len(t_raw) <= 3:
                continue

        norm_t = " ".join(t_raw.split())
        if norm_t and norm_t not in seen_texts:
            seen_texts.add(norm_t)
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
