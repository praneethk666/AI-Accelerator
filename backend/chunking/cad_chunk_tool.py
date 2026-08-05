"""cad_chunk_tool - CAD-specific LLM chunking + enrichment in a single pass.

Replaces the `chunk` + `enrich_chunks` steps for cad_route and circuit_route.
Instead of two sequential LLM calls (one to group blocks, one to label chunks),
this tool does everything in ONE prompt per page:

  1. Groups the raw CAD blocks (image captions, tables, annotation text) into
     cohesive, search-friendly chunks.
  2. Simultaneously generates a `summary` and `keywords` for each chunk, exactly
     matching what EnrichChunksTool would produce.
  3. Carries table_data through faithfully: real table blocks are preserved
     verbatim from the source (headers + rows), so the schema never degrades.

The tool writes state["chunks"] in exactly the same shape as ChunkTool +
EnrichChunksTool would, meaning the downstream embed -> index steps need no change.

Fallback: if the LLM call fails for a page, that page's blocks are individually
passed through as single-block chunks (raw pass-through), so no data is lost.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict

from backend.core.tool import PipelineState

logger = logging.getLogger(__name__)

# Part-number / model-code pattern (mirrors chunk_tool._COMPONENT_CODE_RE)
_COMPONENT_CODE_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{1,6}-?[A-Z0-9]{2,}[-.*][A-Z0-9][A-Z0-9\-.*]*"
    r"|[A-Z][A-Z0-9]{1,5}\d{3,}[A-Z0-9\-]*"
    r"|\d{4,}[A-Z]{1,4}[A-Z0-9\-]*"
    r")\b",
    re.ASCII,
)

_SYSTEM_PROMPT = (
    "You are an expert engineering document processor specializing in CAD drawing analysis.\n\n"
    "Your task: given a list of extracted blocks from a SINGLE CAD drawing page, group them\n"
    "into logical, self-contained chunks for a vector search index, and simultaneously\n"
    "produce a one-sentence summary and 6-10 specific keywords for each chunk.\n\n"
    "CHUNKING GUIDELINES\n"
    "1. Combine semantically related blocks into one chunk:\n"
    "   - A section-view caption + nearby annotation text -> one chunk\n"
    "   - A parts table (type=table) -> one chunk (never split small tables)\n"
    "   - General notes / disclaimers -> their own chunk\n"
    "   - The title block metadata table -> its own chunk\n"
    "2. Each chunk must be SELF-CONTAINED -- a reader should understand what it describes\n"
    "   without needing the other chunks.\n"
    "3. If a block has table_data already set, preserve it EXACTLY in structured_data\n"
    '   using {"headers": [...], "rows": [[...], ...]} format. Do not reformat or merge\n'
    "   separate tables.\n"
    "4. Image caption blocks that describe large drawing views may be merged with\n"
    "   annotation text that labels components visible in the same view.\n\n"
    "ENRICHMENT GUIDELINES\n"
    "- summary: one precise sentence stating what the chunk is about in the document's\n"
    "  own terms (e.g. \"Parts list for the spindle assembly listing 11 standard components\").\n"
    "- keywords: 6-10 SPECIFIC terms from the chunk -- part numbers, model names,\n"
    "  component/procedure names, specs and units. Never use stopwords or terms not\n"
    "  present in the chunk text or table.\n\n"
    "OUTPUT FORMAT\n"
    "Return ONLY a raw JSON array, no markdown fences. Each element:\n"
    "{\n"
    '  "chunk_text":            "<complete markdown text for the chunk>",\n'
    '  "summary":               "<one-sentence summary>",\n'
    '  "keywords":              ["term1", "term2", ...],\n'
    '  "structured_data":       null or {"headers": [...], "rows": [[...], ...]},\n'
    '  "source_block_indices":  [<block_idx integers>]\n'
    "}"
)


def _extract_component_codes(text: str, table_data: dict | None = None) -> list[str]:
    haystack = text or ""
    if table_data:
        for row in table_data.get("rows") or []:
            haystack += " " + " ".join(str(c) for c in row if c)
    codes = {m.group(0).upper().strip(".-") for m in _COMPONENT_CODE_RE.finditer(haystack)}
    codes = {c for c in codes if len(c) >= 3 and not c.replace(".", "").isdigit()}
    return sorted(codes)


def _table_data_from_block(block: dict) -> dict | None:
    """Return the block's table_data normalised to {headers, rows} schema."""
    td = block.get("table_data")
    if not td or not isinstance(td, dict):
        return None
    if "headers" in td and "rows" in td:
        return td
    # Legacy flat-dict format (column_name -> list of values): convert
    keys = [k for k in td.keys() if k not in ("headers", "rows")]
    if not keys:
        return None
    max_len = max((len(td[k]) if isinstance(td[k], list) else 1) for k in keys)
    rows = []
    for i in range(max_len):
        row = []
        for k in keys:
            vals = td[k]
            if isinstance(vals, list):
                row.append(vals[i] if i < len(vals) else "")
            else:
                row.append(str(vals) if i == 0 else "")
        rows.append(row)
    return {"headers": keys, "rows": rows}


def _normalize_structured_data(s_data: dict | None) -> dict | None:
    """Ensure LLM-produced structured_data is in {headers, rows} format."""
    if not s_data or not isinstance(s_data, dict):
        return None
    if "headers" in s_data and "rows" in s_data:
        return s_data
    rows = [[str(k), str(v)] for k, v in s_data.items() if k not in ("headers", "rows")]
    return {"headers": ["Field", "Value"], "rows": rows} if rows else None


def _make_single_block_chunk(block: dict, document_id: str | None, doc_type: str) -> dict:
    """Fallback: convert a single block directly into a chunk with frequency keywords."""
    text = (block.get("text") or "").strip()
    td = _table_data_from_block(block)
    tags: dict = {
        "doc_type": doc_type,
        "document_type": doc_type,
        "chunk_type": "cad_passthrough_chunk",
    }
    codes = _extract_component_codes(text, td)
    if codes:
        tags["component_codes"] = codes
    if td:
        tags["has_table"] = True

    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    stopwords = {
        "the", "and", "for", "this", "that", "with", "from", "visible", "labels",
        "section", "view", "drawing", "shows", "details",
    }
    freq_kws = sorted(
        {w.lower() for w in words if w.lower() not in stopwords},
        key=lambda w: -text.lower().count(w.lower()),
    )
    tags["keywords"] = freq_kws[:8]

    return {
        "chunk_id": str(uuid.uuid4()),
        "document_id": block.get("document_id") or document_id,
        "text": text or "[drawing]",
        "token_count": max(1, len((text or "").split())),
        "tags": tags,
        "table_data": td,
        "image_path": (block.get("metadata") or {}).get("image_path"),
        "source_ref": block.get("source_ref"),
    }


def _chunk_page_with_llm(
    page: int,
    page_blocks: list[dict],
    llm,
    config: dict,
    document_id: str | None,
    doc_type: str,
) -> list[dict]:
    """Call the LLM to chunk + enrich one page's blocks. Returns list of chunks."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from backend.core import usage
    from backend.core.llm_client import resolve_model_provider

    prompt_blocks = []
    for i, b in enumerate(page_blocks):
        pb: dict = {
            "block_idx": i,
            "type": b.get("type"),
            "text": b.get("text"),
        }
        td = _table_data_from_block(b)
        if td:
            pb["table_data"] = td
        lbl = (b.get("metadata") or {}).get("label")
        if lbl:
            pb["label"] = lbl
        prompt_blocks.append(pb)

    user_prompt = (
        f"Document ID: {document_id}\n"
        f"Document type: {doc_type}\n"
        f"Page: {page}\n\n"
        f"Blocks:\n{json.dumps(prompt_blocks, indent=2)}\n\n"
        "Return the JSON array now:"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    try:
        model_name, provider_name = resolve_model_provider(config, {})
        usage.record_from_message(
            "cad_chunk_llm", response,
            prompt=messages, model=model_name, provider=provider_name,
        )
    except Exception:
        pass

    res_text = response.content.strip()
    if res_text.startswith("```"):
        lines = res_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        res_text = "\n".join(lines).strip()

    parsed = json.loads(res_text)
    if not isinstance(parsed, list):
        raise ValueError(f"LLM returned non-list for page {page}")

    chunks = []
    for item in parsed:
        chunk_text = (item.get("chunk_text") or "").strip()
        if not chunk_text:
            continue

        indices = item.get("source_block_indices") or []
        source_blocks = [page_blocks[i] for i in indices if 0 <= i < len(page_blocks)]

        primary_ref: dict = {"page": page}
        for sb in source_blocks:
            sref = sb.get("source_ref")
            if sref and isinstance(sref, dict):
                primary_ref = dict(sref)
                break

        # Prefer verbatim table_data from source blocks, fall back to LLM structured_data
        table_data: dict | None = None
        for sb in source_blocks:
            td = _table_data_from_block(sb)
            if td:
                table_data = td
                break
        if table_data is None:
            table_data = _normalize_structured_data(item.get("structured_data"))

        tags: dict = {
            "doc_type": doc_type,
            "document_type": doc_type,
            "chunk_type": "llm_grouped_cad_chunk",
        }
        summary = (item.get("summary") or "").strip() or None
        if summary:
            tags["summary"] = summary
        kws = item.get("keywords")
        if isinstance(kws, list) and kws:
            tags["keywords"] = kws
        if table_data:
            tags["has_table"] = True

        codes = _extract_component_codes(chunk_text, table_data)
        if codes:
            tags["component_codes"] = codes

        image_path: str | None = None
        for sb in source_blocks:
            ip = (sb.get("metadata") or {}).get("image_path")
            if ip:
                image_path = ip
                break

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "document_id": document_id,
            "text": chunk_text,
            "token_count": max(1, len(chunk_text.split())),
            "tags": tags,
            "table_data": table_data,
            "image_path": image_path,
            "source_ref": primary_ref,
        })

    return chunks


class CADChunkTool:
    """LLM-based chunking + enrichment for CAD drawings and circuit diagrams.

    Replaces the standard chunk -> enrich_chunks two-step for cad_route and
    circuit_route. Reads state["blocks"], writes state["chunks"].
    """

    name = "cad_chunk_llm"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        blocks: list[dict] = state.get("blocks") or []
        document_id: str | None = state.get("document_id")
        doc_type: str = state.get("document_type") or "cad_drawing"
        industry: str | None = state.get("industry")

        if not blocks:
            logger.warning("cad_chunk_llm: no blocks in state; producing empty chunk list")
            state.setdefault("chunks", [])
            return state

        # Resolve LLM using global config (not per-step enrichment override)
        llm = None
        try:
            from backend.core.llm_client import get_llm_for
            llm = get_llm_for(config, {})
        except Exception as exc:
            logger.warning(
                "cad_chunk_llm: could not get LLM (%s); using passthrough chunking", exc
            )

        # Group blocks by page
        pages: dict[int, list[dict]] = defaultdict(list)
        for b in blocks:
            page = (b.get("source_ref") or {}).get("page") or 1
            pages[page].append(b)

        all_chunks: list[dict] = []

        for page, page_blocks in sorted(pages.items()):
            if llm is not None:
                try:
                    page_chunks = _chunk_page_with_llm(
                        page, page_blocks, llm, config, document_id, doc_type
                    )
                    logger.info(
                        "cad_chunk_llm: page %d -> %d LLM chunks (%d source blocks)",
                        page, len(page_chunks), len(page_blocks),
                    )
                    all_chunks.extend(page_chunks)
                    continue
                except Exception as exc:
                    logger.warning(
                        "cad_chunk_llm: LLM failed for page %d (%s); "
                        "falling back to passthrough for this page", page, exc,
                    )

            # Fallback: one chunk per block, no LLM
            fallback = []
            for b in page_blocks:
                if not (b.get("text") or "").strip() and not b.get("table_data"):
                    continue
                fallback.append(_make_single_block_chunk(b, document_id, doc_type))
            all_chunks.extend(fallback)
            logger.info(
                "cad_chunk_llm: page %d -> %d passthrough chunks", page, len(fallback)
            )

        # Stamp industry + doc_type on all chunks (mirrors EnrichChunksTool)
        for chunk in all_chunks:
            tags = chunk.setdefault("tags", {})
            if industry:
                tags.setdefault("industry", industry)
            tags.setdefault("doc_type", doc_type)

        state.setdefault("chunks", []).extend(all_chunks)
        logger.info(
            "cad_chunk_llm: wrote %d chunks for document %s (%d source blocks, %d pages)",
            len(all_chunks), document_id, len(blocks), len(pages),
        )
        return state
