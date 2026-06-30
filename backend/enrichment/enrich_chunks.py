from __future__ import annotations

import re
from collections import Counter
from numbers import Number
from datetime import date, datetime

from backend.core.tool import PipelineState


_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "you", "your", "our", "their", "they",
    "there", "here", "what", "when", "where", "which", "who", "whom", "why",
    "how", "into", "onto", "than", "then", "than", "can", "could", "would",
    "should", "may", "might", "will", "shall", "also", "any", "all", "each",
    "such", "per", "via", "use", "used", "using", "based", "within", "over",
    "under", "between", "among", "more", "most", "less", "least", "one", "two",
    "three", "four", "five", "figure", "fig", "table", "slide", "page", "note",
}

_SECTION_PATTERNS = [
    r"(?im)^\s*(abstract|summary|overview|introduction|background|methodology|methods|results|discussion|conclusion|conclusions|references|appendix)\s*$",
    r"(?im)^\s*(table of contents|toc)\s*$",
]

_INDUSTRY_HINTS = {
    "finance": {"invoice", "balance", "revenue", "profit", "ledger", "audit", "fiscal", "equity"},
    "electronics": {"schematic", "voltage", "resistor", "capacitor", "signal", "pcb", "semiconductor"},
    "manufacturing": {"assembly", "drawing", "tolerance", "weld", "machining", "fixture", "bom"},
    "healthcare": {"patient", "diagnosis", "clinical", "therapy", "medical", "drug", "dosage"},
    "automotive": {"vehicle", "engine", "torque", "transmission", "chassis", "automotive", "motor"},
    "legal": {"contract", "agreement", "clause", "liability", "indemnity", "arbitration"},
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Number) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return _normalize_text(str(value))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_/-]{3,}", (text or "").lower())


def _keywords(text: str, limit: int = 12) -> list[str]:
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and not t.isdigit()]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def _infer_section(text: str) -> str | None:
    for pattern in _SECTION_PATTERNS:
        match = re.search(pattern, text or "")
        if match:
            return match.group(1).lower().replace(" ", "_")
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        first = lines[0].lower()
        if len(first) <= 80 and re.fullmatch(r"[A-Z0-9 ,:/._-]+", lines[0]):
            return "header"
    return None


def _infer_topic(text: str) -> str | None:
    words = set(_tokenize(text))
    if not words:
        return None
    scored = []
    for industry, hints in _INDUSTRY_HINTS.items():
        score = sum(1 for hint in hints if hint in words)
        if score:
            scored.append((score, industry))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _summarize(text: str, max_chars: int = 280) -> str:
    text = _normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _chunk_stats(text: str) -> dict:
    tokens = _tokenize(text)
    return {
        "text_length": len(text or ""),
        "word_count": len((text or "").split()),
        "token_count": len(tokens),
        "digit_ratio": round(sum(ch.isdigit() for ch in (text or "")) / max(1, len(text or "")), 4),
    }


def _walk_values(value, max_items: int = 64) -> list[str]:
    found: list[str] = []

    def visit(obj):
        if len(found) >= max_items:
            return
        if obj is None:
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                if len(found) >= max_items:
                    return
                if isinstance(key, str) and key in {"text", "title", "name", "filename", "sheet", "slide", "page"}:
                    found.append(_coerce_text(item))
                visit(item)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                if len(found) >= max_items:
                    return
                visit(item)
            return
        if isinstance(obj, (str, bytes, Number, datetime, date)):
            text = _coerce_text(obj)
            if text:
                found.append(text)

    visit(value)
    return [item for item in found if item]


def _derive_text(chunk: dict) -> str:
    parts = []
    for field in ("text", "title", "caption", "summary"):
        if field in chunk:
            parts.append(_coerce_text(chunk.get(field)))
    parts.extend(_walk_values(chunk.get("table_data")))
    parts.extend(_walk_values(chunk.get("metadata")))
    parts.extend(_walk_values(chunk.get("source_ref")))
    return _normalize_text(" ".join(part for part in parts if part))


class EnrichChunksTool:
    name: str = "enrich_chunks"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        chunks = state.get("chunks", [])
        doc_id = state.get("document_id", "")

        for index, chunk in enumerate(chunks):
            tags = chunk.setdefault("tags", {})
            text = _derive_text(chunk)
            source_ref = chunk.get("source_ref") or {}
            stats = _chunk_stats(text)

            tags.setdefault("document_id", chunk.get("document_id") or doc_id)
            tags.setdefault("chunk_index", index)
            tags["content_type"] = "table" if chunk.get("table_data") else "image_caption" if chunk.get("image_path") else "text"
            tags["summary"] = _summarize(text)
            tags["keywords"] = _keywords(text)
            tags["section"] = _infer_section(text)
            tags["topic"] = _infer_topic(text)
            tags["text_length"] = stats["text_length"]
            tags["word_count"] = stats["word_count"]
            tags["token_count"] = stats["token_count"]
            tags["digit_ratio"] = stats["digit_ratio"]
            tags["source_filename"] = source_ref.get("filename")
            if source_ref.get("page") is not None:
                tags["source_page"] = source_ref.get("page")
            if source_ref.get("sheet"):
                tags["source_sheet"] = source_ref.get("sheet")
            if source_ref.get("slide") is not None:
                tags["source_slide"] = source_ref.get("slide")

            if chunk.get("table_data"):
                table = chunk["table_data"]
                headers = table.get("headers") or []
                rows = table.get("rows") or []
                tags["table_headers"] = [_coerce_text(h) for h in headers[:12]]
                tags["row_count"] = len(rows)
                tags["column_count"] = len(headers)
                tags["has_tabular_data"] = True
                if rows:
                    sample = []
                    for row in rows[:3]:
                        safe_row = [_coerce_text(cell) for cell in list(row)[:8]]
                        sample.append(" | ".join(safe_row))
                    tags["table_sample"] = sample

            if chunk.get("image_path"):
                tags["has_image"] = True
            if chunk.get("sparse_vector"):
                tags["hybrid_ready"] = True

            chunk["text"] = text

        return state
