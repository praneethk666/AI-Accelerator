"""Detects source content that's been deliberately blanked out in the ORIGINAL
document (e.g. a parts table whose real part numbers are replaced with "***" /
"**-*********-*") -- real finding, 3-Aug: the spindle assembly CAD drawing
(MS03AAA789AB) has exactly this. No extraction quality fix addresses this: the
data genuinely isn't in the source. Left undetected, this silently degrades into
either a hallucinated answer (the LLM "helpfully" invents a plausible-looking
part number) or a confusing one (it just repeats "***" back with no explanation).

Detected blocks get metadata.redacted=True so the answerer (backend/retrieval/
answerer.py) can say so explicitly instead of either of those -- see
_build_user_content's redaction note.
"""
from __future__ import annotations

_PLACEHOLDER_CHARS = set("*-_# ")


def _is_placeholder_cell(value: str) -> bool:
    """True if `value` has content but that content is ENTIRELY placeholder
    characters (*, -, _, #, whitespace) -- no real alphanumeric text at all.
    "***" / "**-*********-*" -> True. "M8x20" / "**bold**" -> False (real
    alphanumeric content present)."""
    v = (value or "").strip()
    if not v:
        return False
    return all(c in _PLACEHOLDER_CHARS for c in v) and any(c in "*#" for c in v)


def is_redacted_table(table_data: dict | None, threshold: float = 0.8) -> bool:
    """True if `threshold` or more of the table's non-empty DATA cells are
    placeholder-only. Headers are real column names ("Parts No.", "Q'ty") even
    in a redacted table, so only row cells are checked -- including headers
    would dilute the ratio and miss genuinely fully-redacted tables."""
    if not table_data:
        return False
    rows = table_data.get("rows") or []
    cells = [c for row in rows for c in row if isinstance(c, str) and c.strip()]
    if not cells:
        return False
    placeholder = sum(1 for c in cells if _is_placeholder_cell(c))
    return (placeholder / len(cells)) >= threshold


def is_redacted_text(text: str, threshold: float = 0.8) -> bool:
    """Same idea for a plain text/caption block, in case a whole line (not a
    table cell) is blanked out. Checked per WORD (whitespace-split) rather than
    the whole string, since real body text unavoidably contains scattered
    punctuation the whole-string version of this check would misfire on."""
    words = [w for w in (text or "").split() if w.strip()]
    if not words:
        return False
    placeholder = sum(1 for w in words if _is_placeholder_cell(w))
    return (placeholder / len(words)) >= threshold


def tag_blocks_with_redaction(blocks: list[dict]) -> list[dict]:
    """Mutates each block in place: metadata.redacted=True + a short human-
    readable metadata.redaction_reason wherever detected. Blocks with no
    redaction are left untouched. Returns the same list."""
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "table" and is_redacted_table(b.get("table_data")):
            b.setdefault("metadata", {})["redacted"] = True
            b["metadata"]["redaction_reason"] = (
                "This table's values are blanked out (e.g. \"***\") in the "
                "source document -- the real data isn't available in this file."
            )
        elif b.get("type") in ("text", "heading", "image_caption") and is_redacted_text(b.get("text")):
            b.setdefault("metadata", {})["redacted"] = True
            b["metadata"]["redaction_reason"] = (
                "This content is blanked out (e.g. \"***\") in the source "
                "document -- the real data isn't available in this file."
            )
    return blocks
