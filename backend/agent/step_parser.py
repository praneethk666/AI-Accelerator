"""Deterministic parser: given a document's raw extracted blocks, locate one
named/paged section and split its body text into an ordered, ID-addressable step
graph -- "Step 1: do X" -> "Step 2: do Y" -- for guided procedure walkthroughs
(backend/agent/procedure_tools.py).

Deliberately NOT an LLM call, matching this codebase's existing preference for
exact-format parsing over LLM guessing wherever the source format is reliable
(same philosophy as backend/categorize/id_graph.py's regex ID matching). Verified
against exactly ONE real manual so far
(.../3.INSTRUCTION MANUAL/EN/20230831_99Y_05_G0797V10_Changeover of work
holder&phase Indexing.pdf), which uses a genuinely consistent format: numbered
sub-sections ("1.1 Replacing the Workpiece Holder") whose body is a flat list of
parenthesized numbered steps ("(1) Place the [Operation mode] selection switch
...", "(2) ...", ...). This must FAIL CLOSED (return None), not guess, on any
document that doesn't match this exact convention -- other manuals in the corpus
may use "Step 1:", bullets, or something else entirely, and a wrong guess on a
safety-relevant industrial procedure is worse than admitting "not parseable".

Reads BLOCKS (backend/storage/postgres_store.py::PostgresStore.get_blocks), not
CHUNKS -- section-aware chunking does not guarantee one procedure stays in one
chunk (chunk_tool.py's token-window splitting can still cut mid-section), so
chunk boundaries are not a reliable place to look for step structure.
"""
from __future__ import annotations

import re
from typing import Any

# "(1) some step text" through the next "(N) " marker or the end of the buffer.
# Matches a marker anywhere it's preceded by whitespace/string-start, NOT only at
# a line start -- real finding, 11-Aug: Docling sometimes extracts several
# consecutive numbered steps as ONE block (e.g. "(1) Turn the... (2) Press the...
# (3) Place the..." all on one joined line, no newlines between them), which a
# line-anchored ^ pattern would only match the FIRST marker of, silently losing
# the rest and failing the section closed. Requiring the following char be
# uppercase is a light precision guard against matching an unrelated
# parenthesized number in running prose (e.g. a figure callout) -- the real
# backstop is parse_procedure_from_blocks' own contiguous-increasing-sequence
# check below, which any spurious match would very likely break anyway. DOTALL
# so a step's own text can still span multiple physical lines/blocks.
_STEP_RE = re.compile(
    r"(?:\A|(?<=\s))\((\d+)\)\s+(?=[A-Z])(.+?)(?=\s\(\d+\)\s+[A-Z]|\Z)",
    re.DOTALL,
)

# Best-effort branch detection: a step's OWN text explicitly naming a conditional
# jump to another step number ("if <condition> ... step (N)"). Never inferred
# beyond an EXPLICIT numeric reference already present in the source text -- no
# confirmed real example of this in the corpus yet, so this returns [] far more
# often than not. Matches this codebase's "never invent, never guess" philosophy
# (same standard id_graph.py's exact-ID regexes and the answerer's verbatim-value
# rule both hold to).
_BRANCH_REF_RE = re.compile(r"\bif\b([^.]{1,100}?)\bstep\s*\(?(\d+)\)?", re.IGNORECASE)

_DEFAULT_MAX_STEPS = 50


def _detect_branches(step_text: str) -> list[dict]:
    branches = []
    for m in _BRANCH_REF_RE.finditer(step_text):
        condition = m.group(1).strip(" ,;:")
        target = m.group(2)
        if condition and target:
            branches.append({"condition": condition, "next": target})
    return branches


def _heading_entries(blocks: list[dict]) -> list[tuple[int, int, str, int | None]]:
    """[(block_index, level, title, page), ...] for every heading block, in
    document order. Reuses chunk_tool's own heading-numbering regex so section
    depth is inferred the same way chunking already infers it."""
    from backend.chunking.chunk_tool import _get_heading_level

    out = []
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or b.get("type") != "heading":
            continue
        text = (b.get("text") or "").strip()
        if not text:
            continue
        ref = b.get("source_ref") or {}
        level = _get_heading_level(text, ref.get("bbox"))
        out.append((i, level, text, ref.get("page")))
    return out


def _locate_section(
    blocks: list[dict], section_hint: str | None, start_page: int | None,
) -> tuple[str, int, int] | None:
    """Returns (title, block_index, level) for the target heading, or None if
    neither section_hint nor start_page resolves to one."""
    headings = _heading_entries(blocks)
    if not headings:
        return None

    if section_hint:
        hint = section_hint.strip().lower()
        for idx, level, title, _page in headings:
            if hint in title.lower():
                return title, idx, level

    if start_page is not None:
        # The last heading AT OR BEFORE start_page -- the section that page
        # actually falls under, not necessarily one that starts exactly there.
        candidates = [(idx, level, title) for idx, level, title, page in headings
                      if page is not None and page <= start_page]
        if candidates:
            idx, level, title = candidates[-1]
            return title, idx, level

    return None


def _collect_section_text(
    blocks: list[dict], start_idx: int, level: int,
) -> tuple[int, int, list[tuple[int, int | None]], str]:
    """Body text of the section starting at start_idx, up to (not including) the
    next heading at the same or higher level. Returns (page_start, page_end,
    offsets, combined_text) -- offsets is [(char_offset, page), ...] in the
    combined text, used afterward to recover which page each parsed step
    belongs to without needing one block per step."""
    headings = _heading_entries(blocks)
    end_idx = len(blocks)
    for idx, hlevel, _title, _page in headings:
        if idx > start_idx and hlevel <= level:
            end_idx = idx
            break

    parts: list[str] = []
    offsets: list[tuple[int, int | None]] = []
    cursor = 0
    page_start: int | None = None
    page_end: int | None = None
    for b in blocks[start_idx:end_idx]:
        if not isinstance(b, dict) or b.get("type") == "heading":
            continue
        text = b.get("text") or ""
        if not text.strip():
            continue
        ref = b.get("source_ref") or {}
        page = ref.get("page")
        if page_start is None:
            page_start = page
        if page is not None:
            page_end = page if page_end is None else max(page_end, page)
        offsets.append((cursor, page))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" joiner below

    return page_start or 0, page_end or page_start or 0, offsets, "\n".join(parts)


def _page_for_offset(offsets: list[tuple[int, int | None]], pos: int) -> int | None:
    page = None
    for off, p in offsets:
        if off > pos:
            break
        page = p
    return page


def parse_procedure_from_blocks(
    blocks: list[dict],
    section_hint: str | None = None,
    start_page: int | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> dict[str, Any] | None:
    """Locate a section (by section_hint substring match, or the section
    start_page falls under) and parse its body into an ordered step graph.

    Returns None (never raises, never guesses) when:
      - neither section_hint nor start_page resolves to a real heading,
      - the section's body doesn't match the "(N) ..." step convention at all,
      - the matched step numbers aren't a clean, contiguous, increasing
        sequence starting from a real first step (a partial/garbled match is
        worse than admitting failure), or
      - more steps were found than max_steps (pathological over-split guard).

    On success: {"section_title": str, "page_range": [start, end],
                 "steps": {"1": {"text", "page", "next", "branches"?}, ...}}
    """
    located = _locate_section(blocks, section_hint, start_page)
    if located is None:
        return None
    title, idx, level = located

    page_start, page_end, offsets, combined_text = _collect_section_text(blocks, idx + 1, level)
    matches = list(_STEP_RE.finditer(combined_text))
    if not matches:
        return None

    step_numbers: list[int] = []
    steps: dict[str, dict] = {}
    for m in matches:
        text = m.group(2).strip()
        if not text:
            continue
        num = int(m.group(1))
        step_numbers.append(num)
        steps[str(num)] = {"text": text, "page": _page_for_offset(offsets, m.start()) or page_start}

    if not step_numbers:
        return None
    if len(step_numbers) > max_steps:
        return None
    # Must be a clean, contiguous, increasing sequence from its own first value --
    # a partial/garbled match (skipped numbers, out-of-order, duplicates) means
    # this isn't confidently the real step list, and presenting a broken
    # numbering to a technician mid-procedure is worse than refusing.
    if step_numbers != list(range(step_numbers[0], step_numbers[0] + len(step_numbers))):
        return None

    ordered_ids = [str(n) for n in step_numbers]
    for i, sid in enumerate(ordered_ids):
        steps[sid]["next"] = ordered_ids[i + 1] if i + 1 < len(ordered_ids) else None
        branches = _detect_branches(steps[sid]["text"])
        if branches:
            steps[sid]["branches"] = branches

    return {
        "section_title": title,
        "page_range": [page_start, page_end],
        "steps": steps,
        "first_step": ordered_ids[0],
    }
