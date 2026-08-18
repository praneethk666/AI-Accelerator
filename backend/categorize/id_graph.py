"""Deterministic cross-document ID linking -- the "correlate CAD/manual/circuit"
answer for corpora like Toyoda's, where every drawing carries its own strict,
machine-generated identifier (validated live against real files, 3-Aug):

  drawing_number  "00-83547001-0"   JTEKT title-block drawing number
  cad_sheet_id    "MS03AAA789AB"    CAD sheet's own drawing number (also usually
                                    its filename, e.g. MS03AAA789AB-spindle
                                    assembly.pdf)

These follow strict, unambiguous formats -- unlike free text, an exact regex
match is a genuine, 100%-precision link between documents: if a manual's
maintenance section and a CAD sheet both mention "MS03AAA789AB", they are
provably about the same drawing, no embedding similarity needed. This is
deliberately NOT a semantic/fuzzy system -- it complements vector retrieval by
covering the one thing embeddings are bad at (exact technical identifier match),
not replacing it.

Storage: no schema change. Matched IDs are tagged directly onto the block's own
`metadata` dict (already free-form JSONB on document_blocks, see
_checkpoint_page_blocks / write_blocks) as metadata["mentioned_ids"] = {kind: [ids]}.
Cross-document lookup then becomes a JSONB query over blocks already being
written anyway -- see find_documents_by_id().
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Order matters only for readability; each pattern is applied independently.
_ID_PATTERNS: dict[str, re.Pattern] = {
    "drawing_number": re.compile(r"\b\d{2}-\d{7,9}-\d\b"),
    "cad_sheet_id": re.compile(r"\bMS\d{2}[A-Z]{3}\d{3}[A-Z]{2}\b"),
    # ADDED 10-Aug: a THIRD real format family, found in an actual parts-list Excel's
    # "Drawing No" column (.../1.Expendable parts drawing/20230831_99Y_NEW-TIGG303_E.xlsx)
    # -- 27 distinct real values sampled, e.g. "KE-MC000D93-A", "TA-61BG0002-B",
    # "PA-GA006010-C". Shape: 2-char alphanumeric prefix (first char always a
    # letter) - 8-char alphanumeric body - 1-char letter suffix, always dash-
    # separated. Same CAD PDFs in this corpus are literally filenamed after this
    # exact value (e.g. "20230831_99Y_KE-MC000954-G.pdf") -- see
    # extract_id_from_filename() below, a complementary, regex-free join.
    # Deliberately does NOT match the plain-digits "drawing_number" pattern above
    # (that family's suffix is always a digit, this one's is always a letter), so
    # no double-tagging under two kinds for the same string.
    "parts_drawing_no": re.compile(r"\b[A-Z][A-Z0-9]-[A-Z0-9]{8}-[A-Z]\b"),
}


def _compile_extra_patterns(extra_patterns: dict[str, str] | None) -> dict[str, re.Pattern]:
    """Raw regex strings from a client's config profile (categorization.id_patterns)
    -> compiled patterns. A bad regex is logged and skipped, not fatal -- one
    malformed client-supplied pattern must not break ID extraction for everyone."""
    if not extra_patterns:
        return {}
    compiled: dict[str, re.Pattern] = {}
    for kind, raw in extra_patterns.items():
        try:
            compiled[kind] = re.compile(raw)
        except re.error:
            logger.warning("id_graph: skipping invalid extra_patterns regex for %r: %r", kind, raw)
    return compiled


def extract_ids(text: str, extra_patterns: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{"drawing_number": [...], "cad_sheet_id": [...]} for one string, deduped,
    first-seen order preserved. Empty dict (not per-key empty lists) for kinds
    with zero matches, so callers can trivially check `if ids: ...`.

    extra_patterns: optional {kind: raw_regex_string} from a client's config
    profile (categorization.id_patterns) -- lets a deployment recognize its own
    ID formats (e.g. a different customer's drawing-number convention) without a
    code change. Overrides a built-in kind of the same name; omitted/empty
    behaves identically to the built-in-only patterns."""
    if not text:
        return {}
    patterns = {**_ID_PATTERNS, **_compile_extra_patterns(extra_patterns)}
    out: dict[str, list[str]] = {}
    for kind, pattern in patterns.items():
        seen: list[str] = []
        for m in pattern.finditer(text):
            v = m.group(0)
            if v not in seen:
                seen.append(v)
        if seen:
            out[kind] = seen
    return out


def tag_blocks_with_ids(blocks: list[dict], extra_patterns: dict[str, str] | None = None) -> list[dict]:
    """Mutates each block in place: adds metadata["mentioned_ids"] wherever its
    own text contains a matched ID. Blocks with no match are left untouched (no
    empty key added -- keeps the common case's JSON small). Returns the same
    list (mutated), matching the in-place-mutation convention the rest of this
    pipeline uses for block metadata.

    extra_patterns: see extract_ids()."""
    for b in blocks:
        if not isinstance(b, dict):
            continue
        ids = extract_ids(b.get("text") or "", extra_patterns)
        if ids:
            meta = b.setdefault("metadata", {})
            meta["mentioned_ids"] = ids
            # Flat, kind-less list alongside the structured one above -- a plain
            # JSONB array-contains-element query (metadata->'mentioned_ids_flat'
            # @> '["X"]') is simple and index-friendly; querying "does ANY value
            # array under ANY key contain X" directly against the nested dict
            # shape needs jsonb_path_exists, which is more fragile across PG
            # versions. Kept both: nested for readability, flat for querying.
            flat: list[str] = []
            for id_list in ids.values():
                for v in id_list:
                    if v not in flat:
                        flat.append(v)
            meta["mentioned_ids_flat"] = flat
    return blocks


def extract_id_from_filename(filename: str) -> str | None:
    """A CAD drawing PDF in this corpus is often literally filenamed after its own
    drawing number (confirmed live, e.g. "20230831_99Y_KE-MC000954-G.pdf" — the
    exact same value as a parts-list Excel's "Drawing No" cell). No regex needed on
    the CALLER's side to exploit this: reuse the same _ID_PATTERNS against the
    filename's stem, which is far higher-precision than scanning prose (a filename
    has no surrounding words an ID pattern could spuriously match against). Returns
    the first match found, or None."""
    if not filename:
        return None
    import os
    stem = os.path.splitext(filename)[0]
    # \b in each pattern only fires at a transition to/from a NON-word character,
    # and regex treats "_" as a word character -- real filenames in this corpus
    # segment with underscores right up against the ID ("20230831_99Y_KE-MC000954-G"),
    # so \b silently never fires there. Underscores are a segment separator in a
    # filename, never part of an ID's own body -- swap them for spaces so \b can
    # actually see the boundary.
    stem = stem.replace("_", " ")
    for pattern in _ID_PATTERNS.values():
        m = pattern.search(stem)
        if m:
            return m.group(0)
    return None


def tag_blocks_with_filename_id(blocks: list[dict], filename: str) -> list[dict]:
    """Stamp the filename-derived ID (if any) onto every block's mentioned_ids --
    unconditional, no corpus_root gate needed (filename parsing needs no folder
    structure). Merged into whatever tag_blocks_with_ids() already found in the
    block's own text, not overwritten -- a block can legitimately mention its own
    ID string in its text too."""
    file_id = extract_id_from_filename(filename)
    if not file_id:
        return blocks
    for b in blocks:
        if not isinstance(b, dict):
            continue
        meta = b.setdefault("metadata", {})
        flat = meta.get("mentioned_ids_flat") or []
        if file_id not in flat:
            meta["mentioned_ids_flat"] = flat + [file_id]
        ids = meta.setdefault("mentioned_ids", {})
        by_filename = ids.setdefault("filename", [])
        if file_id not in by_filename:
            by_filename.append(file_id)
    return blocks


def document_id_summary(blocks: list[dict]) -> dict[str, list[str]]:
    """All distinct IDs mentioned ANYWHERE in one document's blocks, merged
    across kinds -- e.g. for a per-document summary/debug view. Call
    tag_blocks_with_ids() first (or pass already-tagged blocks); this just
    aggregates what's already on each block's metadata."""
    merged: dict[str, list[str]] = defaultdict(list)
    for b in blocks:
        if not isinstance(b, dict):
            continue
        for kind, ids in ((b.get("metadata") or {}).get("mentioned_ids") or {}).items():
            for v in ids:
                if v not in merged[kind]:
                    merged[kind].append(v)
    return dict(merged)


def find_documents_by_id(target_id: str, document_id: str | None = None) -> list[dict]:
    """Every block across the corpus (or, if document_id is given, just that one
    document -- useful for "does THIS document also mention this ID") whose
    metadata.mentioned_ids contains target_id, in any kind. Returns raw rows
    {document_id, block_id, type, text, source_ref} -- the caller decides how to
    present/dedupe by document. Real, actual DB query (not a pure function like
    the rest of this module) -- the whole point is CROSS-document lookup, which
    only the DB has visibility into."""
    from backend.storage.postgres_store import PostgresStore
    import json as _json

    pg = PostgresStore()
    try:
        sql = """
            SELECT document_id, block_id, type, text, source_ref
            FROM document_blocks
            WHERE metadata -> 'mentioned_ids_flat' @> %s::jsonb
        """
        params: list = [_json.dumps([target_id])]
        if document_id:
            sql += " AND document_id::text = %s"
            params.append(str(document_id))
        rows = pg.conn.execute(sql, params).fetchall()
    finally:
        pg.close()
    return [
        {"document_id": r[0], "block_id": r[1], "type": r[2], "text": r[3], "source_ref": r[4]}
        for r in rows
    ]
