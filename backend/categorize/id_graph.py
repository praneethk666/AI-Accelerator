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

import re
from collections import defaultdict

# Order matters only for readability; each pattern is applied independently.
_ID_PATTERNS: dict[str, re.Pattern] = {
    "drawing_number": re.compile(r"\b\d{2}-\d{7,9}-\d\b"),
    "cad_sheet_id": re.compile(r"\bMS\d{2}[A-Z]{3}\d{3}[A-Z]{2}\b"),
}


def extract_ids(text: str) -> dict[str, list[str]]:
    """{"drawing_number": [...], "cad_sheet_id": [...]} for one string, deduped,
    first-seen order preserved. Empty dict (not per-key empty lists) for kinds
    with zero matches, so callers can trivially check `if ids: ...`."""
    if not text:
        return {}
    out: dict[str, list[str]] = {}
    for kind, pattern in _ID_PATTERNS.items():
        seen: list[str] = []
        for m in pattern.finditer(text):
            v = m.group(0)
            if v not in seen:
                seen.append(v)
        if seen:
            out[kind] = seen
    return out


def tag_blocks_with_ids(blocks: list[dict]) -> list[dict]:
    """Mutates each block in place: adds metadata["mentioned_ids"] wherever its
    own text contains a matched ID. Blocks with no match are left untouched (no
    empty key added -- keeps the common case's JSON small). Returns the same
    list (mutated), matching the in-place-mutation convention the rest of this
    pipeline uses for block metadata."""
    for b in blocks:
        if not isinstance(b, dict):
            continue
        ids = extract_ids(b.get("text") or "")
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
