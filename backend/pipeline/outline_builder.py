"""Builds a navigable document_outline (chapter -> section -> subsection tree) so
the agent can locate an exact section by NAVIGATING a document's own structure,
instead of trusting semantic search to happen to retrieve the right chunk among
several similarly-worded procedures (real risk in this corpus: "Replacing the
Workpiece Holder" / "Replacing the Phase Datum Pad" / "Replacing the Pad of Phase
Indexing Unit" are three distinct procedures in one manual).

Research basis (external, 10-Aug): PageIndex-style tree navigation -- an LLM walks
a document's own structure node by node rather than similarity-searching flat
chunks -- is validated on exactly this document class (long, hierarchical
technical manuals). See the plan doc for sources.

Two sources, tried in order, per document:
1. Real PDF bookmarks (fitz.get_toc()) -- free, instant, when the source PDF has
   them (confirmed on real corpus files: TOYOPUC-PLUS.pdf has 388, WJ200Series.pdf
   has 13).
2. Heading stack derived during extraction -- reuses chunk_tool._get_heading_level's
   numbering regex over the document's own `heading`-type blocks, for documents with
   numbered headings but no embedded bookmarks (e.g. the changeover manual: "1." /
   "1.1" / "1.2" numbered sections).

Fails INERT (writes nothing), not just closed: fewer than ~3 detected headings/
bookmarks is a normal, common, silently-fine state (a CAD sheet, a short flat
document), never an error -- callers must treat "no outline" as "fall back to
search_documents," never retry or warn.

Building is UNCONDITIONAL (no config gate) -- cheap, deterministic, no LLM call,
degrades to nothing harmlessly. Only the AGENT TOOL that exposes this
(backend/retrieval/browse_document_outline.py) is config-gated pending a live
validation pass, so a document already has its outline the moment the tool is
turned on rather than needing re-ingestion.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MIN_NODES = 3  # fewer than this is not a real outline -- see module docstring


def _fill_page_ends(nodes: list[dict], total_pages: int) -> None:
    """Each node's page_end = the page before the next node at the SAME OR
    HIGHER level (its next sibling, or its parent's next sibling), or the
    document's last page if it's the final node in its branch."""
    for i, node in enumerate(nodes):
        end = total_pages
        for later in nodes[i + 1:]:
            if later["level"] <= node["level"]:
                end = later["page_start"] - 1
                break
        node["page_end"] = max(end, node["page_start"])


def _nodes_from_flat_list(entries: list[tuple[int, str, int]], source: str) -> list[dict]:
    """entries: [(level, title, page_start), ...] in document order. Builds
    parent_id via a level-indexed stack -- a node's parent is the nearest
    preceding entry with a STRICTLY lower level."""
    nodes: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, node_id)
    for i, (level, title, page) in enumerate(entries):
        node_id = f"n{i}"
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_id = stack[-1][1] if stack else None
        nodes.append({
            "node_id": node_id, "parent_id": parent_id, "title": title,
            "level": level, "page_start": page, "page_end": None, "source": source,
        })
        stack.append((level, node_id))
    return nodes


def _outline_from_bookmarks(pdf_path: str) -> list[dict] | None:
    if not pdf_path:
        return None
    try:
        import fitz
        doc = fitz.open(pdf_path)
    except Exception:
        return None
    try:
        toc = doc.get_toc()
        total_pages = doc.page_count
    except Exception:
        return None
    finally:
        doc.close()

    if len(toc) < _MIN_NODES:
        return None
    entries = [(level, (title or "").strip(), page) for level, title, page in toc
               if title and page]
    if len(entries) < _MIN_NODES:
        return None
    nodes = _nodes_from_flat_list(entries, "pdf_bookmark")
    _fill_page_ends(nodes, total_pages)
    return nodes


def _outline_from_headings(blocks: list[dict]) -> list[dict] | None:
    from backend.chunking.chunk_tool import _get_heading_level

    entries: list[tuple[int, str, int]] = []
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "heading":
            continue
        text = (b.get("text") or "").strip()
        if not text:
            continue
        ref = b.get("source_ref") or {}
        page = ref.get("page")
        if page is None:
            continue
        level = _get_heading_level(text, ref.get("bbox"))
        entries.append((level, text, page))

    if len(entries) < _MIN_NODES:
        return None
    max_page = max(p for _, _, p in entries)
    nodes = _nodes_from_flat_list(entries, "heading_detect")
    _fill_page_ends(nodes, max_page)
    return nodes


def build_outline(pdf_path: str | None, blocks: list[dict]) -> list[dict] | None:
    """Try real PDF bookmarks first, fall back to heading detection. Returns
    None (not []) when neither source yields a real outline -- the document-
    type-dependent "silently inert" behavior described in the module docstring."""
    nodes = _outline_from_bookmarks(pdf_path) if pdf_path else None
    if nodes:
        return nodes
    return _outline_from_headings(blocks)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS document_outline (
    id           BIGSERIAL PRIMARY KEY,
    document_id  UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    node_id      TEXT NOT NULL,
    parent_id    TEXT,
    title        TEXT,
    level        INTEGER,
    page_start   INTEGER,
    page_end     INTEGER,
    source       TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, node_id)
)
"""


def _ensure_schema(pg) -> None:
    # Lazy runtime migration -- same pattern as conversation_store.py's
    # _ensure_schema, since this project's live Supabase DB doesn't get
    # scripts/init_db.sql re-run against it. MUST stay in sync with that file.
    pg.conn.execute(_SCHEMA_SQL)
    pg.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_outline_doc "
        "ON document_outline (document_id, parent_id)"
    )


def write_document_outline(document_id: str, nodes: list[dict] | None) -> None:
    """No-op if nodes is None/empty -- most documents in a corpus won't have a
    detectable outline, and that must never write empty rows or raise."""
    if not nodes:
        return
    from backend.storage.postgres_store import PostgresStore
    pg = PostgresStore()
    try:
        _ensure_schema(pg)
        # Re-ingestion idempotency -- same convention as QdrantStore.delete_by_document
        # before re-indexing: clear any stale outline for this doc first.
        pg.conn.execute("DELETE FROM document_outline WHERE document_id = %s", [document_id])
        for n in nodes:
            pg.conn.execute(
                "INSERT INTO document_outline "
                "(document_id, node_id, parent_id, title, level, page_start, page_end, source) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [document_id, n["node_id"], n.get("parent_id"), n["title"], n["level"],
                 n.get("page_start"), n.get("page_end"), n["source"]],
            )
    finally:
        pg.close()


def get_outline_children(document_id: str, node_id: str | None = None) -> list[dict]:
    """Children of node_id (or the root's top-level nodes when node_id is None),
    ordered by page. Real DB lookup, matching id_graph.py::find_documents_by_id's
    own "not a pure function" pattern -- the whole point is a fresh read of
    ingest-time state."""
    from backend.storage.postgres_store import PostgresStore
    pg = PostgresStore()
    try:
        _ensure_schema(pg)
        if node_id is None:
            sql = ("SELECT node_id, title, level, page_start, page_end FROM document_outline "
                   "WHERE document_id = %s AND parent_id IS NULL ORDER BY page_start")
            params = [document_id]
        else:
            sql = ("SELECT node_id, title, level, page_start, page_end FROM document_outline "
                   "WHERE document_id = %s AND parent_id = %s ORDER BY page_start")
            params = [document_id, node_id]
        rows = pg.conn.execute(sql, params).fetchall()
    finally:
        pg.close()
    return [{"node_id": r[0], "title": r[1], "level": r[2], "page_start": r[3], "page_end": r[4]}
            for r in rows]
