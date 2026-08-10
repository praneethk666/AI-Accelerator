"""Folder-path-driven document routing for corpora that arrive in a consistent
machine/category folder structure (validated live against the real Toyoda TNGA
manual tree, 3-Aug). Two things this buys over per-file classification:

1. document_type comes from the folder name (deterministic, free, no VLM call)
   instead of the categorize LLM/vision guess -- more reliable than
   deployment.force_document_type for a MULTI-type corpus, since that override
   is a single global value and this corpus mixes manuals/CAD/circuits/parts
   lists under one root.
2. Every routed file also gets machine/doc_category/component tags -- the
   folder structure IS the correlation key a real user would expect ("show me
   everything about the spindle assembly on OP200"): a CAD drawing and a
   maintenance-manual section for the same component live in different folders
   but under the same machine, often the same component subfolder name.

This is intentionally a pure, standalone function -- NOT wired into
CategorizeTool yet. Attaching machine/doc_category/component onto every block's
existing free-form metadata dict (no schema change needed, document_blocks.metadata
is already JSONB) is the natural next step once this routing is validated on a
real corpus; retrieval-time filtering by these tags is the step after that.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# (substring to match in a folder name, case-insensitive) -> document_type.
# Order matters: first match wins, so more specific keywords go first.
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("instruction manual", "manual"),
    ("maintenance manual", "manual"),
    ("operation manual", "manual"),
    ("lubrication standard", "manual"),
    ("investigation sheet", "report"),
    ("machine assembly drawing", "cad_drawing"),
    ("expendable parts drawing", "cad_drawing"),
    ("service part list", "cad_drawing"),
    ("control related drawing", "circuit_diagram"),   # text-layer check may override
    ("pdf for printing drawing", "circuit_diagram"),   # text-layer check may override
    ("tooling layout", "cad_drawing"),
    ("electronic data", "unknown"),   # usually raw CAD source (DXF/DWG), not text/PDF
    # ADDED 10-Aug: this project's real corpus has a SECOND, unrelated numbering
    # scheme (the GD/BF-5 line) that shares almost none of the above phrases --
    # surveyed live via a direct folder listing, not guessed. Placed after the
    # original entries so nothing above changes match order/precedence.
    ("parts change procedure manual", "manual"),        # "08-Parts change procedure manual"
    ("lubricating manual", "manual"),                    # "Equipment Oiling & Lubricating Manual"
    ("circuit diagram", "circuit_diagram"),              # "Electrical circuit diagram", "Pneumatic circuit diagram"
    ("parts drawing", "cad_drawing"),                    # "Manufactured parts drawing（製作品）", "Purchased parts drawing（追加工図）"
    # A cycle/timing-sequence diagram isn't a literal electrical circuit --
    # "schematic" is the closest existing document_types fit, not strongly
    # validated (only 2 real folder names seen, no document content reviewed).
    ("cycle line diagram", "schematic"),                 # "Cycle line diagram", "Timing chart type cycle line diagram"
]

# A category-folder-shaped segment ("01-Foo", "08_Bar", "3.Baz") that fails every
# keyword above is a real, actionable gap -- log it so future unmatched schemes
# are discoverable in ingestion logs instead of a silent, unmeasured guess about
# corpus coverage. Deliberately loose (leading digits + separator) so it fires on
# both this corpus's numbering conventions (TNGA's "3.INSTRUCTION MANUAL", GD's
# "08-Parts change...") without needing to enumerate every real prefix style.
_NUMBERED_CATEGORY_RE = re.compile(r"^\d{1,2}[.\-_]")

# document_type values whose folder-keyword hit should be double-checked against
# whether the actual PDF has a real text layer -- these categories in this corpus
# contain BOTH huge single-sheet vector drawings (no text layer -> genuinely need
# cad_route/tiling) AND standard-page-size print sets with a full text layer
# (confirmed live: a 1147-page "Electric circuit diagram(MAIN).pdf" has real,
# searchable text on every page -- routing that through cad_route would fire a
# VLM call per page for no reason; text_default handles it for near-zero cost).
_TEXT_LAYER_SENSITIVE = {"circuit_diagram", "cad_drawing"}


def _has_text_layer(pdf_path: str, sample_pages: int = 3, min_chars: int = 40) -> bool | None:
    """True/False, or None if it can't be determined (missing file, not a PDF,
    open failure) -- callers should treat None as "unknown, don't override"."""
    if not pdf_path.lower().endswith(".pdf") or not os.path.isfile(pdf_path):
        return None
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            n = min(sample_pages, len(doc))
            total = sum(len(doc[i].get_text().strip()) for i in range(n))
            return total >= min_chars
        finally:
            doc.close()
    except Exception:
        return None


def route_from_path(file_path: str, corpus_root: str) -> dict[str, Any] | None:
    """Derive {machine, doc_category, component, force_document_type,
    force_route (rarely), reasoning} from file_path's position under corpus_root.
    Returns None if file_path isn't under corpus_root, or no category folder in
    the path matched a known keyword (caller should fall back to normal
    categorize / a manual override in that case).

    machine     = the top-level folder directly under corpus_root (e.g.
                  "120_CYLINDRICAL GRINDER", "OP200_TIGG300_HONING MC_JTEKT").
    doc_category = the matched category folder's own name, verbatim (e.g.
                  "3.INSTRUCTION MANUAL", "04_Machine assembly drawing").
    component   = the folder immediately containing the file, ONLY when it's
                  neither the machine nor the doc_category folder itself (e.g.
                  "04_Spindlehead" under ".../04_Machine assembly drawing/") --
                  None when the file sits directly in the category folder.
    """
    try:
        rel = os.path.relpath(file_path, corpus_root)
    except ValueError:
        return None
    if rel.startswith(".."):
        return None
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None  # file sits directly at corpus_root -- no machine folder

    machine = parts[0]
    doc_type = None
    doc_category = None
    for seg in parts[1:-1]:
        for kw, dt in _CATEGORY_KEYWORDS:
            if kw in seg.lower():
                doc_type, doc_category = dt, seg
                break
        if doc_type:
            break
    if doc_type is None:
        for seg in parts[1:-1]:
            if _NUMBERED_CATEGORY_RE.match(seg):
                logger.info(
                    "folder_router: numbered category folder %r under machine %r matched "
                    "no keyword -- a real, unmapped scheme, not a crash. File falls through "
                    "to normal categorize.", seg, machine,
                )
        return None

    # cad_drawing/circuit_diagram map to a route_extractor override (cad_route/
    # circuit_route) that REPLACES the normal file_type-based extractor entirely
    # (backend/pipeline/graph.py) -- forcing either on a non-PDF (e.g. the .xlsx
    # BOM sitting in an "Expendable parts drawing" folder) would route it into
    # cad_extract.py, which opens files via fitz and would just crash on it.
    # Keep the machine/doc_category/component tags either way -- only the force
    # is suppressed -- so it still falls through to normal categorize, and its
    # file_type extractor (excel/ppt/word) still gets picked correctly.
    if doc_type in ("cad_drawing", "circuit_diagram") and not file_path.lower().endswith(".pdf"):
        doc_type = None

    if doc_type in _TEXT_LAYER_SENSITIVE:
        has_text = _has_text_layer(file_path)
        if has_text is True:
            doc_type = "manual"   # real text layer -> cheap text_default, not cad_route

    cat_idx = parts.index(doc_category)
    component = None
    if cat_idx < len(parts) - 2:   # something between the category folder and the file
        component = parts[cat_idx + 1]

    return {
        "machine": machine,
        "doc_category": doc_category,
        "component": component,
        "force_document_type": doc_type,
        "reasoning": f"folder_router: matched '{doc_category}' under machine '{machine}'"
                     + (f", component '{component}'" if component else ""),
    }


def tag_blocks_with_folder_info(blocks: list[dict], file_path: str, corpus_root: str) -> list[dict]:
    """Stamp machine/doc_category/component onto every block's metadata (mutates
    in place, same convention as id_graph.tag_blocks_with_ids) -- the OTHER half
    of cross-document correlation alongside exact drawing/part-number matching:
    "everything under this machine/component" vs "everything mentioning this
    exact drawing number". No-op if the path doesn't match a known corpus
    structure (route_from_path returned None) -- most deployments won't set
    deployment.corpus_root at all, and this must be silently inert then."""
    info = route_from_path(file_path, corpus_root)
    if not info:
        return blocks
    tag = {k: v for k, v in info.items() if k in ("machine", "doc_category", "component") and v}
    for b in blocks:
        if isinstance(b, dict):
            b.setdefault("metadata", {})["folder"] = tag
    return blocks
