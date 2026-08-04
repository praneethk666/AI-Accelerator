"""Centralized prompt library for every VLM/LLM task in extraction.

One place so prompts are reviewable, versioned, and consistent — instead of flat
strings scattered across modules. All prompts are DOCUMENT-AGNOSTIC: they describe
the task and its edge cases, never a specific corpus (Argo/Mendoza are only test
fixtures). They must work for manuals, invoices, BOQs, quotes, forms, spec sheets,
PPT/Word-exported PDFs, scans, and engineering drawings (CAD / circuit schematics).

Conventions every prompt enforces:
  - VERBATIM identifiers: copy numbers, units, currency, part/serial/reference
    numbers, dates and codes exactly — never normalize, round, translate or invent.
  - Reading order: top-to-bottom; for multi-column layouts, finish the left column
    before the right. Respect rotated/landscape orientation.
  - Say [illegible] rather than guessing; transcribe each piece of content ONCE.
"""
from __future__ import annotations

# --- Shared rule blocks (composed into the task prompts) ---------------------
_VERBATIM = (
    "Copy all numbers, units, currency amounts, dates, part/serial/reference numbers "
    "and codes EXACTLY as written — never normalize, round, reformat, translate or "
    "guess them. If something is unreadable, write [illegible] instead of inventing it."
)
_ONCE = "Transcribe each distinct piece of content ONCE; do not repeat or duplicate."
_NO_COMMENTARY = "Do NOT summarize, explain, translate, or add any commentary."


# --- Full-page transcription (OCR + structure) -------------------------------
PAGE_TRANSCRIBE = (
    "You are a precise OCR + document-structure engine. Transcribe EVERYTHING on this "
    "page image as clean Markdown, preserving the original reading order.\n"
    "- Reading order: top-to-bottom. For multi-column layouts, transcribe the LEFT "
    "column fully before the right. Honor the page's actual orientation (rotated or "
    "landscape pages read in their upright direction).\n"
    "- Tables: render as GitHub Markdown tables with a single header row. Keep "
    "merged/spanning headers as repeated cells. Preserve every row and column; keep "
    "blank cells blank.\n"
    "- Forms / key-value layouts: render each labeled field as `Label: value` on its "
    "own line. Keep checkbox/selection state (e.g. [x] / [ ]).\n"
    "- Lists: keep bullet and numbered list structure.\n"
    "- Headings/titles: keep them as Markdown headings.\n"
    "- Stamps, handwriting, signatures, watermarks: transcribe if legible, else note "
    "them briefly in brackets (e.g. [stamp: PAID], [handwritten], [signature]).\n"
    f"- {_VERBATIM}\n"
    f"- {_ONCE}\n"
    f"{_NO_COMMENTARY} Output ONLY the page content as Markdown."
)


# --- One figure crop: classify AND caption in a single call ------------------
# The detectors (Docling/YOLO) over-propose: logos, banners, headers, rules and text
# blocks come back tagged as "picture". This call is the SEMANTIC GATE — the model
# says what the crop actually is and whether it's real content worth indexing — so we
# need no geometry thresholds (which can't tell a logo from a wide schematic).
FIGURE_CLASSIFY_CAPTION = (
    "Look at this image cropped from a document page. Decide WHAT it is and whether it "
    "is meaningful content worth indexing, then describe it.\n\n"
    "Return ONLY a JSON object:\n"
    '{"kind": "<one of: photo, diagram, schematic, circuit, cad_drawing, chart, '
    'flowchart, map, screenshot, illustration, table, logo, banner, header_footer, '
    'decoration, rule_line, text, blank>", '
    '"keep": <true|false>, '
    '"reasoning": "<explain keep/discard decision here>", '
    '"caption": "<description, see rules>"}\n\n'
    "Rules:\n"
    "- keep=false (and caption=\"\") when kind is logo, banner, header_footer, "
    "decoration, rule_line, text or blank — these are page furniture, not content.\n"
    "- keep=true for real visual content (photo, diagram, schematic, circuit, "
    "cad_drawing, chart, flowchart, map, screenshot, illustration, table).\n"
    "- caption for kept content: 1-3 precise sentences naming visible labels, part "
    "numbers, callouts and components. For schematic/circuit/cad_drawing, also list "
    "the key components with their reference designators/values and the connections "
    "you can read. For chart, state the chart type, axes and the trend/key values.\n"
    "- Do not include keep/discard reasons, keep=true/false justifications, or other internal "
    "reasoning inside the caption field. All keep/discard commentary must be put strictly in the reasoning field.\n"
    f"- {_VERBATIM}\n"
    "Describe ONLY what the image shows; use page context only to identify it. No text "
    "outside the JSON."
)

# Page context is appended to the caption call so the model can identify ambiguous crops.
def figure_prompt(page_context: str = "", known_caption: str = "") -> str:
    p = FIGURE_CLASSIFY_CAPTION
    if known_caption:
        # This is the document's OWN caption text for THIS figure (Docling linked it via
        # its text layer, not a guess) -- authoritative, unlike page_context below.
        p += ("\n\nKNOWN CAPTION FOR THIS EXACT FIGURE (from the document's text layer, "
              "verbatim, authoritative — start your caption with this text unless it "
              "obviously describes something else, then extend it with what you see):\n"
              + known_caption[:500])
    if page_context:
        p += "\n\nPAGE CONTEXT (for identification only):\n" + page_context[:1000]
    return p


# --- A TILE of a large engineering drawing (CAD / circuit / E-size sheet) -----
# Large-format sheets exceed a VLM's legible resolution when sent whole, so we render
# at high DPI and split into overlapping tiles; each tile is transcribed at full detail
# and the parts are merged. This prompt transcribes ONE tile.
SCHEMATIC_TILE = (
    "This image is ONE TILE (a sub-region) of a larger engineering drawing — a "
    "schematic, circuit, or CAD/mechanical drawing. Transcribe EVERYTHING visible in "
    "THIS tile at full detail:\n"
    "- Components with their reference designators and values "
    "(e.g. `R12 = 10k`, `C4 = 100nF`, `U3 = LM358`, `Q1 = 2N2222`).\n"
    "- Wire/net labels, bus names, pin numbers and the connections you can trace.\n"
    "- Dimensions, tolerances, callouts, balloon numbers, weld/finish symbols.\n"
    "- Any notes, legends, and title-block fields (drawing no., revision, sheet, "
    "scale, material, author, date).\n"
    f"- {_VERBATIM}\n"
    "Return Markdown (use a `Components:` list and a `Connections:`/`Notes:` list where "
    "they apply). Transcribe the tile's CONTENTS; do not describe it in prose. If the "
    "tile is empty drawing space, return an empty response."
)

# --- COARSE region locator for large-format sheets (agentic zoom, not blind tiling) ---
# The alternative to exhaustive grid tiling (SCHEMATIC_TILE above): one cheap call on a
# low-res render to find WHERE the real content is, then a targeted high-res crop per
# region (see backend/extraction/large_format.py::transcribe_large_page_regions). Real
# 2025-2026 research pattern -- "coarse-to-fine" / "localized zoom" (Zoom-Refine,
# ZoomEye) -- locate first, zoom only where it matters, instead of mechanically slicing
# the whole sheet into a fixed grid regardless of content density.
LOCATE_REGIONS = (
    "You are looking at a technical engineering drawing sheet (CAD assembly, circuit "
    "diagram, or similar large-format print). Identify the DISTINCT REGIONS on this "
    "sheet — do NOT transcribe their contents in detail yet, just locate and briefly "
    "describe each one. This is a coarse pass; fine print does not need to be legible "
    "to you right now.\n\n"
    "For each region:\n"
    '- "type": one of "table" (title block, parts list, revision table, connector '
    'pinout, etc.) or "view" (a drawing/section/detail view, schematic diagram) or '
    '"text" (notes/annotations block).\n'
    '- "label": a short slug identifying it, e.g. "title_block", "parts_list_main", '
    '"view_A-A", "notes". If a sheet has two separate tables with the same kind of '
    'content, give them DISTINCT labels (e.g. "parts_list_main", "parts_list_sub") — '
    "never merge two physically separate regions into one entry.\n"
    '- "description": ONE short sentence describing what this region visually '
    'contains (e.g. "Parts table with item numbers and part numbers, ~15 rows" or '
    '"Section view A-A showing the spindle bore and bearing seats").\n'
    '- "bbox": [x1, y1, x2, y2] normalized 0.0-1.0, tightly enclosing JUST this '
    "region — err toward slightly wider rather than clipping content.\n\n"
    "Identify EVERY distinct region on the sheet, even small ones (a small parts "
    "table or a single detail view still counts). A region must be visually "
    "separable — its own border/grid, or a clearly distinct area of the sheet — "
    "don't split one continuous view into multiple regions, and don't invent a "
    "region that isn't really there.\n\n"
    "Return a JSON array only. No markdown fences. No text outside the array.\n"
    '[{"type": "table", "label": "title_block", "description": "...", '
    '"bbox": [0.7, 0.0, 1.0, 0.15]}, ...]'
)

# Merge the per-tile transcriptions of one drawing into a single deduplicated summary.
SCHEMATIC_MERGE = (
    "Below are transcriptions of overlapping TILES of ONE engineering drawing, in "
    "row-major order. Merge them into a single coherent description of the whole "
    "drawing. Deduplicate content that appears in overlapping tiles (each component, "
    "net, note and title-block field should appear ONCE). Preserve every distinct "
    "component (with designator/value), connection, dimension, note and title-block "
    "field, VERBATIM. Output Markdown with sections: `Title block`, `Components`, "
    "`Connections`, `Notes`. No commentary.\n\nTILES:\n"
)


# --- A cropped TABLE region (used when a table is escalated to the VLM) -------
TABLE_TRANSCRIBE = (
    "Transcribe ONLY the table in this image as a GitHub Markdown table. Preserve "
    "every row and column. Keep merged/spanning headers as repeated cells, and keep "
    "blank cells blank. Do not merge or split cells, and do not let a wrapped line in "
    "one cell bleed into another column.\n"
    f"- {_VERBATIM}\n"
    "Output ONLY the Markdown table, nothing else."
)


# --- Cross-page table stitch arbitration (LLM, text-only) --------------------
STITCH_CONTINUATION = (
    "Two tables appear on consecutive PDF pages with the same number of columns. Is "
    "table B a CONTINUATION of table A (the same table split across a page break), or a "
    "SEPARATE table that merely shares the column count? Answer with one word: "
    "CONTINUATION or SEPARATE.\n\nTABLE A:\n{a}\n\nTABLE B:\n{b}"
)
