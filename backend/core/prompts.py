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
