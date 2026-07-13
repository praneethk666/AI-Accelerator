# Extraction

Every extractor turns one input file into a `NormalizedBlock[]` (text, table,
image_caption, heading — see `backend/core/schemas.py`), following the same
`Tool.run(state, config)` contract (`backend/core/tool.py`) so the pipeline graph
can call any of them the same way.

| Input | Tool | Module |
|---|---|---|
| PDF (digital, scanned, or mixed) | `DoclingPDFTool` | `docling_pdf/` |
| Excel (.xlsx/.xls/.xlsm) | `ExcelExtractorTool` | `excel/` |
| PowerPoint (.pptx/.ppt) | `PPTExtractorTool` | `ppt/` |
| Word (.docx/.doc) | `WordExtractorTool` | `word/` |
| Standalone image (.jpg/.png/.tif) | `ImageExtractorTool` | `image/` |
| CAD / circuit drawings (route override) | `CADExtractionTool` | `cad/` |

## PDF: one extractor for all three kinds

`docling_pdf/` is the only PDF extractor — one hybrid tool handles digital,
scanned, and mixed PDFs (a per-page router picks native text, Docling's layout +
table model, or a VLM rescue, per page — see `page_router.py`). There is no
separate "digital" vs "scanned" extractor module; `categorize`'s `detector.py`
still classifies a PDF's overall kind (`state["pdf_kind"]`) so the graph/config can
route CAD-style documents differently, but `docling_pdf` extracts all of them.

Supporting modules used by the PDF path:
- `detector.py` — classifies a PDF as digital / scanned / mixed.
- `page_router.py` — per-page signals -> extraction method (Docling native /
  table-escalate / VLM).
- `orientation.py` — corrects sideways scans before rendering.
- `large_format.py` — resolution-driven tiling for large CAD/circuit sheets so
  small reference designators survive downsampling.
- `table_reconcile.py` — stitches a table that spans multiple pages back into one.
- `vision_ocr.py` — routes COMPLEX pages (garbled text, tables, figures) through a
  VLM for transcription instead of native/OCR text; see the root README's "Vision"
  section for the full flow.
- `scanned_pdf/` — OCR/layout utility functions (`page_to_pil`, layout-region
  detection, Surya structure recognition) consumed by `vision_ocr.py`. Not a
  standalone extractor — despite the folder name, there's no separately-registered
  "scanned PDF" tool anymore; docling_pdf covers scanned PDFs directly.

## Registration

Every extractor is registered in `backend/pipeline/default_registry.py` and
enabled per-route in `config/global.yaml` (`extractors`, `pdf_extractors`,
`route_extractors` — see the root README's "Config" section for how routing works).
