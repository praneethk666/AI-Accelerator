# Docling PDF Extraction Module

The Docling PDF Extraction module implements a unified, high-fidelity parser for PDF documents (supporting digital, scanned, or mixed formats).

## Dependencies

* **docling**: IBM/MIT Document Ingestion Engine. Used for parsing document layout hierarchies, reading order, and table cell matrices.
* **docling.datamodel**: Handles configuration mapping (`PdfPipelineOptions`, `AcceleratorOptions`, `AcceleratorDevice`) for hardware acceleration (CPU, CUDA, MPS).
* **PyMuPDF (`fitz`)**: Handles page coordinate conversions and structural bounding box crops.
* **backend.extraction.vision_ocr (`route_and_rescue`)**: Directs VLM transcription rescue loops for garbled/scanned pages.
* **backend.extraction.table_reconcile (`reconcile_tables`)**: Merges split tables that span page boundaries.

## Ingest Processing Logic

The entrypoint is `DoclingPDFTool::run()`:

1. **Converter Cache Initialization (`_converter`)**:
   * Loads docling model singletons (layout classification and TableFormer models).
   * Configures thread limits and target execution devices (`device: cpu|cuda|mps|auto`).
2. **Structural Document Conversion**:
   * Parses the PDF into a hierarchical `DoclingDocument`.
   * Filters out page headers and page footers (`page_header`, `page_footer`) to prevent indexing noise.
   * Maps headings (`section_header`, `title`) and tables into standard `NormalizedBlock` schemas in logical reading order.
3. **VLM Rescue Routing (`route_and_rescue`)**:
   * Inspects extracted blocks. If pages are scanned (no text layer) or garbled, the router bypasses native parses and runs a VLM-based page transcription.
4. **Table Reconciliation (`reconcile_tables`)**:
   * Scans sequential tables. If a table spans multiple pages (where continuation rows inherit the header layout from the previous page), it stitches them back into a single structured block.
5. **Ingestion Metadata Overrides**:
   * Overrides `state["page_profiles"] = []` to prevent the downstream `VisionEnrichmentTool` from reprocessing figures, as Docling captures and captions illustrations directly.
