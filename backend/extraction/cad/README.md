# CAD Extraction Module

The CAD Extraction module parses engineering sheets, blueprints, and schematic diagrams (both PDF format and images) into structured region blocks.

## Dependencies

* **PyMuPDF (`fitz`)**: Used to open, render, and inspect dimensions of PDF drawings.
* **backend.core.vision_client (`describe_image`)**: Connects to multimodal vision models (Gemini, OpenAI, or local vLLM endpoints).
* **backend.extraction.large_format**: Handles grid-tiling logic for oversized drawings.

## Step-by-Step Logic

The pipeline entrypoint is `CADExtractionTool::run()`:

1. **Route Resolution**:
   * Evaluates the active routing profile:
     * `cad_route` (mechanical blueprints) -> Selects mechanical prompt structures.
     * `circuit_route` (electrical/PCB schematics) -> Selects electrical block and wiring prompts.
2. **Large-Format Tiling**:
   * If a drawing sheet exceeds standard paper sizes (A2, A3, E-size, or custom blueprints), a normal VLM resize will shrink text labels, making part numbers and designators unreadable.
   * Oversized sheets are rendered at a high resolution (default: 300 DPI) and split into a grid of overlapping tiles (`large_format` module). Each tile is transcribed individually by the VLM.
3. **Region Identification Prompting**:
   * For normal sheets, it sends the full page rendering to the VLM, requesting a structured JSON array containing recognized sections:
     * `title_block`: Project name, author, scale, drawings IDs.
     * `table`: Parts tables, revision blocks, wiring lists.
     * `notes`: Drawing legends, reference notes.
4. **JSON Array Recovery (`_region_blocks` / `_balanced_objects`)**:
   * Standard VLM endpoints can hit token constraints and return truncated JSON arrays for complex blueprints.
   * The tool uses regular expression parsers (`_balanced_objects`) to isolate and extract complete JSON structures from within a broken/truncated array, salvaging extracted sections instead of failing the page.
5. **Ingestion Safeguards**:
   * The tool returns `pending_vision: False` for all generated blocks, informing the downstream vision pipeline that visual transcription is already completed.
