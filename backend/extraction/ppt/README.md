# PowerPoint Extraction Module

The PowerPoint Extraction module parses slides (`.pptx`, `.ppt`) and structures them into text, tables, and visual blocks.

## Dependencies

* **python-pptx (`pptx`)**: Core library for reading PPTX elements.
* **zipfile / re**: Parses text directly from PPTX XML structures as a backup.
* **Pillow (`PIL`)**: Processes and saves embedded slide illustrations and diagrams.

## Step-by-Step Logic

The extraction process runs in `PPTExtractorTool::run()`:

1. **Format Validation**:
   * Rejects binary legacy `.ppt` files, logging an error in the state since python-pptx only supports open-xml `.pptx` formats.
2. **Slide Scanning**:
   * Loops through slides, scanning shapes recursively using `_iter_shapes()` (which resolves nested slide group shapes).
   * **Text Shapes**: Extracts text from text frames, concatenates paragraphs, and appends a `text` block.
   * **Table Shapes**: Extracts text from table shapes row-by-row, formatting it as a markdown table.
   * **Visual Shapes**: Extracts embedded images, sniffs formats via magic bytes, saves files under `uploads/images/`, and appends a `pending_vision` block.
3. **VLM Slide Fallback**:
   * If slide shapes contain minimal text (e.g., slides made entirely of images), the slide is flagged as `pending_vision` so the vision pipeline captions it.
4. **XML Extraction Fallback (`_fallback_from_pptx_xml`)**:
   * If python-pptx fails to read a slide deck due to corruption, the tool opens the `.pptx` file as a zip archive, reads raw slide XMLs (`ppt/slides/slideN.xml`), removes XML tags using regular expressions, and extracts the plain text.
