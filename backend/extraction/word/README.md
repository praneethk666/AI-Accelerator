# Microsoft Word Extraction Module

The Word Extraction module parses Word documents (`.docx`, `.doc`) and converts them into text, heading, table, and image blocks.

## Dependencies

* **python-docx (`docx`)**: Core parser for reading Word open-xml schemas.
* **pandas**: Reconstructs and formats tables.
* **langdetect**: Detects paragraph languages.
* **backend.extraction.ppt.tool (`_save_image_blob`, `_detect_image_ext`)**: Saves raw visual data crops and creates PNG previews.

## Step-by-Step Logic

The pipeline entrypoint is `WordExtractorTool::run()`:

1. **Format Validation**:
   * Inspects the file extension. Rejects legacy binary `.doc` formats, appending a conversion warning to `state["errors"]` (only OpenXML `.docx` files are supported by python-docx).
2. **Document Structure Parsing**:
   * Opens the file using `docx.Document()`.
   * **Paragraph Extraction (`_paragraphs`)**: Iterates through paragraphs, identifying text layers.
     * *Heading Detection*: Matches styling tags. If a style begins with `"heading"` or `"title"`, it creates a `"heading"` block; otherwise, it creates a `"text"` block.
   * **Table Extraction (`_tables`)**: Iterates through document tables.
     * Translates cells row-by-row into grid tables.
     * Converts the table data into a markdown table representation and builds structured `headers` and `rows` lists inside `table_data` for citations.
   * **Image Extraction (`_images`)**: Scans XML slide relationships to find inline images.
     * Sniffs the format using magic bytes to prevent image file corruption.
     * Saves raw image files and creates PNG previews inside `uploads/images/{doc_id}/`.
     * Registers them as `pending_vision: True` caption blocks for downstream vision captioning.
3. **Registration**:
   * Saves the parsed blocks list to `state["blocks"]`.
