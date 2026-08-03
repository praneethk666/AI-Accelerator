# PyMuPDF PDF Extraction Module

The PyMuPDF PDF Extraction module implements a fast, CPU-native parser for digital PDF documents.

## Dependencies

* **PyMuPDF (`fitz`)**: Core engine for parsing PDF layouts, bounding boxes, text frames, and tables.
* **uuid**: Generates unique identifiers for parsed document blocks.

## Step-by-Step Logic

The pipeline entrypoint is `PyMuPDFTool::run()`:

1. **Document Loading**:
   * Opens the file at `state["file_path"]` using `fitz.open()`.
2. **Page Ingestion Loop**:
   * Iterates slide-by-slide or page-by-page. For each page:
3. **Table Extraction (`page.find_tables()`)**:
   * Runs PyMuPDF's native table finder.
   * Extracts cell matrices, maps them into Markdown table formats, and builds structured `headers` and `rows` grids inside `table_data` for citation displays.
   * Records table bounding boxes (`bbox`) in page coordinates to filter them from the subsequent text parsing pass.
4. **Text & Heading Extraction**:
   * Invokes `page.get_text("dict")` to extract layout blocks, lines, and spans.
   * Computes a histogram of font sizes to find the median body text font size.
   * Runs `page.get_text("blocks")` to extract text blocks.
   * **Heading Detection**: If a text block is short (under 120 characters), has no newlines, and its font size is larger than the median body text size, the tool flags it as a `heading`; otherwise, it registers it as a `text` block.
   * **Table Overlap Filtering**: Prevents duplicate text indexing by checking if text block coordinates intersect with active table bounding boxes.
5. **Output Generation**:
   * Builds `NormalizedBlock` schemas and appends them to the pipeline block list.
