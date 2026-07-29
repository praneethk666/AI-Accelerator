# Excel Extraction Module

The Excel Extraction module parses spreadsheets (`.xlsx`, `.xls`, `.xlsm`) and converts sheets into structured blocks.

## Dependencies

* **pandas**: Parses data sheets and structures tables.
* **openpyxl**: Parser engine for modern `.xlsx` and `.xlsm` files.
* **xlrd**: Parser engine for legacy `.xls` files.
* **Pillow (`PIL`)**: Identifies, extracts, and saves embedded diagrams and charts.
* **langdetect**: Detects text language.

## Step-by-Step Logic

The extraction process runs in `ExcelExtractorTool::_extract()`:

1. **Workbook Loading**:
   * Attempts to load the workbook using `pd.read_excel()` with the appropriate engine (`openpyxl` or `xlrd`).
   * If pandas fails, the tool falls back to a manual row-by-row cell parse via openpyxl (`_fallback_sheet_text`).
2. **Table Processing**:
   * Loops through sheets, drops completely empty columns/rows, and replaces NaN values with empty strings.
   * Generates a markdown rendering of the table data (`df.to_markdown()`).
   * Maps sheet names and grid coordinate cell ranges (e.g. `Sheet1!A1:D40`) to `source_ref`.
3. **Embedded Image Extraction**:
   * Scans openpyxl worksheet image listings.
   * Sniffs image formats using magic bytes (`_detect_image_ext`) to prevent file corruption.
   * Persists image files to `uploads/images/{doc_id}/{block_id}.png`.
   * Appends a `pending_vision` block containing the image path to the output block list.
4. **Normalized Block Construction**:
   * Wraps parsed content in `NormalizedBlock` objects and registers them in the pipeline state.
