# Excel Spreadsheet Extraction Module

The **Excel Extraction Module** (`backend/extraction/excel/`) parses spreadsheet workbooks (`.xlsx`, `.xls`, `.xlsm`) into structured tabular blocks and markdown data representations.

---

## 1. Key Capabilities & Features

- **Dual-Engine Robustness** ([`excel_extract.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/excel/excel_extract.py)):
  - Primary parsing via `pandas` with `openpyxl` / `xlrd`.
  - Automatic fallback to cell-by-cell iteration (`_fallback_sheet_text`) when corrupted formatting causes dataframe parse failures.
- **Bilingual Header Normalization**:
  - Automatically identifies and normalizes multi-language column headers to preserve consistent indexing across mixed-language sheets.
- **Embedded Diagram & Chart Extraction**:
  - Scans worksheets for embedded raster/vector images, sniffs image format magic bytes, and persists visual artifacts to `uploads/images/` for multimodal enrichment.
- **Spreadsheet Guardrails**:
  - Removes empty rows/columns, replaces NaN artifacts, and enforces grid coordinate cell references (e.g. `Sheet1!A1:D40`) in block metadata.
- **Agent Sandbox Analytics (`excel_tool`)**:
  - Supports sandboxed Python/Pandas data analytics execution for dynamic question answering.

---

## 2. Dependencies & Integrations

- **pandas**: Dataframe manipulation and table markdown export.
- **openpyxl**: Modern `.xlsx` and `.xlsm` XML parsing engine.
- **xlrd**: Legacy `.xls` binary workbook parsing engine.
- **Pillow (`PIL`)**: Embedded image extraction and byte sniffing.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    Workbook[Excel File .xlsx / .xls] --> Parser{Load Sheets via Pandas}
    Parser -->|Success| DF[DataFrame Representation]
    Parser -->|Failure| CellScan[Openpyxl Cell-by-Cell Fallback]

    DF & CellScan --> Clean[Drop Empty Rows/Cols & Normalize Headers]
    Clean --> TableBlock[Generate NormalizedBlock: Markdown Table + JSON]
    
    Workbook --> ImageScan[Extract Embedded Drawings & Charts via PIL]
    ImageScan --> ImageBlock[NormalizedBlock: pending_vision Image]

    TableBlock & ImageBlock --> Out[state['blocks']]
```

---

## 4. Verification & Testing

```powershell
# Test Excel and PPT extraction
pytest tests/test_excel_ppt_tools.py
```
