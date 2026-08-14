# Document Extraction Subsystem

The **Extraction Subsystem** (`backend/extraction/`) converts heterogeneous enterprise file formats (PDFs, Excel spreadsheets, PowerPoint decks, Word documents, standalone images, and CAD/circuit drawings) into normalized document blocks (`NormalizedBlock[]`) complying with the unified `Tool.run(state, config)` contract.

---

## 1. Extractor Registry & Capability Matrix

| File Type | Primary Tool | Engine / Library | Key Capabilities |
|---|---|---|---|
| **PDF** (Digital, Scanned, Mixed) | `DoclingPDFTool` ([`docling_pdf/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/)) | IBM Docling + TableFormer + VLM | Hybrid page-by-page routing, layout parsing, TableFormer table structure recovery, VLM OCR rescue for degraded pages, and 64-step procedure extraction. |
| **CAD / Circuit** | `CADExtractionTool` ([`cad/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/cad/)) | PyMuPDF + Google Gemini / NVIDIA Vision | High-DPI large-format tiling (ANSI-D/E sheets), title block extraction, BOM parsing, schematic net names, and reference designator extraction. |
| **Excel** (`.xlsx`, `.xls`, `.xlsm`) | `ExcelExtractorTool` ([`excel/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/excel/)) | pandas + openpyxl + xlrd | Bilingual header normalization, multi-sheet traversal, embedded chart/diagram extraction, and markdown table serialization. |
| **PowerPoint** (`.pptx`, `.ppt`) | `PPTExtractorTool` ([`ppt/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/ppt/)) | python-pptx | Slide layout analysis, speaker notes extraction, nested shape hierarchy traversal, and embedded slide image cropping. |
| **Word** (`.docx`, `.doc`) | `WordExtractorTool` ([`word/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/word/)) | python-docx | Paragraph XML parsing, heading hierarchy recovery, table parsing, embedded image extraction, and docx-html preview generation. |
| **Images** (`.png`, `.jpg`, `.tif`) | `ImageExtractorTool` ([`image/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/image/)) | Pillow (`PIL`) | Magic-byte format sniffing, visual captioning queueing, and fallback OCR. |

---

## 2. Shared Extraction Contract

All extractors conform to the `Tool` interface:
```python
class Tool(Protocol):
    name: str
    def run(self, state: PipelineState, config: dict) -> PipelineState: ...
```

### Data Contract (`NormalizedBlock`)
Extractors output a list of `NormalizedBlock` dataclasses containing:
- `block_id`: Unique identifier (`{doc_id}_{page}_{idx}`).
- `document_id`: Target document UUID.
- `type`: Block category (`text`, `heading`, `table`, `image`, `image_caption`).
- `text`: Extracted plain text or markdown table rendering.
- `table_data`: Structured `{ "headers": [...], "rows": [[...]] }` (for `table` blocks).
- `source_ref`: Anchor object (`filename`, `page`, `sheet`, `slide`, `bbox`).
- `metadata`: Format-specific metadata (e.g. `raw_image_path`, `cell_range`).

---

## 3. Architecture & Extraction Flow

```mermaid
graph TD
    File[Input Document File] --> Cat[Categorize Step]
    Cat --> Dispatch{Graph Extractor Dispatch}

    Dispatch -->|PDF Route| Docling[DoclingPDFTool]
    Dispatch -->|CAD / Circuit Route| CAD[CADExtractionTool]
    Dispatch -->|Excel File| Excel[ExcelExtractorTool]
    Dispatch -->|PPT File| PPT[PPTExtractorTool]
    Dispatch -->|Word File| Word[WordExtractorTool]
    Dispatch -->|Image File| Image[ImageExtractorTool]

    Docling --> BlockCache[(Immediate Block Cache to Postgres)]
    CAD --> BlockCache
    Excel --> BlockCache
    PPT --> BlockCache
    Word --> BlockCache
    Image --> BlockCache

    BlockCache --> StateOut[state['blocks'] = NormalizedBlock List]
```

---

## 4. Submodule Navigation

- [`cad/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/cad/README.md): Mechanical CAD & Circuit schematic vision extraction.
- [`docling_pdf/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/README.md): IBM Docling hybrid PDF parser & table recovery.
- [`excel/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/excel/README.md): Excel spreadsheet extractor with bilingual headers.
- [`image/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/image/README.md): Standalone image extractor.
- [`ppt/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/ppt/README.md): PowerPoint presentation deck extractor.
- [`word/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/word/README.md): Word document XML extractor.
- [`pymupdf_pdf/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/pymupdf_pdf/README.md) & [`scanned_pdf/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/scanned_pdf/README.md): Fast text detectors and layout utility modules.

---

## 5. Verification & Testing

```powershell
# Test PDF, Excel, PPT, Word, and Image extractors
pytest tests/test_docling_scanned.py tests/test_excel_ppt_tools.py tests/test_word_tool.py tests/test_image_tool.py
```
