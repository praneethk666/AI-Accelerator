# IBM Docling PDF Extraction & Table Recovery Module

The **Docling PDF Module** (`backend/extraction/docling_pdf/`) is the unified PDF extraction engine for AI-Accelerator. It handles digital, scanned, and mixed PDFs through an intelligent per-page routing architecture combining IBM Docling layout parsing, TableFormer table structure recovery, and VLM OCR rescue.

---

## 1. Key Capabilities & Features

- **Hybrid Page-by-Page Routing** ([`tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/tool.py), [`docling_extract.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/docling_extract.py)):
  - *Digital Pages*: Fast native layout analysis with Docling (running with OCR disabled for speed).
  - *Scanned / Degraded Pages*: Routed to VLM OCR rescue (`vision_ocr.py`) to transcribe degraded scans, complex schematics, and mixed layouts with zero character corruption.
- **TableFormer Table Recovery**:
  - Automatically reconstructs structured rows, headers, and merged cells from complex PDF tables into clean JSON matrices (`table_data`) and Markdown strings.
- **Multi-Page Table Stitching**:
  - Reconciles and stitches tables spanning page boundaries back into cohesive unified tabular datasets.
- **64-Step Procedure Extraction**:
  - Extracts end-to-end sequential step-by-step operating procedures, safety guidelines, and maintenance protocols, tracking exact page and bounding box citations for every action.
- **Inline Figure Captioning**:
  - Cropped diagrams and figure regions are saved to disk (`uploads/images/`) and registered as `image_caption` blocks, bypassing redundant downstream vision processing.

---

## 2. Dependencies & Integrations

- **docling / docling-core**: IBM Docling document parsing engine and TableFormer.
- **fitz (PyMuPDF)**: Fast bitmap rendering and vector geometry analysis.
- **backend.extraction.vision_ocr**: Multimodal VLM OCR rescue.
- **backend.extraction.table_reconcile**: Multi-page table stitching algorithms.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    PDF[Input PDF Document] --> Classify[Categorize: Detect Digital / Scanned / Mixed]
    Classify --> DoclingTool[DoclingPDFTool]

    DoclingTool --> PageRouter{Per-Page Inspection}
    PageRouter -->|Clean Digital Text| DoclingEngine[Docling Layout AI + TableFormer]
    PageRouter -->|Scanned / Complex| VLMRescue[VLM OCR Rescue via Gemini / NVIDIA]

    DoclingEngine --> TableCheck{Table Spans Pages?}
    TableCheck -->|Yes| Stitcher[Table Reconcile Engine]
    TableCheck -->|No| BlockGen[NormalizedBlock Builder]
    Stitcher --> BlockGen
    VLMRescue --> BlockGen

    BlockGen --> CacheBlocks[(Write Blocks to Postgres)]
    CacheBlocks --> Out[state['blocks']]
```

---

## 4. Configuration & Testing

### Configuration Blueprint (`config/global.yaml`)
```yaml
extraction:
  deskew: true
  stitch_tables: true
  docling:
    do_ocr: false
    do_table_structure: true
    images_scale: 2.0
    page_rescue: true
    max_vlm_pages: 60
    mode: local                       # local | remote server
```

### Verification & Unit Tests
```powershell
# Test Docling and scanned PDF extraction flows
pytest tests/test_docling_scanned.py
```
