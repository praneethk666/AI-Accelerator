# Document Categorization & Route Classification Module

The **Categorization Module** (`backend/categorize/`) inspects incoming documents at the start of ingestion to determine their document type, industry domain, execution route, and PDF physical structure (`digital`, `scanned`, `mixed`).

---

## 1. Key Capabilities & Features

- **Multimodal Visual Inspection** ([`classifier.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/categorize/classifier.py)):
  - *PDFs & Images*: Renders cover pages at 150 DPI and passes them to multimodal vision models (Gemini Flash, OpenAI GPT-4o, NVIDIA Llama 3.2 Vision) for visual classification.
  - *Office Files* (`.docx`, `.pptx`, `.xlsx`): Reads initial text/TOC segments and classifies via structured LLM prompt.
- **Evidence-Grounded Verification (`_evidence_supported`)**:
  - Requires the model to return an exact `"evidence"` snippet alongside its classification. Verifies that the evidence string exists within the document's actual text layer to eliminate hallucinations.
- **Dynamic Ingestion Route Mapping**:
  - Maps `document_type` to pipeline routes (`text_default`, `diagram_heavy`, `cad_route`, `circuit_route`, `image_route`) via `config.type_to_route`.
- **PDF Kind Classification** ([`detector.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/categorize/detector.py)):
  - Scans vector text operators vs raster bitmaps across pages to label PDFs as `digital`, `scanned`, or `mixed` in `state["pdf_kind"]`.

---

## 2. Dependencies & Integrations

- **fitz (PyMuPDF)**: First-page and TOC rendering.
- **backend.core.vision_client & llm_client**: Vision and text model completions.
- **langdetect**: Language identification.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    Doc[Ingested File] --> TypeSniff{Sniff File Extension}
    
    TypeSniff -->|PDF / Image| RenderFirst[Render First Page at 150 DPI via PyMuPDF]
    TypeSniff -->|Word / Excel / PPT| ExtractText[Extract Initial 5k Characters / TOC]

    RenderFirst --> VLM[Multimodal VLM Classification]
    ExtractText --> LLM[Text LLM Classification]

    VLM & LLM --> ExtractJSON[Extract: doc_type, industry, confidence, evidence]
    ExtractJSON --> GroundCheck{Evidence Exists in Document Text?}
    
    GroundCheck -->|Yes| MapRoute[Map Route via config.type_to_route]
    GroundCheck -->|No| SafeFallback[Fallback: text_default & default_industry]

    TypeSniff -->|PDF File| PDFDetect[detector.py: Count Text vs Bitmap Pages]
    PDFDetect --> SetKind[state['pdf_kind'] = digital / scanned / mixed]

    MapRoute & SafeFallback & SetKind --> StateUpdate[Write state: route, document_type, industry, confidence]
```

---

## 4. Configuration & Testing

### Configuration Blueprint (`config/global.yaml`)
```yaml
categorization:
  confidence_thresholds:
    categorization_low_confidence: 0.5
  industry_keywords:
    automotive: [toyota, ford, engine, transmission, torque]
    electronics: [circuit, pcb, schematic, resistor, voltage]
    manufacturing: [assembly, drawing, tolerance, weld, bom]

type_to_route:
  cad_drawing: cad_route
  circuit_diagram: circuit_route
  schematic: diagram_heavy
  report: text_default
```

### Verification & Unit Tests
```powershell
# Run categorization handler tests
pytest tests/test_category_handler.py
```
