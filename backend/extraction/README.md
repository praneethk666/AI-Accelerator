# Extraction — PDF

**Owner:** Manoj  
**Part of:** AI-Accelerator document intelligence pipeline.

This folder contains the **PDF extraction** modules. They transform a PDF file into a structured list of `NormalizedBlock` objects (text, tables, images, vector drawings) and generate per-page metadata (`PageProfile`). The extraction is split into three independent but cooperating sub-modules:

- `digital_pdf/` – extracts content from **digital PDFs** (native text layer) using PyMuPDF.
- `scanned_pdf/` – handles **scanned PDFs** (image-only pages) using PaddleOCR + YOLO + contour detection.
- `mixed_pdf/` – dispatches each page to the appropriate pipeline (digital or scanned) and merges the results.

Two shared utility files live directly in this folder:
- `detector.py` – classifies a PDF as digital, scanned, or mixed (per-page and overall).
- `page_profile.py` – generates per-page metadata (text length, image regions, vector count, table hint) without altering the block stream.

> **Important:** Page-level metadata is **not** included in the block list. It is produced separately by `page_profile.py` to keep the extraction clean and follow the shared contract.

---

## Modules (subfolders)

| Subfolder | Description |
|-----------|-------------|
| `digital_pdf/` | Extracts text, tables, images, vector drawings from digital PDFs. Uses `pymupdf`. |
| `scanned_pdf/` | OCR + vision extraction for scanned PDFs. Uses `paddleocr`, `ultralytics`, `opencv`. |
| `mixed_pdf/` | Routes each page to digital or scanned pipeline, merges blocks, cleans up per-page JSONs. |

Each subfolder contains its own `README.md` with detailed usage, configuration, and implementation notes.

---

## Shared Utilities

### `detector.py`

```python
def detect_pdf_type(file_path: str) -> Tuple[str, List[str]]
```

- **Input:** path to a PDF file.
- **Output:** `(overall_type, per_page_types)` where `overall_type` is `"digital"`, `"scanned"`, or `"mixed"`, and `per_page_types` is a list of `"digital"`/`"scanned"` for each page (based on whether `page.get_text().strip()` length > 5).
- **Purpose:** Used by `mixed_pdf` to decide which pipeline to call for each page. Also used by the test suite to automatically select the right tool.

### `page_profile.py`

```python
def page_profile(pdf_path: str) -> List[PageProfile]
```

- **Input:** path to a PDF file.
- **Output:** a list of `PageProfile` objects (one per page) as defined in `backend/core/schemas.py`.
- **Purpose:** Generates a per-page "x-ray" – metadata that tells the pipeline how to handle each page (digital/scanned/mixed, image regions, vector presence, table hint). This is the only place where such metadata is stored; no `page_metrics` blocks appear in the extraction output.

---

## Usage Overview

### 1. Extract a digital PDF directly

```python
from backend.extraction.digital_pdf.digital import extract_digital
blocks = extract_digital("document.pdf", document_id="my_doc")
```

### 2. Extract a scanned PDF directly

```python
from backend.extraction.scanned_pdf.scanned import extract_scanned
blocks = extract_scanned("scanned.pdf", document_id="my_doc")
```

### 3. Extract a mixed PDF (automatically routes per page)

```python
from backend.extraction.mixed_pdf.mixed import extract_mixed
blocks = extract_mixed("mixed.pdf", document_id="my_doc")
```

### 4. Get page metadata

```python
from backend.extraction.page_profile import page_profile
profiles = page_profile("any.pdf")
```

### 5. Use as Tools in the LangGraph pipeline

Each subfolder exports a `Tool` class (`PDFDigitalTool`, `ScannedPDFTool`, `MixedPDFTool`) that reads `state["file_path"]` and writes `state["blocks"]` and `state["page_profiles"]`. Example:

```python
from backend.extraction.mixed_pdf.tool import MixedPDFTool
tool = MixedPDFTool()
state = {"file_path": "mixed.pdf", "document_id": "my_doc"}
new_state = tool.run(state, {})
```

---

## Testing

Run the integrated test for all three PDF types:

```bash
python -m pytest tests/test_extraction_tools.py -v
```

The test automatically classifies the PDF, selects the correct tool, and verifies that `blocks` and `page_profiles` are produced.

---

## Dependencies

- `PyMuPDF` – all PDF operations.
- `PaddleOCR` + `paddlepaddle` – OCR for scanned pages.
- `ultralytics` (YOLO) – visual region detection (optional, falls back to contours).
- `opencv-python` – contour detection and image processing.
- `Pillow`, `numpy` – image handling.

---

## Related Documentation

- `backend/core/schemas.py` – defines `NormalizedBlock`, `PageProfile`, `ImageRegion`.
- `digital_pdf/README.md` – detailed docs for the digital extractor.
- `scanned_pdf/README.md` – detailed docs for the scanned extractor.
- `mixed_pdf/README.md` – detailed docs for the mixed dispatcher.

---

*Part of the AI-Accelerator – a config-driven document intelligence + RAG pipeline.*