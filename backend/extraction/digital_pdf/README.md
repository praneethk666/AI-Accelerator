# Digital PDF Extractor

**Owner:** Manoj  
**Part of:** AI-Accelerator document intelligence pipeline.

Extracts text, tables, images, and vector drawings from **digital PDFs** (PDFs that contain a native text layer). Uses PyMuPDF (fitz) – no OCR. Outputs `NormalizedBlock` objects that conform to the shared schema (`backend/core/schemas.py`).

---

## What this module does

- **Text** – extracts and merges spans into paragraphs; detects headings (font size ≥14 or bold ≥12).
- **Tables** – extracts structured tables using PyMuPDF's `find_tables()`; outputs `table_data` with headers and rows.
- **Images** – extracts significant images (larger than `MIN_IMAGE_AREA`), creates `image_caption` placeholders (text = `"[Image - awaiting vision enrichment]"`).
- **Vector drawings** – extracts vector graphics, filtered by area, complexity, and table‑overlap avoidance.

> **Important:** Page‑level metadata (text length, image count, vector count, table hint) is **not** included in the output blocks. That data is generated separately by `backend/extraction/page_profile.py` (shared across digital, scanned, and mixed PDF tools). This separation follows the reviewer's requirement: *"Remove the page_metrics blocks from digital.py, that data stays in page_profiles"*.

---

## Files

| File | Description |
|------|-------------|
| `digital.py` | Core extraction logic: `extract_digital()` function. |
| `tool.py` | Wraps `extract_digital` as a `Tool` for use in the LangGraph pipeline. |
| `README.md` | This file. |

---

## Usage

### Standalone (without the pipeline)

```python
from backend.extraction.digital_pdf.digital import extract_digital

blocks = extract_digital("path/to/document.pdf", document_id="my_doc")
for block in blocks:
    print(f"{block.type}: {block.text[:50] if block.text else 'table'}")
```

### As a Tool (inside the LangGraph pipeline)

```python
from backend.extraction.digital_pdf.tool import PDFDigitalTool

tool = PDFDigitalTool()
state = {
    "file_path": "path/to/document.pdf",
    "document_id": "my_doc"
}
new_state = tool.run(state, {})
blocks = new_state["blocks"]          # list of NormalizedBlock
profiles = new_state["page_profiles"] # list of PageProfile (metadata)
```

---

## Configuration

The following constants are currently hardcoded in `digital.py`. They will be moved to a central configuration file in the future.

| Constant | Default | Explanation |
|----------|---------|-------------|
| `MIN_IMAGE_AREA` | 1000 px² | Ignore images smaller than ~32×32 pixels (e.g., logos). |
| `MIN_DRAWING_AREA_RATIO` | 0.005 | Minimum drawing area relative to page (0.5% of page area). |
| `MIN_VECTOR_COMPLEXITY` | 5 | Minimum number of drawing items; skip simple vectors (e.g., single rectangle). |
| `LINE_GAP_FACTOR` | 1.5 | Multiplier for vertical line gap to merge lines into paragraphs. |
| `TABLE_OVERLAP_THRESHOLD` | 0.5 | Skip vector drawing if more than 50% of its area overlaps a table. |

---

## Output Schema (`NormalizedBlock`)

See `backend/core/schemas.py` for the exact definition. Example:

```json
{
  "block_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "my_doc",
  "type": "text",
  "text": "The extracted paragraph text...",
  "table_data": null,
  "source_ref": {
    "filename": "document.pdf",
    "page": 3,
    "sheet": null,
    "slide": null,
    "bbox": [x0, y0, x1, y1]
  },
  "confidence": 1.0,
  "language": "en",
  "metadata": {}
}
```

- For `type` = `"table"`, `text` is `null` and `table_data` contains `{"headers": [...], "rows": [...]}`.  
- For `type` = `"image_caption"`, `text` is a placeholder and `metadata` includes `pending_vision: true`.

---

## Page Profiles

Page metadata (text length, image regions, vector count, table hint) is produced by `backend/extraction/page_profile.py`. That module is **shared** among digital, scanned, and mixed PDF tools.

---

## Dependencies

- `PyMuPDF` (fitz) – PDF parsing.
- `backend.core.schemas` – `NormalizedBlock`, `SourceRef`.
- `backend.utils.save_json` – optional JSON serialization (used for debugging).

---

## Testing

The digital extractor is covered by the integration test suite. Run from project root:

```bash
python -m pytest tests/test_extraction_tools.py -v
```

The test automatically routes a sample digital PDF to the `PDFDigitalTool` and verifies that blocks and page profiles are produced.

---

## Related Modules

- `backend/extraction/scanned_pdf/` – OCR-based extraction for scanned PDFs.
- `backend/extraction/mixed_pdf/` – dispatches per page to digital or scanned pipeline.
- `backend/extraction/page_profile.py` – generates per‑page metadata (shared).

---

*Part of the AI-Accelerator – a config‑driven document intelligence + RAG pipeline.*