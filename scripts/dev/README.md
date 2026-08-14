# Developer Research & Benchmarking Tools

The **Dev Scripts Module** (`scripts/dev/`) contains research tools, model bake-off benchmarks, and layout comparison utilities.

---

## 1. Benchmarking Utilities

| Script | Purpose |
|---|---|
| `ocr_bakeoff.py` | Benchmarks PaddleOCR, Surya, and Multimodal VLMs against ground-truth texts to evaluate Character Error Rates (CER). |
| `ocr_compare.py` / `ocr_diff.py` | Generates visual diffs and accuracy reports comparing OCR engine outputs. |
| `table_compare.py` | Compares IBM Docling TableFormer against multimodal VLMs for structured table recovery accuracy. |
| `make_scanned.py` | Converts clean digital PDFs into synthetic scanned bitmaps with simulated skew and noise to test OCR resilience. |
| `vlm_ocr_test.py` | Evaluates prompt variations across multimodal vision models for technical drawing transcription. |
