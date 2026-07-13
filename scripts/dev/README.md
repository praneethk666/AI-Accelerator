# Dev / R&D scripts

One-off tools used to make engineering decisions (which OCR engine, which table
extraction approach) — not part of the product pipeline. Each is self-contained;
run with `--help` or read its docstring for usage. Kept for re-running a bake-off
if a new OCR engine or model shows up; not needed to run the pipeline itself.

- `make_scanned.py` — rasterize a digital PDF into a scanned (image-only) one, for OCR testing.
- `ocr_bakeoff.py` — score PaddleOCR / Docling(RapidOCR) / a VLM against a digital "ground truth" twin.
- `ocr_compare.py` — run one OCR engine over a scanned PDF, emit per-page metrics.
- `ocr_diff.py` — compare scanned OCR output against the digital twin's native text.
- `table_compare.py` — Docling TableFormer vs a VLM, head-to-head on the same table.
- `vlm_ocr_test.py` — test a vision model as page-OCR against the native text layer.
