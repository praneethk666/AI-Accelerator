# Scanned PDF OCR & Layout Utilities Module

The **Scanned PDF Module** (`backend/extraction/scanned_pdf/`) houses low-level OCR and layout-region detection utilities consumed by `docling_pdf` and `vision_ocr`.

---

## 1. Overview & Components

- **Layout Recognition**: Wraps layout detection models (Surya, PaddleOCR) for bounding box coordinate detection.
- **Orientation Correction**: Detects and corrects sideways/upside-down scanned pages prior to OCR processing.
- **Unified Production Workflow**: Scanned PDF extraction is executed directly by [`backend/extraction/docling_pdf/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/) via hybrid VLM rescue and TableFormer.
