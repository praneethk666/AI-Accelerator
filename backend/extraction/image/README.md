# Standalone Image Extraction Module

The **Image Extraction Module** (`backend/extraction/image/`) ingests standalone images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`) and registers them for multimodal VLM captioning and OCR analysis.

---

## 1. Key Capabilities & Features

- **Format & Metadata Sniffing**: Validates image dimensions, formats, and orientation using Pillow (`PIL`).
- **Pipeline Staging**: Copies raw image bytes to `uploads/images/{doc_id}/` and generates a single `NormalizedBlock` with `type="image"` and `metadata["pending_vision"]=True`.
- **Downstream Hand-Off**: Directly consumed by `VisionEnrichmentTool` for automated visual captioning, diagram interpretation, and OCR text recovery.

---

## 2. Dependencies & Testing

- **Pillow (`PIL`)**: Image inspection and byte stream handling.
- **Verification**:
  ```powershell
  pytest tests/test_image_tool.py
  ```
