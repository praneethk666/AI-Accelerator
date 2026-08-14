# PyMuPDF PDF Utilities & Fallback Module

The **PyMuPDF PDF Module** (`backend/extraction/pymupdf_pdf/`) provides high-speed native text parsing and rendering utilities using **PyMuPDF (`fitz`)**.

---

## 1. Overview & Operational Role

- **Primary Role**: Serves as a high-speed lightweight fallback extractor and page rendering utility for the categorization and CAD subsystems.
- **Key Utilities**:
  - `page_to_image()`: Renders PDF pages to high-resolution PNG buffers for multimodal vision inspection.
  - `extract_native_text()`: Fast zero-dependency text layer extraction.
- **Production Routing**: Primary PDF processing in production is coordinated by [`backend/extraction/docling_pdf/`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/extraction/docling_pdf/) for advanced layout AI and TableFormer recovery.
