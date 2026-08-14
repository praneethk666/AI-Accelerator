# Word Document Extraction Module

The **Word Extraction Module** (`backend/extraction/word/`) parses Microsoft Word documents (`.docx`, `.doc`) into structured text paragraphs, heading hierarchies, tables, and embedded images.

---

## 1. Key Capabilities & Features

- **Document Hierarchy Recovery**: Analyzes heading styles (`Heading 1`, `Heading 2`, `Heading 3`) to structure document sections.
- **Table Parsing**: Reconstructs XML table rows and cells into structured JSON matrices and Markdown representations.
- **Embedded Graphic Extraction**: Identifies and extracts embedded images to `uploads/images/`.
- **HTML Preview Endpoint Support**: Integrates with `/files/{id}/docx-html` for direct, high-fidelity browser rendering in the frontend.

---

## 2. Dependencies & Testing

- **python-docx**: Word OpenXML document parsing library.
- **Verification**:
  ```powershell
  pytest tests/test_word_tool.py
  ```
