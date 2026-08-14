# PowerPoint Presentation Extraction Module

The **PowerPoint Extraction Module** (`backend/extraction/ppt/`) parses presentation decks (`.pptx`, `.ppt`) into structured slide elements, tables, speaker notes, and embedded graphics.

---

## 1. Key Capabilities & Features

- **Slide Structure Analysis**: Traverses slide shape hierarchies to extract slide titles, bullet points, and text boxes with spatial order preservation.
- **Table Reconstruction**: Extracts slide tables into structured Markdown and JSON representations.
- **Speaker Notes Parsing**: Captures hidden slide notes and associates them with slide metadata.
- **Embedded Graphic Cropping**: Extracts embedded slide diagrams, saving image files to `uploads/images/` for vision enrichment.

---

## 2. Dependencies & Testing

- **python-pptx**: PowerPoint XML document model parser.
- **Verification**:
  ```powershell
  pytest tests/test_excel_ppt_tools.py
  ```
