"""Excel / PPT extractor tests.

Blocks flow through state as PLAIN DICTS (see schemas.as_dicts) — assert with
dict access, not attribute access.
"""
from backend.extraction.excel.tool import ExcelExtractorTool
from backend.extraction.ppt.tool import PPTExtractorTool

EXCEL_FILE = "test-data/test.xlsx"
PPT_FILE   = "test-data/test.pptx"


# ── Excel ──────────────────────────────────────────────────────────────────────

def test_excel_blocks_nonempty():
    state = ExcelExtractorTool().run({"file_path": EXCEL_FILE, "document_id": "test-doc-001"}, {})
    assert state["blocks"], "No blocks extracted from Excel file"
    assert all(isinstance(b, dict) for b in state["blocks"]), "blocks must be plain dicts"


def test_excel_errors_key_exists():
    state = ExcelExtractorTool().run({"file_path": EXCEL_FILE, "document_id": "test-doc-001"}, {})
    assert "errors" in state


def test_excel_table_blocks():
    state  = ExcelExtractorTool().run({"file_path": EXCEL_FILE, "document_id": "test-doc-001"}, {})
    tables = [b for b in state["blocks"]
              if b["type"] == "table" and "pivot_name" not in (b.get("metadata") or {})]
    assert tables, "No table blocks found"
    for t in tables:
        assert t["text"], "Table block missing markdown text"
        assert t["table_data"], "Table block missing table_data"
        assert "headers" in t["table_data"]
        assert "rows" in t["table_data"]


def test_excel_image_blocks():
    state  = ExcelExtractorTool().run({"file_path": EXCEL_FILE, "document_id": "test-doc-001"}, {})
    images = [b for b in state["blocks"] if b["type"] == "image_caption"]
    assert images, "No image blocks found"
    for img in images:
        meta = img.get("metadata") or {}
        assert meta.get("raw_image_path"), "Missing raw_image_path"
        assert meta.get("pending_vision") is True, "Missing pending_vision flag"


def test_excel_formula_blocks():
    state    = ExcelExtractorTool().run({"file_path": EXCEL_FILE, "document_id": "test-doc-001"}, {})
    formulas = [b for b in state["blocks"]
                if b["type"] == "text" and "formula" in (b.get("metadata") or {})]
    assert formulas, "No formula blocks found"
    for f in formulas:
        assert f["text"], "Formula block missing text"
        assert (f.get("metadata") or {}).get("cell_ref"), "Formula block missing cell_ref"


# ── PPT ───────────────────────────────────────────────────────────────────────

def test_ppt_blocks_nonempty():
    state = PPTExtractorTool().run({"file_path": PPT_FILE, "document_id": "test-doc-002"}, {})
    assert state["blocks"], "No blocks extracted from PPT file"
    assert all(isinstance(b, dict) for b in state["blocks"]), "blocks must be plain dicts"


def test_ppt_errors_key_exists():
    state = PPTExtractorTool().run({"file_path": PPT_FILE, "document_id": "test-doc-002"}, {})
    assert "errors" in state


def test_ppt_text_blocks():
    state = PPTExtractorTool().run({"file_path": PPT_FILE, "document_id": "test-doc-002"}, {})
    texts = [b for b in state["blocks"] if b["type"] == "text"]
    assert texts, "No text blocks found"
    for t in texts:
        assert t["text"], "Text block has no text"


def test_ppt_table_blocks():
    state  = PPTExtractorTool().run({"file_path": PPT_FILE, "document_id": "test-doc-002"}, {})
    tables = [b for b in state["blocks"] if b["type"] == "table"]
    assert tables, "No table blocks found"
    for t in tables:
        assert t["text"], "Table block missing markdown text"
        assert t["table_data"], "Table block missing table_data"
        assert "headers" in t["table_data"]
        assert "rows" in t["table_data"]


def test_ppt_image_blocks():
    state  = PPTExtractorTool().run({"file_path": PPT_FILE, "document_id": "test-doc-002"}, {})
    images = [b for b in state["blocks"] if b["type"] == "image_caption"]
    assert images, "No image blocks found"
    for img in images:
        meta = img.get("metadata") or {}
        assert meta.get("raw_image_path"), "Missing raw_image_path"
        assert meta.get("pending_vision") is True, "Missing pending_vision flag"
