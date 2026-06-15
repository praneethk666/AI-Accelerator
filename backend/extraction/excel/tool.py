import os
import io
import uuid
import pandas as pd
from typing import List, Dict, Any, Optional
from langdetect import detect, LangDetectException
from backend.core.schemas import NormalizedBlock, SourceRef
from backend.core.tool import Tool


def _detect_image_ext(data: bytes) -> str:
    if not data: return "png"
    if data[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if data[:3] == b"\xff\xd8\xff": return "jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"): return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return "webp"
    if data[:4] in (b"MM\x00*", b"II*\x00"): return "tiff"
    if data[:2] == b"BM": return "bmp"
    if data[:4] == b"\xd7\xcd\xc6\x9a": return "wmf"
    if data[:4] == b"\x01\x00\x00\x00" or (len(data) > 44 and data[40:44] == b" EMF"): return "emf"
    if b"<svg" in data[:100].lower(): return "svg"
    return "png"


class ExcelExtractorTool(Tool):
    name = "excel_extraction"

    def run(self, state: dict, config: dict) -> dict:
        file_path   = state.get("file_path")
        document_id = state.get("document_id")
        filename    = state.get("filename", os.path.basename(file_path) if file_path else "unknown.xlsx")

        if not document_id or not file_path:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  "Missing document_id or file_path in state — aborting.",
                "block_id": None,
            })
            return state

        blocks = self._extract(file_path, filename, str(document_id), config, state)

        if "blocks" not in state:
            state["blocks"] = blocks
        else:
            state["blocks"].extend(blocks)

        state.setdefault("errors", [])
        return state

    def _detect_language(self, text: str) -> str:
        try:
            return detect(text) if text and text.strip() else "en"
        except LangDetectException:
            return "en"

    def _extract(
        self,
        file_path: str,
        filename: str,
        doc_id: str,
        config: Optional[Dict[str, Any]],
        state: dict,
    ) -> List[NormalizedBlock]:
        import openpyxl
        blocks = []
        cfg    = config or {}

        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".xls":
            engine = "xlrd"
        elif ext == ".xlsb":
            engine = "pyxlsb"
        else:
            engine = "openpyxl"

        try:
            wb_pandas   = pd.read_excel(file_path, sheet_name=None, engine=engine)
            wb_openpyxl = openpyxl.load_workbook(file_path, data_only=False) if engine == "openpyxl" else None
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  f"Failed to open {file_path}: {e}",
                "block_id": None,
            })
            return blocks

        for sheet_name, df in wb_pandas.items():
            try:
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                df = df.fillna("")

                block_id   = str(uuid.uuid4())
                cell_range = None
                if wb_openpyxl and sheet_name in wb_openpyxl.sheetnames:
                    cell_range = wb_openpyxl[sheet_name].dimensions

                blocks.append(NormalizedBlock(
                    block_id=block_id,
                    document_id=doc_id,
                    type="table",
                    text=df.to_markdown(index=False),
                    table_data={
                        "headers": [str(c) for c in df.columns.tolist()],
                        "rows":    df.values.tolist(),
                    },
                    source_ref=SourceRef(
                        filename=filename,
                        sheet=str(sheet_name),
                    ),
                    confidence=cfg.get("extraction_confidence", 1.0),
                    language=self._detect_language(df.to_string()),
                    metadata={
                        "cell_range":        cell_range,
                        "enrichment_failed": False,
                    },
                ))

            except Exception as e:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  f"Skipping sheet '{sheet_name}': {e}",
                    "block_id": None,
                })
                continue

        if wb_openpyxl:
            blocks.extend(self._extract_images(wb_openpyxl, doc_id, filename, cfg, state))
            blocks.extend(self._extract_formulas(wb_openpyxl, doc_id, filename, cfg, state))

        blocks.extend(self._extract_pivots(file_path, doc_id, filename, cfg, state))

        return blocks

    def _extract_images(
        self,
        wb,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks  = []
        out_dir = cfg.get("image_output_dir", os.path.join("uploads", "images", doc_id))
        os.makedirs(out_dir, exist_ok=True)

        for sheet in wb.worksheets:
            for image in getattr(sheet, "_images", []):
                try:
                    block_id   = str(uuid.uuid4())
                    image_data = None
                    ext        = "png"

                    pil_img = getattr(image, "image", None) or getattr(image, "_image", None)
                    if pil_img:
                        try:
                            buf = io.BytesIO()
                            fmt = pil_img.format or "PNG"
                            pil_img.save(buf, format=fmt)
                            image_data = buf.getvalue()
                            ext = fmt.lower()
                        except Exception:
                            pil_img = None

                    if not pil_img:
                        if hasattr(image, "_data"):
                            image_data = image._data() if callable(image._data) else image._data
                        else:
                            if hasattr(image.ref, "seek"):
                                image.ref.seek(0)
                            image_data = image.ref.read() if hasattr(image.ref, "read") else bytes(image.ref)

                        ext = getattr(image, "format", "").lower()
                        if not ext or ext not in ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "emf", "wmf", "svg"]:
                            ext = _detect_image_ext(image_data)

                    if not image_data:
                        continue

                    ext      = "jpg" if ext == "jpeg" else ext
                    raw_path = os.path.join(out_dir, f"{block_id}_raw.{ext}")

                    with open(raw_path, "wb") as f:
                        f.write(image_data)

                    blocks.append(NormalizedBlock(
                        block_id=block_id,
                        document_id=doc_id,
                        type="image_caption",
                        text="",
                        source_ref=SourceRef(
                            filename=filename,
                            sheet=sheet.title,
                        ),
                        confidence=cfg.get("extraction_confidence", 1.0),
                        language="en",
                        metadata={
                            "raw_image_path":    raw_path,
                            "pending_vision":    True,
                            "enrichment_failed": False,
                        },
                    ))

                except Exception as e:
                    state.setdefault("errors", []).append({
                        "tool":     self.name,
                        "level":    "error",
                        "message":  f"Skipping image in sheet '{sheet.title}': {e}",
                        "block_id": None,
                    })
                    continue

        return blocks

    def _extract_formulas(
        self,
        wb,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks = []

        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.data_type != "f" or cell.value is None:
                        continue
                    try:
                        formula  = str(cell.value)
                        cell_ref = f"{sheet.title}!{cell.coordinate}"
                        block_id = str(uuid.uuid4())

                        blocks.append(NormalizedBlock(
                            block_id=block_id,
                            document_id=doc_id,
                            type="text",
                            text=formula,
                            source_ref=SourceRef(
                                filename=filename,
                                sheet=sheet.title,
                            ),
                            confidence=cfg.get("extraction_confidence", 1.0),
                            language="en",
                            metadata={
                                "cell_range":        cell_ref,
                                "formula":           formula,
                                "enrichment_failed": False,
                            },
                        ))

                    except Exception as e:
                        state.setdefault("errors", []).append({
                            "tool":     self.name,
                            "level":    "error",
                            "message":  f"Skipping formula at {cell.coordinate}: {e}",
                            "block_id": None,
                        })
                        continue

        return blocks

    def _extract_pivots(
        self,
        file_path: str,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        import openpyxl
        blocks = []

        try:
            wb = openpyxl.load_workbook(file_path, data_only=True)
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  f"Pivot load failed: {e}",
                "block_id": None,
            })
            return blocks

        for sheet in wb.worksheets:
            for pivot in getattr(sheet, "_pivots", []):
                try:
                    cache_def    = getattr(pivot, "cacheDefinition", None)
                    source_range = ""

                    if cache_def:
                        src = getattr(cache_def, "cacheSource", None)
                        if src and hasattr(src, "worksheetSource"):
                            source_range = getattr(src.worksheetSource, "ref", "")

                    headers = []
                    if cache_def and hasattr(cache_def, "cacheFields"):
                        headers = [
                            getattr(cf, "name", f"field_{i}")
                            for i, cf in enumerate(cache_def.cacheFields)
                        ]

                    rows = []
                    if cache_def and hasattr(cache_def, "records") and cache_def.records:
                        for record in cache_def.records.r:
                            rows.append([getattr(item, "v", None) for item in record])

                    if not headers and not rows:
                        continue

                    df       = pd.DataFrame(rows, columns=headers if headers else None)
                    block_id = str(uuid.uuid4())

                    blocks.append(NormalizedBlock(
                        block_id=block_id,
                        document_id=doc_id,
                        type="table",
                        text=df.to_markdown(index=False),
                        table_data={
                            "headers": headers,
                            "rows":    rows,
                        },
                        source_ref=SourceRef(
                            filename=filename,
                            sheet=sheet.title,
                        ),
                        confidence=cfg.get("extraction_confidence", 1.0),
                        language=self._detect_language(df.to_string()),
                        metadata={
                            "cell_range":        source_range,
                            "enrichment_failed": False,
                        },
                    ))

                except Exception as e:
                    state.setdefault("errors", []).append({
                        "tool":     self.name,
                        "level":    "error",
                        "message":  f"Skipping pivot '{getattr(pivot, 'name', '?')}': {e}",
                        "block_id": None,
                    })
                    continue

        return blocks

    @staticmethod
    def _col_letter(n: int) -> str:
        result = ""
        while n:
            n, r = divmod(n - 1, 26)
            result = chr(65 + r) + result
        return result or "A"


# ------------------------------------------------------------------
# SANDBOX TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_file = "test-data/test.xlsx"
    doc_id    = "doc-excel-001"

    mock_state = {
        "file_path":   test_file,
        "document_id": doc_id,
        "filename":    "test.xlsx",
        "blocks":      [],
        "errors":      [],
    }
    mock_config = {
        "extraction_confidence": 0.95,
        "image_output_dir":      f"uploads/images/{doc_id}",
    }

    tool  = ExcelExtractorTool()
    state = tool.run(mock_state, mock_config)

    blocks   = state["blocks"]
    errors   = state["errors"]

    tables   = [b for b in blocks if b.type == "table"]
    images   = [b for b in blocks if b.type == "image_caption"]
    formulas = [b for b in blocks if b.type == "text"]

    print(f"\n=== Excel Extraction Results ===")
    print(f"Total blocks : {len(blocks)}")
    print(f"  table      : {len(tables)}")
    print(f"  image      : {len(images)}")
    print(f"  formula    : {len(formulas)}")
    print(f"  errors     : {len(errors)}")

    if tables:
        print(f"\n--- table block (sheet: {tables[0].source_ref.sheet}) ---")
        print(f"  cell_range : {tables[0].metadata.get('cell_range')}")
        print(f"  headers    : {tables[0].table_data['headers']}")
        print(f"  rows       : {len(tables[0].table_data['rows'])}")
        print(f"  preview    : {tables[0].text[:200]}")

    if formulas:
        print(f"\n--- formula block ---")
        print(f"  text       : {formulas[0].text}")
        print(f"  cell_range : {formulas[0].metadata.get('cell_range')}")
        print(f"  formula    : {formulas[0].metadata.get('formula')}")

    if images:
        print(f"\n--- image block (sheet: {images[0].source_ref.sheet}) ---")
        print(f"  raw_image_path : {images[0].metadata['raw_image_path']}")
        print(f"  pending_vision : {images[0].metadata['pending_vision']}")

    if errors:
        print(f"\n--- errors ---")
        for e in errors:
            print(f"  [{e['level']}] {e['tool']} — {e['message']}")