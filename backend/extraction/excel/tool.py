import os
import uuid
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from backend.core.schemas import NormalizedBlock, SourceRef, as_dicts


class ExcelExtractorTool:
    """
    Extracts tables, embedded images, formulas, and pivot table data from Excel files.
    Output strictly follows the NormalizedBlock contract in schemas.py.
    """
    name = "excel_extraction"

    def run(self, state: dict, config: dict) -> dict:
        file_path   = state["file_path"]
        document_id = state.get("document_id")
        blocks      = self._extract(file_path, document_id=document_id, config=config)
        state["blocks"] = as_dicts(blocks)
        state.setdefault("errors", [])
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[NormalizedBlock]:
        blocks: List[NormalizedBlock] = []
        cfg      = config or {}
        doc_id   = str(document_id) if document_id else str(uuid.uuid4())
        filename = os.path.basename(file_path)

        try:
            wb = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception as e:
            return blocks

        # ── 1. Table blocks ────────────────────────────────────────────
        for sheet_name, df in wb.items():
            try:
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                df = df.fillna("")

                block_id   = str(uuid.uuid4())
                cell_range = f"{sheet_name}!A1:{self._col_letter(len(df.columns))}{len(df) + 1}"

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
                    language=cfg.get("default_language", "en"),
                    metadata={
                        "cell_range":        cell_range,
                        "enrichment_failed": cfg.get("enrichment_failed_flag", False),
                    },
                ))

            except Exception:
                continue

        # ── 2. Embedded image blocks ───────────────────────────────────
        blocks.extend(self._extract_images(file_path, doc_id, filename, cfg))

        # ── 3. Formula blocks ──────────────────────────────────────────
        blocks.extend(self._extract_formulas(file_path, doc_id, filename, cfg))

        # ── 4. Pivot table blocks ──────────────────────────────────────
        blocks.extend(self._extract_pivots(file_path, doc_id, filename, cfg))

        return blocks

    def _extract_images(
        self,
        file_path: str,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
    ) -> List[NormalizedBlock]:
        import openpyxl

        blocks: List[NormalizedBlock] = []

        try:
            wb = openpyxl.load_workbook(file_path)
        except Exception:
            return blocks

        out_dir = os.path.join("uploads", "images", doc_id)
        os.makedirs(out_dir, exist_ok=True)

        for sheet in wb.worksheets:
            for image in getattr(sheet, "_images", []):
                try:
                    block_id   = str(uuid.uuid4())
                    raw_path   = os.path.join(out_dir, f"{block_id}_raw.png")
                    image_data = image.ref.read() if hasattr(image.ref, "read") else bytes(image.ref)
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
                        language=cfg.get("default_language", "en"),
                        metadata={
                            "raw_image_path":    raw_path,
                            "pending_vision":    True,
                            "enrichment_failed": False,
                        },
                    ))

                except Exception:
                    continue

        return blocks

    def _extract_formulas(
        self,
        file_path: str,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
    ) -> List[NormalizedBlock]:
        import openpyxl

        blocks: List[NormalizedBlock] = []

        try:
            wb = openpyxl.load_workbook(file_path, data_only=False)
        except Exception:
            return blocks

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
                            language=cfg.get("default_language", "en"),
                            metadata={
                                "cell_ref":          cell_ref,
                                "formula":           formula,
                                "sheet":             sheet.title,
                                "enrichment_failed": False,
                            },
                        ))

                    except Exception:
                        continue

        return blocks

    def _extract_pivots(
        self,
        file_path: str,
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
    ) -> List[NormalizedBlock]:
        import openpyxl

        blocks: List[NormalizedBlock] = []

        try:
            wb = openpyxl.load_workbook(file_path, data_only=True)
        except Exception:
            return blocks

        for sheet in wb.worksheets:
            for pivot in getattr(sheet, "_pivots", []):
                try:
                    pivot_name   = getattr(pivot, "name", "unnamed_pivot")
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
                        language=cfg.get("default_language", "en"),
                        metadata={
                            "pivot_name":         pivot_name,
                            "pivot_source_range": source_range,
                            "cell_range":         source_range,
                            "enrichment_failed":  False,
                        },
                    ))

                except Exception:
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

    mock_state  = {"file_path": test_file, "document_id": "doc-001"}
    mock_config = {"extraction_confidence": 0.95, "default_language": "en"}

    tool    = ExcelExtractorTool()
    state   = tool.run(mock_state, mock_config)
    results = state["blocks"]

    tables   = [b for b in results if b.type == "table" and "pivot_name" not in b.metadata]
    images   = [b for b in results if b.type == "image_caption"]
    formulas = [b for b in results if b.type == "text" and "formula" in b.metadata]
    pivots   = [b for b in results if b.type == "table" and "pivot_name" in b.metadata]

    print(f"Extracted {len(tables)} table(s), {len(images)} image(s), {len(formulas)} formula(s), {len(pivots)} pivot(s).\n")

    if tables:
        print("--- First table block ---")
        print(f"  text preview: {(tables[0].text or '')[:200]}")
        print(f"  headers: {tables[0].table_data['headers']}")

    if formulas:
        print("\n--- First formula block ---")
        print(f"  text: {formulas[0].text}")
        print(f"  cell_ref: {formulas[0].metadata['cell_ref']}")

    if images:
        print("\n--- First image block ---")
        print(f"  raw_image_path: {images[0].metadata['raw_image_path']}")
        print(f"  pending_vision: {images[0].metadata['pending_vision']}")

    if pivots:
        print("\n--- First pivot block ---")
        print(f"  pivot_name: {pivots[0].metadata['pivot_name']}")
        print(f"  text preview: {(pivots[0].text or '')[:200]}")