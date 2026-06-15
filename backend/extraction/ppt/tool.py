import os
import uuid
import pandas as pd
from typing import List, Dict, Any, Optional
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
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


class PPTExtractorTool(Tool):
    name = "ppt_extraction"

    def run(self, state: dict, config: dict) -> dict:
        file_path   = state.get("file_path")
        document_id = state.get("document_id")
        filename    = state.get("filename", os.path.basename(file_path) if file_path else "unknown.pptx")

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

    def _iter_shapes(self, shapes):
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from self._iter_shapes(shape.shapes)
            else:
                yield shape

    def _extract(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        config: Optional[Dict[str, Any]],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks = []
        cfg    = config or {}

        try:
            prs = Presentation(file_path)
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  f"Failed to open {file_path}: {e}",
                "block_id": None,
            })
            return blocks

        for slide_index, slide in enumerate(prs.slides):
            slide_num = slide_index + 1
            try:
                text_block = self._extract_slide_text(slide, slide_num, document_id, filename, cfg)
                if text_block:
                    blocks.append(text_block)

                blocks.extend(self._extract_slide_tables(slide, slide_num, document_id, filename, cfg, state))
                blocks.extend(self._extract_slide_images(slide, slide_num, document_id, filename, cfg, state))

            except Exception as e:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  f"Skipping slide {slide_num}: {e}",
                    "block_id": None,
                })
                continue

        return blocks

    def _extract_slide_text(
        self,
        slide,
        slide_num: int,
        document_id: str,
        filename: str,
        cfg: Dict[str, Any],
    ) -> Optional[NormalizedBlock]:
        parts = []

        for shape in self._iter_shapes(slide.shapes):
            if shape.has_table or shape.shape_type == MSO_SHAPE_TYPE.CHART:
                continue
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Speaker Notes: {notes}")

        full_text = "\n".join(parts).strip()
        if not full_text:
            return None

        return NormalizedBlock(
            block_id=str(uuid.uuid4()),
            document_id=document_id,
            type="text",
            text=full_text,
            source_ref=SourceRef(
                filename=filename,
                slide=slide_num,
            ),
            confidence=cfg.get("extraction_confidence", 1.0),
            language=self._detect_language(full_text),
            metadata={
                "enrichment_failed": False,
            },
        )

    def _extract_slide_tables(
        self,
        slide,
        slide_num: int,
        document_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks = []

        for shape in self._iter_shapes(slide.shapes):
            if not shape.has_table:
                continue
            try:
                tbl     = shape.table
                rows    = []
                headers = []

                for i, row in enumerate(tbl.rows):
                    cells = [cell.text.strip() for cell in row.cells]
                    if i == 0:
                        headers = cells
                    else:
                        rows.append(cells)

                df       = pd.DataFrame(rows, columns=headers if headers else None)
                md_text  = df.to_markdown(index=False)
                block_id = str(uuid.uuid4())

                blocks.append(NormalizedBlock(
                    block_id=block_id,
                    document_id=document_id,
                    type="table",
                    text=md_text,
                    table_data={
                        "headers": headers,
                        "rows":    rows,
                    },
                    source_ref=SourceRef(
                        filename=filename,
                        slide=slide_num,
                    ),
                    confidence=cfg.get("extraction_confidence", 1.0),
                    language=self._detect_language(md_text),
                    metadata={
                        "enrichment_failed": False,
                    },
                ))

            except Exception as e:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  f"Skipping table on slide {slide_num}: {e}",
                    "block_id": None,
                })
                continue

        return blocks

    def _extract_slide_images(
        self,
        slide,
        slide_num: int,
        document_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks  = []
        out_dir = cfg.get("image_output_dir", os.path.join("uploads", "images", document_id))
        os.makedirs(out_dir, exist_ok=True)

        for shape in self._iter_shapes(slide.shapes):
            image_bytes = None
            ext         = ""

            try:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_bytes = shape.image.blob
                    ext = getattr(shape.image, "ext", "") or ""

                elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                    chart_part = shape.chart._part
                    for rel in chart_part.rels.values():
                        if "image" in rel.reltype:
                            image_bytes = rel.target_part.blob
                            content_type = getattr(rel.target_part, "content_type", "") or ""
                            ext = content_type.split("/")[-1] if "/" in content_type else ""
                            break

                if not image_bytes:
                    continue

                ext = ext.lower().strip()
                if ext in ("", "octet-stream", "tmp"):
                    ext = _detect_image_ext(image_bytes)
                if ext == "x-emf": ext = "emf"
                if ext == "x-wmf": ext = "wmf"
                if ext == "jpeg":  ext = "jpg"

                block_id = str(uuid.uuid4())
                raw_path = os.path.join(out_dir, f"{block_id}_raw.{ext}")

                with open(raw_path, "wb") as f:
                    f.write(image_bytes)

                blocks.append(NormalizedBlock(
                    block_id=block_id,
                    document_id=document_id,
                    type="image_caption",
                    text="",
                    source_ref=SourceRef(
                        filename=filename,
                        slide=slide_num,
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
                    "message":  f"Skipping image/chart on slide {slide_num}: {e}",
                    "block_id": None,
                })
                continue

        return blocks


# ------------------------------------------------------------------
# SANDBOX TEST
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# SANDBOX TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os
    
    test_file = "test-data/waste3.pptx" 
    doc_id    = "doc-ppt-001"

    mock_state = {
        "file_path":   test_file,
        "document_id": doc_id,
        "filename":    "waste3.pptx",
        "blocks":      [],
        "errors":      [],
    }
    mock_config = {
        "extraction_confidence": 0.95,
        "image_output_dir":      f"uploads/images/{doc_id}",
    }

    tool  = PPTExtractorTool()
    state = tool.run(mock_state, mock_config)

    blocks = state.get("blocks", [])
    errors = state.get("errors", [])

    texts   = [b for b in blocks if b.type == "text"]
    tables  = [b for b in blocks if b.type == "table"]
    images  = [b for b in blocks if b.type == "image_caption"]

    print(f"\n=========================================")
    print(f"===        PPT EXTRACTION SUMMARY     ===")
    print(f"=========================================")
    print(f"Total Blocks Extracted : {len(blocks)}")
    print(f"  ├─ Text Blocks       : {len(texts)}")
    print(f"  ├─ Tables            : {len(tables)}")
    print(f"  ├─ Images/Charts     : {len(images)}")
    print(f"  └─ Errors            : {len(errors)}")

    # Loop through ALL text blocks (Slides + Speaker Notes)
    if texts:
        print(f"\n=========================================")
        print(f"===          ALL TEXT BLOCKS          ===")
        print(f"=========================================")
        for idx, txt in enumerate(texts, start=1):
            print(f"\n[Text {idx}/{len(texts)}] Slide: {txt.source_ref.slide}")
            print(f"  └─ Full Text :\n{txt.text}")
            print("-" * 50)

    # Loop through ALL table blocks
    if tables:
        print(f"\n=========================================")
        print(f"===          ALL TABLE BLOCKS         ===")
        print(f"=========================================")
        for idx, table in enumerate(tables, start=1):
            print(f"\n[Table {idx}/{len(tables)}] Slide: {table.source_ref.slide}")
            print(f"  ├─ Headers     : {table.table_data.get('headers', [])}")
            print(f"  ├─ Total Rows  : {len(table.table_data.get('rows', []))}")
            print(f"  ├─ First 3 Rows: {table.table_data.get('rows', [])[:3]}")
            print(f"  └─ Markdown Representation :\n{table.text}")
            print("-" * 50)

    # Loop through ALL extracted image/chart metadata
    if images:
        print(f"\n=========================================")
        print(f"===          ALL IMAGE BLOCKS         ===")
        print(f"=========================================")
        for idx, img in enumerate(images, start=1):
            print(f"\n[Image {idx}/{len(images)}] Slide: {img.source_ref.slide}")
            print(f"  ├─ Raw Path      : {img.metadata.get('raw_image_path')}")
            print(f"  └─ Pending Vision: {img.metadata.get('pending_vision')}")

    # Show any errors logged along the pipeline
    if errors:
        print(f"\n=========================================")
        print(f"===               ERRORS              ===")
        print(f"=========================================")
        for e in errors:
            print(f"  [{e['level'].upper()}] {e['tool']} — {e['message']}")