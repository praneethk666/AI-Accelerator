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
    """Fallback sniffer for common image formats and Microsoft vectors."""
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
    """
    Extracts text, notes, tables, and embedded images/charts from PowerPoint files.
    Output follows the NormalizedBlock contract as directed by lead.
    """
    name = "ppt_extraction"

    def run(self, state: dict, config: dict) -> dict:
        file_path   = state.get("file_path")
        document_id = state.get("document_id")
        filename    = state.get("filename", os.path.basename(file_path) if file_path else "unknown.pptx")

        if not document_id or not file_path:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  "Missing document_id or file_path in state — aborting extraction.",
                "block_id": None,
            })
            return state

        # LangGraph State merging structure as requested by lead
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
        """Recursively yields shapes, traversing inside grouped shapes."""
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
        import tempfile

        blocks: List[NormalizedBlock] = []
        cfg    = config or {}
        doc_id = str(document_id)

        target_file = file_path
        
        # Intercept legacy .ppt and use Microsoft PowerPoint (Windows only) to convert it
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".ppt":
            try:
                import win32com.client
                import pythoncom
                
                # Initialize COM for multithreaded environments
                pythoncom.CoInitialize()
                
                powerpoint = win32com.client.Dispatch("PowerPoint.Application")
                
                # COM requires absolute paths to work
                abs_file_path = os.path.abspath(file_path)
                
                temp_dir  = tempfile.mkdtemp()
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                target_file = os.path.abspath(os.path.join(temp_dir, f"{base_name}.pptx"))
                
                # Open the old PPT silently and save as PPTX (24 is the MS enum for PPTX)
                presentation = powerpoint.Presentations.Open(abs_file_path, WithWindow=False)
                presentation.SaveAs(target_file, 24)
                presentation.Close()
                
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "info", 
                    "message":  f"Converted legacy .ppt to .pptx via Windows COM.",
                    "block_id": None,
                })
            except ImportError:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  "pywin32 is not installed. Run 'pip install pywin32'.",
                    "block_id": None,
                })
                return blocks
            except Exception as e:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  f"Failed to convert .ppt (Requires MS PowerPoint installed on Windows): {e}",
                    "block_id": None,
                })
                return blocks

        # --- Extract the converted (or original) .pptx file ---
        try:
            prs = Presentation(target_file)
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool":     self.name,
                "level":    "error",
                "message":  f"Failed to open {target_file}: {e}",
                "block_id": None,
            })
            return blocks

        for slide_index, slide in enumerate(prs.slides):
            slide_num = slide_index + 1
            try:
                text_block = self._extract_slide_text(slide, slide_num, doc_id, filename, cfg)
                if text_block:
                    blocks.append(text_block)

                blocks.extend(self._extract_slide_tables(slide, slide_num, doc_id, filename, cfg, state))
                blocks.extend(self._extract_slide_images(slide, slide_num, doc_id, filename, cfg, state))

            except Exception as e:
                state.setdefault("errors", []).append({
                    "tool":     self.name,
                    "level":    "error",
                    "message":  f"Skipping slide {slide_num}: {e}",
                    "block_id": None,
                })
                continue

        # Clean up the temporary .pptx file
        if target_file != file_path and os.path.exists(target_file):
            try:
                os.remove(target_file)
            except Exception:
                pass

        return blocks

        for slide_index, slide in enumerate(prs.slides):
            slide_num = slide_index + 1
            try:
                text_block = self._extract_slide_text(slide, slide_num, doc_id, filename, cfg)
                if text_block:
                    blocks.append(text_block)

                blocks.extend(self._extract_slide_tables(slide, slide_num, doc_id, filename, cfg, state))
                blocks.extend(self._extract_slide_images(slide, slide_num, doc_id, filename, cfg, state))

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
        doc_id: str,
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
            document_id=doc_id,
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
        doc_id: str,
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
                        slide=slide_num,
                    ),
                    confidence=cfg.get("extraction_confidence", 1.0),
                    language=self._detect_language(df.to_string()),
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
        doc_id: str,
        filename: str,
        cfg: Dict[str, Any],
        state: dict,
    ) -> List[NormalizedBlock]:
        blocks  = []
        out_dir = cfg.get("image_output_dir", os.path.join("uploads", "images", doc_id))
        os.makedirs(out_dir, exist_ok=True)

        for shape in self._iter_shapes(slide.shapes):
            image_bytes = None
            ext = ""

            try:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_bytes = shape.image.blob
                    ext = getattr(shape.image, "ext", "").lower()
                    if not ext:
                        ext = _detect_image_ext(image_bytes)

                elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                    chart_part = shape.chart._part
                    for rel in chart_part.rels.values():
                        if "image" in rel.reltype:
                            image_bytes = rel.target_part.blob
                            content_type = getattr(rel.target_part, "content_type", "") or ""
                            ext = content_type.split("/")[-1].lower() if "/" in content_type else ""
                            if not ext or ext in ["octet-stream", "tmp", ""]:
                                ext = _detect_image_ext(image_bytes)
                            break

                if not image_bytes:
                    continue

                ext = ext.lower()
                if ext == "x-emf": ext = "emf"
                if ext == "x-wmf": ext = "wmf"
                ext = "jpg" if ext == "jpeg" else ext

                block_id = str(uuid.uuid4())
                raw_path = os.path.join(out_dir, f"{block_id}_raw.{ext}")

                with open(raw_path, "wb") as f:
                    f.write(image_bytes)

                blocks.append(NormalizedBlock(
                    block_id=block_id,
                    document_id=doc_id,
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
if __name__ == "__main__":
    test_file = "test-data/test.pptx"

    mock_state  = {"file_path": test_file, "document_id": "doc-002"}
    mock_config = {"extraction_confidence": 0.95, "default_language": "en"}

    tool    = PPTExtractorTool()
    state   = tool.run(mock_state, mock_config)
    results = state.get("blocks", [])

    texts   = [b for b in results if b.type == "text"]
    tables  = [b for b in results if b.type == "table"]
    images  = [b for b in results if b.type == "image"] # Updated to "image" based on your recent fix

    print(f"Extracted {len(texts)} text block(s), {len(tables)} table(s), {len(images)} image/chart block(s).\n")

    if texts:
        print("--- First text block ---")
        print(f"  text preview: {texts[0].text[:200]}")

    if tables:
        print("\n--- First table block ---")
        print(f"  text preview: {(tables[0].text or '')[:200]}")
        print(f"  headers: {tables[0].table_data['headers']}")

    if images:
        print("\n--- First image block ---")
        print(f"  raw_image_path: {images[0].metadata['raw_image_path']}")
        print(f"  pending_vision: {images[0].metadata['pending_vision']}")