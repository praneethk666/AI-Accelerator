import os
import uuid

from backend.core.paths import display_filename
from backend.core.schemas import NormalizedBlock, SourceRef, as_dicts
from backend.extraction.ppt.tool import _detect_image_ext, _save_image_blob


class ImageExtractorTool:
    name = "image_extraction"

    def run(self, state: dict, config: dict) -> dict:
        file_path = state["file_path"]
        document_id = state.get("document_id")
        cfg = config or {}
        doc_id = str(document_id) if document_id else str(uuid.uuid4())
        filename = display_filename(file_path)

        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            state.setdefault("errors", []).append(f"image_extraction: could not read file ({e})")
            state["blocks"] = []
            return state

        try:
            from PIL import Image
            from io import BytesIO
            Image.open(BytesIO(image_bytes)).verify()
        except Exception as e:
            state.setdefault("errors", []).append(f"image_extraction: not a valid/readable image ({e})")
            state["blocks"] = []
            return state

        out_dir = os.path.join("uploads", "images", doc_id)
        os.makedirs(out_dir, exist_ok=True)
        block_id = str(uuid.uuid4())
        raw_path, preview_path = _save_image_blob(image_bytes, out_dir, block_id)

        block = NormalizedBlock(
            block_id=block_id,
            document_id=doc_id,
            type="image_caption",
            text="",
            source_ref=SourceRef(filename=filename),
            confidence=cfg.get("extraction_confidence", 1.0),
            language=cfg.get("default_language", "en"),
            metadata={
                "raw_image_path": raw_path,
                "image_path": preview_path or raw_path,
                "pending_vision": True,
                "enrichment_failed": False,
            },
        )

        state["blocks"] = as_dicts([block])
        state.setdefault("errors", [])
        return state
    
if __name__ == "__main__":
    test_file = "test-data/test.jpg"
    state = {"file_path": test_file, "document_id": "doc-004", "errors": []}
    config = {"extraction_confidence": 0.95, "default_language": "en"}

    result = ImageExtractorTool().run(state, config)
    blocks = result["blocks"]
    images = [b for b in blocks if b["type"] == "image_caption"]

    print(f"{len(images)} image block(s)")
    print("errors:", result.get("errors"))

