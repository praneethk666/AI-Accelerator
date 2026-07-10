import os
import pytest

from backend.extraction.image.tool import ImageExtractorTool

IMAGE_FILE = "test-data/test.jpg"

pytestmark = pytest.mark.skipif(
    not os.path.exists(IMAGE_FILE),
    reason="test-data fixture not present (git-ignored)",
)


def test_image_blocks_nonempty():
    state = ImageExtractorTool().run({"file_path": IMAGE_FILE, "document_id": "test-doc-004"}, {})
    assert state["blocks"], "No blocks extracted from image file"
    assert all(isinstance(b, dict) for b in state["blocks"]), "blocks must be plain dicts"


def test_image_errors_key_exists():
    state = ImageExtractorTool().run({"file_path": IMAGE_FILE, "document_id": "test-doc-004"}, {})
    assert "errors" in state


def test_image_block_metadata():
    state = ImageExtractorTool().run({"file_path": IMAGE_FILE, "document_id": "test-doc-004"}, {})
    images = [b for b in state["blocks"] if b["type"] == "image_caption"]
    assert images, "No image blocks found"
    for img in images:
        meta = img.get("metadata") or {}
        assert meta.get("raw_image_path")
        assert meta.get("pending_vision") is True

        