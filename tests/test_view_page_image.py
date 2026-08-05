from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.retrieval.view_page_image import ViewPageImageTool


def _fake_store(doc):
    store = MagicMock()
    store.get_document.return_value = doc
    return store


def test_missing_required_args_returns_error():
    assert "error" in ViewPageImageTool().run(document_id="", page=1, question="what?")
    assert "error" in ViewPageImageTool().run(document_id="d1", page=None, question="what?")
    assert "error" in ViewPageImageTool().run(document_id="d1", page=1, question="")


def test_page_not_an_int_returns_error():
    result = ViewPageImageTool().run(document_id="d1", page="not-a-number", question="what?")
    assert "error" in result


def test_unresolvable_pdf_returns_error():
    with patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store(None)):
        result = ViewPageImageTool().run(document_id="d1", page=1, question="what icon is this?")
    assert "error" in result


def test_successful_call_renders_page_and_returns_vision_answer():
    doc = {"file_path": "/fake/manual.pdf"}
    fake_page = MagicMock()
    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-png-bytes"
    fake_page.get_pixmap.return_value = fake_pix
    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 10
    fake_doc.__getitem__.return_value = fake_page

    with patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store(doc)), \
         patch("os.path.isfile", return_value=True), \
         patch("fitz.open", return_value=fake_doc), \
         patch("backend.core.config.load_config", return_value={}), \
         patch("backend.core.vision_client.describe_image",
               return_value="Factor 3 is next to the fire-hazard icon.") as mock_describe:
        result = ViewPageImageTool().run(
            document_id="d1", page=3, question="which icon is next to Factor 3?"
        )

    assert result["document_id"] == "d1"
    assert result["page"] == 3
    assert result["answer"] == "Factor 3 is next to the fire-hazard icon."
    mock_describe.assert_called_once()
    assert mock_describe.call_args[0][0] == b"fake-png-bytes"


def test_page_out_of_range_returns_error():
    doc = {"file_path": "/fake/manual.pdf"}
    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 5

    with patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store(doc)), \
         patch("os.path.isfile", return_value=True), \
         patch("fitz.open", return_value=fake_doc):
        result = ViewPageImageTool().run(document_id="d1", page=99, question="what?")
    assert "error" in result


def test_block_id_uses_saved_crop_directly_skips_page_render(tmp_path, monkeypatch):
    # Real feature added 4-Aug: every figure (deferred or not) already has its
    # crop saved to disk at ingest time -- when block_id is given, use that
    # EXACT saved crop instead of re-rendering the whole page fresh.
    monkeypatch.chdir(tmp_path)
    img_dir = tmp_path / "uploads" / "images" / "d1"
    img_dir.mkdir(parents=True)
    (img_dir / "b1.png").write_bytes(b"saved-crop-bytes")

    store = MagicMock()
    store.get_blocks.return_value = [
        {"block_id": "b1", "metadata": {"image_path": "/images/d1/b1.png"}},
    ]

    with patch("backend.storage.postgres_store.PostgresStore", return_value=store), \
         patch("backend.core.config.load_config", return_value={}), \
         patch("fitz.open") as mock_fitz_open, \
         patch("backend.core.vision_client.describe_image",
               return_value="This crop shows a fire-hazard icon.") as mock_describe:
        result = ViewPageImageTool().run(
            document_id="d1", page=3, question="what icon is this?", block_id="b1"
        )

    mock_fitz_open.assert_not_called()  # never fell back to a page render
    mock_describe.assert_called_once()
    assert mock_describe.call_args[0][0] == b"saved-crop-bytes"
    assert result["used_saved_crop"] is True
    assert result["answer"] == "This crop shows a fire-hazard icon."


def test_block_id_not_found_falls_back_to_page_render():
    doc = {"file_path": "/fake/manual.pdf"}
    fake_page = MagicMock()
    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-page-bytes"
    fake_page.get_pixmap.return_value = fake_pix
    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 10
    fake_doc.__getitem__.return_value = fake_page

    store = MagicMock()
    store.get_document.return_value = doc
    store.get_blocks.return_value = []  # no matching block

    with patch("backend.storage.postgres_store.PostgresStore", return_value=store), \
         patch("os.path.isfile", return_value=True), \
         patch("fitz.open", return_value=fake_doc), \
         patch("backend.core.config.load_config", return_value={}), \
         patch("backend.core.vision_client.describe_image",
               return_value="answer") as mock_describe:
        result = ViewPageImageTool().run(
            document_id="d1", page=3, question="what?", block_id="does-not-exist"
        )

    assert result["used_saved_crop"] is False
    mock_describe.assert_called_once()
    assert mock_describe.call_args[0][0] == b"fake-page-bytes"


def test_no_block_id_uses_page_render_by_default():
    doc = {"file_path": "/fake/manual.pdf"}
    fake_page = MagicMock()
    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-page-bytes"
    fake_page.get_pixmap.return_value = fake_pix
    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 10
    fake_doc.__getitem__.return_value = fake_page

    with patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store(doc)), \
         patch("os.path.isfile", return_value=True), \
         patch("fitz.open", return_value=fake_doc), \
         patch("backend.core.config.load_config", return_value={}), \
         patch("backend.core.vision_client.describe_image", return_value="answer"):
        result = ViewPageImageTool().run(document_id="d1", page=3, question="what?")

    assert result["used_saved_crop"] is False


def test_vision_call_failure_returns_error_not_exception():
    doc = {"file_path": "/fake/manual.pdf"}
    fake_page = MagicMock()
    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-png-bytes"
    fake_page.get_pixmap.return_value = fake_pix
    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 10
    fake_doc.__getitem__.return_value = fake_page

    with patch("backend.storage.postgres_store.PostgresStore",
               return_value=_fake_store(doc)), \
         patch("os.path.isfile", return_value=True), \
         patch("fitz.open", return_value=fake_doc), \
         patch("backend.core.config.load_config", return_value={}), \
         patch("backend.core.vision_client.describe_image",
               side_effect=RuntimeError("provider down")):
        result = ViewPageImageTool().run(document_id="d1", page=3, question="what?")
    assert "error" in result


if __name__ == "__main__":
    test_missing_required_args_returns_error()
    test_page_not_an_int_returns_error()
    test_unresolvable_pdf_returns_error()
    test_successful_call_renders_page_and_returns_vision_answer()
    test_page_out_of_range_returns_error()
    test_vision_call_failure_returns_error_not_exception()
    print("view_page_image tests passed")
