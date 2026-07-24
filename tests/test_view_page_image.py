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
