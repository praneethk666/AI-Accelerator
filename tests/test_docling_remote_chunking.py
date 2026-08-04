"""Tests for extract_docling_remote()'s page-range chunking + checkpoint/resume.

Real gap found live, 3-Aug: remote mode was ONE HTTP call for the WHOLE document,
no matter how large. At this project's own measured ~1.2s/page on its GPU server,
a 1147-page document (~23min) would exceed even a generous timeout, and any
failure mid-call loses the ENTIRE document, not just the unprocessed tail. Large
documents now upload+checkpoint page-range chunks one at a time; a restart skips
chunks a prior (crashed/killed) run already checkpointed.

Run: pytest tests/test_docling_remote_chunking.py
"""
from unittest.mock import MagicMock, patch

import fitz
import pytest

from backend.extraction.docling_remote import extract_docling_remote


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _make_pdf(path, n_pages):
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page(width=400, height=600)
    doc.save(path)
    doc.close()


def _config(chunk_pages=4, **extra):
    cfg = {"extraction": {"docling": {
        "server_url": "http://gpu-box:8083", "remote_chunk_pages": chunk_pages,
    }}}
    cfg["extraction"]["docling"].update(extra)
    return cfg


def _text_block(page, text="x"):
    return {
        "block_id": f"t{page}", "document_id": "d1", "type": "text", "text": text,
        "source_ref": {"filename": "m.pdf", "page": page, "sheet": None, "slide": None, "bbox": None},
        "metadata": {},
    }


def _resp(blocks, n_pages):
    r = MagicMock()
    r.status_code = 200
    r.ok = True
    r.json.return_value = {"blocks": blocks, "n_pages": n_pages, "elapsed_s": 0.5}
    return r


def _patch_no_op_deps():
    """Common patches that keep this test about CHUNKING, not figure-captioning
    or table-escalation (both covered by their own test files)."""
    return patch.multiple(
        "backend.extraction.docling_pdf.docling_extract",
        _local_table_engine=MagicMock(return_value=False),
    )


def test_small_document_takes_single_call_path(tmp_path):
    """total_pages <= chunk_pages -> exactly the old single-call behavior."""
    pdf = str(tmp_path / "small.pdf")
    _make_pdf(pdf, 3)
    server_blocks = [_text_block(1), _text_block(2), _text_block(3)]

    with patch("requests.post", return_value=_resp(server_blocks, 3)) as mock_post, \
         patch("backend.extraction.docling_remote._caption_remote_figures",
               side_effect=lambda blocks, *a, **k: blocks), \
         _patch_no_op_deps():
        result = extract_docling_remote(pdf, "d1", _config(chunk_pages=10))

    assert mock_post.call_count == 1
    assert [b["source_ref"]["page"] for b in result] == [1, 2, 3]


def test_large_document_splits_into_chunks_with_remapped_pages(tmp_path):
    pdf = str(tmp_path / "big.pdf")
    _make_pdf(pdf, 10)   # chunk_pages=4 -> chunks [1-4] [5-8] [9-10]

    # Each server call numbers ITS OWN pages 1-based within the uploaded sub-PDF.
    responses = [
        _resp([_text_block(1), _text_block(2), _text_block(3), _text_block(4)], 4),
        _resp([_text_block(1), _text_block(2), _text_block(3), _text_block(4)], 4),
        _resp([_text_block(1), _text_block(2)], 2),
    ]

    with patch("requests.post", side_effect=responses) as mock_post, \
         patch("backend.extraction.docling_pdf.docling_extract._checkpoint_page_blocks") as mock_ckpt, \
         patch("backend.extraction.docling_remote._already_checkpointed_blocks", return_value=[]), \
         patch("backend.extraction.docling_remote._caption_remote_figures",
               side_effect=lambda blocks, *a, **k: blocks), \
         _patch_no_op_deps():
        result = extract_docling_remote(pdf, "d1", _config(chunk_pages=4))

    assert mock_post.call_count == 3
    # remapped to ORIGINAL document page numbers, not the per-chunk 1-based ones
    assert sorted(b["source_ref"]["page"] for b in result) == list(range(1, 11))
    # checkpointed once per page (10 pages total)
    checkpointed_pages = sorted(c.args[1] for c in mock_ckpt.call_args_list)
    assert checkpointed_pages == list(range(1, 11))


def test_resume_skips_chunks_already_checkpointed(tmp_path):
    pdf = str(tmp_path / "big.pdf")
    _make_pdf(pdf, 10)   # chunks [1-4] [5-8] [9-10]

    # Pages 1-4 (the whole first chunk) already checkpointed by a prior run.
    cached = [_text_block(p, text="cached") for p in range(1, 5)]
    responses = [
        _resp([_text_block(1), _text_block(2), _text_block(3), _text_block(4)], 4),  # chunk [5-8]
        _resp([_text_block(1), _text_block(2)], 2),                                   # chunk [9-10]
    ]

    with patch("requests.post", side_effect=responses) as mock_post, \
         patch("backend.extraction.docling_pdf.docling_extract._checkpoint_page_blocks"), \
         patch("backend.extraction.docling_remote._already_checkpointed_blocks",
               return_value=cached), \
         patch("backend.extraction.docling_remote._caption_remote_figures",
               side_effect=lambda blocks, *a, **k: blocks), \
         _patch_no_op_deps():
        result = extract_docling_remote(pdf, "d1", _config(chunk_pages=4))

    # only 2 HTTP calls -- the already-cached first chunk was never re-uploaded
    assert mock_post.call_count == 2
    assert sorted(b["source_ref"]["page"] for b in result) == list(range(1, 11))
    reused = [b for b in result if b.get("text") == "cached"]
    assert len(reused) == 4


def test_partial_chunk_overlap_is_not_treated_as_done(tmp_path):
    """Only pages 1-2 of chunk [1-4] are cached (a crash could plausibly leave a
    partial state depending on timing) -- the WHOLE chunk must still be redone,
    since a partial chunk isn't a safe thing to trust as complete."""
    pdf = str(tmp_path / "big.pdf")
    _make_pdf(pdf, 4)
    cached = [_text_block(1), _text_block(2)]   # only half of the single chunk
    responses = [_resp([_text_block(1), _text_block(2), _text_block(3), _text_block(4)], 4)]

    with patch("requests.post", side_effect=responses) as mock_post, \
         patch("backend.extraction.docling_pdf.docling_extract._checkpoint_page_blocks"), \
         patch("backend.extraction.docling_remote._already_checkpointed_blocks",
               return_value=cached), \
         patch("backend.extraction.docling_remote._caption_remote_figures",
               side_effect=lambda blocks, *a, **k: blocks), \
         _patch_no_op_deps():
        extract_docling_remote(pdf, "d1", _config(chunk_pages=4))

    assert mock_post.call_count == 1   # redone, not skipped
