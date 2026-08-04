"""Tests for reading Docling's OWN figure<->caption link (item.captions /
item.caption_text()) instead of guessing crop padding to catch the caption, PLUS
the figure-to-figure exclusion-zone padding fix (see the bottom of this file).

Real bug (3-Aug): the crop region around a figure was sized by heuristic padding
only (a fixed collision-aware pad + a fixed caption_pad_pts in PDFCropper), so
long/oddly-placed captions sometimes got cut off, while dense pages sometimes had
unrelated neighboring text pulled in as noise. Docling already parses "Figure 12:
..." as its own linked text item -- _picture_caption_info() reads that link, and
_figure_block() now (a) unions the caption's own bbox into the crop region so it's
guaranteed to be physically inside the image, and (b) passes the exact caption text
to the VLM gate as ground truth, then hard-appends it to the final caption so the
real wording survives even if the model paraphrases or drops it.

Run: pytest tests/test_docling_caption_link.py
"""
from unittest.mock import patch

import pytest

from backend.extraction.docling_pdf.docling_extract import (
    _picture_caption_info,
    _figure_block,
)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """_figure_block writes crops to a relative 'uploads/images/<doc_id>/' path --
    every gate_result={'keep': True, ...} test below reaches that write. Without
    this the fake PNG bytes land in the REAL repo's uploads/ dir on every run."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Fakes standing in for docling_core objects (no docling dependency in this test)
# ---------------------------------------------------------------------------

class FakeBBox:
    """No to_top_left_origin() -> _bbox_topleft_pts falls to its raw l/t/r/b path."""
    def __init__(self, l, t, r, b):
        self.l, self.t, self.r, self.b = l, t, r, b


class FakeProv:
    def __init__(self, page_no, bbox):
        self.page_no = page_no
        self.bbox = bbox


class FakeTextItem:
    def __init__(self, page_no, bbox, text=""):
        self.prov = [FakeProv(page_no, bbox)]
        self.text = text


class FakeRef:
    def __init__(self, target):
        self._target = target

    def resolve(self, doc):
        return self._target


class FakePictureItem:
    def __init__(self, captions):
        self.captions = captions

    def caption_text(self, doc):
        return " ".join(c.resolve(doc).text for c in self.captions)


class FakeSize:
    def __init__(self, w, h):
        self.width, self.height = w, h


class FakePage:
    def __init__(self, w, h):
        self.size = FakeSize(w, h)


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages


# ---------------------------------------------------------------------------
# _picture_caption_info
# ---------------------------------------------------------------------------

def test_resolves_text_and_bbox_from_linked_caption():
    doc = FakeDoc({1: FakePage(400, 600)})
    cap_item = FakeTextItem(1, FakeBBox(40, 500, 360, 520), text="Figure 3: Battery pack")
    item = FakePictureItem([FakeRef(cap_item)])

    info = _picture_caption_info(item, doc, 1)

    assert info["text"] == "Figure 3: Battery pack"
    assert info["bbox"] == [40, 500, 360, 520]


def test_none_when_picture_has_no_caption_link():
    item = FakePictureItem([])
    assert _picture_caption_info(item, FakeDoc({1: FakePage(400, 600)}), 1) is None


def test_bbox_none_when_caption_ref_is_on_a_different_page():
    # caption_text() itself doesn't filter by page (matches real Docling behavior) --
    # only the bbox union is page-scoped, since a mismatched page's bbox would be
    # meaningless for sizing THIS page's crop.
    doc = FakeDoc({1: FakePage(400, 600)})
    cap_item = FakeTextItem(2, FakeBBox(40, 500, 360, 520), text="Figure 3")
    item = FakePictureItem([FakeRef(cap_item)])

    info = _picture_caption_info(item, doc, 1)

    assert info == {"text": "Figure 3", "bbox": None}


def test_none_when_caption_text_is_blank():
    doc = FakeDoc({1: FakePage(400, 600)})
    cap_item = FakeTextItem(1, FakeBBox(40, 500, 360, 520), text="   ")
    item = FakePictureItem([FakeRef(cap_item)])
    assert _picture_caption_info(item, doc, 1) is None


def test_multiple_caption_refs_union_into_one_bbox():
    doc = FakeDoc({1: FakePage(400, 600)})
    c1 = FakeTextItem(1, FakeBBox(40, 500, 200, 515), text="Figure 3:")
    c2 = FakeTextItem(1, FakeBBox(40, 516, 360, 530), text="Battery pack detail")
    item = FakePictureItem([FakeRef(c1), FakeRef(c2)])

    info = _picture_caption_info(item, doc, 1)

    assert info["text"] == "Figure 3: Battery pack detail"
    assert info["bbox"] == [40, 500, 360, 530]


# ---------------------------------------------------------------------------
# _figure_block: crop union + known-caption passthrough + hard-append guarantee
# ---------------------------------------------------------------------------

_FAKE_PDF = "/nonexistent/fake.pdf"  # fitz.open() fails -> falls to the except
                                     # branch, which must still use the unioned
                                     # crop_bbox (this is exactly what the fix covers)


def _run_figure_block(cap_info, gate_result):
    captured = {}

    def fake_crop_region(self, pdf_path, page_no, padded_bbox):
        captured["padded_bbox"] = padded_bbox
        return b"fakepngbytes"

    with patch("backend.vision.pdf_cropper.PDFCropper.crop_region", fake_crop_region), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        mock_gate.return_value = gate_result
        block = _figure_block(_FAKE_PDF, "doc1", 1, "manual.pdf",
                              [100, 100, 300, 200], ["some page text"], {},
                              cap_info=cap_info)
    return block, captured, mock_gate


def test_no_cap_info_behaves_as_before():
    block, captured, mock_gate = _run_figure_block(
        cap_info=None,
        gate_result={"keep": True, "kind": "diagram", "caption": "a diagram"},
    )
    assert captured["padded_bbox"] == [100, 100, 300, 200]
    assert mock_gate.call_args.kwargs["known_caption"] == ""
    assert block["text"] == "a diagram"
    assert "docling_caption" not in block["metadata"]


def test_caption_bbox_unioned_into_crop_region():
    cap_info = {"text": "Figure 3: Battery pack", "bbox": [80, 200, 320, 230]}
    block, captured, mock_gate = _run_figure_block(
        cap_info=cap_info,
        gate_result={"keep": True, "kind": "photo", "caption": "a battery"},
    )
    # union of figure bbox [100,100,300,200] and caption bbox [80,200,320,230]
    assert captured["padded_bbox"] == [80, 100, 320, 230]


def test_known_caption_passed_to_gate():
    cap_info = {"text": "Figure 3: Battery pack", "bbox": None}
    _, _, mock_gate = _run_figure_block(
        cap_info=cap_info,
        gate_result={"keep": True, "kind": "photo", "caption": "a battery"},
    )
    assert mock_gate.call_args.kwargs["known_caption"] == "Figure 3: Battery pack"


def test_known_caption_hard_appended_when_model_drops_it():
    cap_info = {"text": "Figure 3: Battery pack", "bbox": None}
    block, _, _ = _run_figure_block(
        cap_info=cap_info,
        # model's caption doesn't mention "Figure 3" or "Battery pack" at all
        gate_result={"keep": True, "kind": "photo", "caption": "A grey rectangular object."},
    )
    assert "Figure 3: Battery pack" in block["text"]
    assert "A grey rectangular object." in block["text"]
    assert block["metadata"]["docling_caption"] == "Figure 3: Battery pack"


def test_known_caption_not_duplicated_when_model_already_included_it():
    cap_info = {"text": "Figure 3: Battery pack", "bbox": None}
    block, _, _ = _run_figure_block(
        cap_info=cap_info,
        gate_result={"keep": True, "kind": "photo",
                     "caption": "Figure 3: Battery pack, showing the terminal layout."},
    )
    assert block["text"].count("Figure 3: Battery pack") == 1


def test_dropped_furniture_returns_none_even_with_cap_info():
    cap_info = {"text": "Figure 3: Battery pack", "bbox": None}
    block, _, _ = _run_figure_block(
        cap_info=cap_info,
        gate_result={"keep": False, "kind": "logo", "caption": ""},
    )
    assert block is None


# ---------------------------------------------------------------------------
# Exclusion-zone padding: real bug found on a scanned Hammond service manual (no
# text layer at all -> the pre-existing text-block collision guard had nothing to
# check against) -- two figures 5.6pt apart on the same page, default padding
# (~65pt via PDFCropper's caption_pad_pts) would bleed straight into the neighbor.
# Fixed by also feeding OTHER kept figures on the page into the same collision
# check used for text blocks. Needs a REAL pdf (fitz.open must succeed) since the
# collision math lives inside the try block.
# ---------------------------------------------------------------------------

def _make_blank_pdf(path, width=600, height=800):
    import fitz
    doc = fitz.open()
    doc.new_page(width=width, height=height)
    doc.save(path)
    doc.close()


def _run_with_real_pdf(tmp_path, fig_bbox, other_figure_bboxes=None):
    import fitz
    pdf_path = str(tmp_path / "blank.pdf")
    _make_blank_pdf(pdf_path)

    captured = {}

    def fake_crop_region(self, pdf_path, page_no, padded_bbox, **kwargs):
        captured["padded_bbox"] = padded_bbox
        captured["kwargs"] = kwargs
        return b"fakepngbytes"

    with patch("backend.vision.pdf_cropper.PDFCropper.crop_region", fake_crop_region), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        mock_gate.return_value = {"keep": True, "kind": "photo", "caption": "a photo"}
        _figure_block(pdf_path, "doc1", 1, "manual.pdf", fig_bbox, [], {},
                     other_figure_bboxes=other_figure_bboxes)
    return captured


def test_padding_capped_by_neighboring_figure(tmp_path):
    # 5pt gap below the figure -- default padding (10pt here, then PDFCropper's own
    # unconditional ~52pt caption_pad_pts on top) would bleed straight through it.
    fig_bbox = [100, 300, 300, 400]
    neighbor_bbox = [100, 405, 300, 500]

    captured = _run_with_real_pdf(tmp_path, fig_bbox, other_figure_bboxes=[neighbor_bbox])

    # stage 1: the collision-aware bbox passed to crop_region must not cross the gap
    assert captured["padded_bbox"][3] <= 405
    # stage 2: PDFCropper's OWN much larger caption padding must be suppressed too --
    # otherwise it silently reintroduces the exact bleed stage 1 just prevented
    assert captured["kwargs"].get("bottom_frac") == 0.0
    assert captured["kwargs"].get("caption_pad_pts") == 0.0


def test_padding_not_capped_without_neighbor_info(tmp_path):
    fig_bbox = [100, 300, 300, 400]

    captured = _run_with_real_pdf(tmp_path, fig_bbox)

    # no neighbor info -> the pre-existing fixed pad (10pt) applies in full, since
    # there's no text layer on this blank test page to collide with either
    assert captured["padded_bbox"][3] == 410.0
    assert captured["kwargs"] == {}


# ---------------------------------------------------------------------------
# Deferred-captioning path (_figure_block(defer=True))
# ---------------------------------------------------------------------------

def test_defer_skips_gate_and_marks_pending():
    with patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepngbytes"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        block = _figure_block(_FAKE_PDF, "doc1", 1, "manual.pdf",
                              [100, 100, 300, 200], ["some page text"], {}, defer=True)

    mock_gate.assert_not_called()
    assert block["metadata"]["caption_deferred"] is True
    assert block["metadata"]["pending_vision"] is True
    assert block["metadata"]["image_path"] == f"/images/doc1/{block['block_id']}.png"
    assert block["text"] == "[figure]"


def test_defer_reason_scanned_no_text_keeps_placeholder_text():
    with patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepngbytes"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        block = _figure_block(_FAKE_PDF, "doc1", 1, "manual.pdf",
                              [100, 100, 300, 200], [], {}, defer=True,
                              defer_reason="scanned_no_text")

    mock_gate.assert_not_called()
    assert block["metadata"]["defer_reason"] == "scanned_no_text"
    assert block["text"] == "[figure]"


def test_defer_reason_large_document_lazy_gets_useful_placeholder_text():
    # Real design, 3-Aug: these are NEVER auto-resolved during ingestion (see
    # caption_deferred_figures), so they need real, search-findable text now
    # instead of a bare "[figure]" that would sit useless forever.
    with patch("backend.vision.pdf_cropper.PDFCropper.crop_region",
               return_value=b"fakepngbytes"), \
         patch("backend.extraction.vision_ocr.classify_caption_crop") as mock_gate:
        block = _figure_block(_FAKE_PDF, "doc1", 1, "manual.pdf",
                              [100, 100, 300, 200], ["real page text"], {}, defer=True,
                              defer_reason="large_document_lazy")

    mock_gate.assert_not_called()
    assert block["metadata"]["defer_reason"] == "large_document_lazy"
    assert block["text"] != "[figure]"
    assert "view_page_image" in block["text"]
