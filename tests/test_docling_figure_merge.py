"""_merge_small_repeated_icons tests (no infra). Run: pytest tests/test_docling_figure_merge.py

Validated live (24-Jul) on a real Toyota manual's safety-warnings pages: the VLM gate
correctly classifies small hazard icons (diamond/triangle warning symbols) as real
figures, not page furniture — but a page can carry 5-9 of them, each becoming its own
near-duplicate image_caption chunk. This collapses same-page tiny icon clusters into
one combined chunk instead of dropping or leaving them fragmented.
"""
from backend.extraction.docling_pdf.docling_extract import _merge_small_repeated_icons


def _icon_block(page, text, w=30.0, h=28.0, x=50.0, y=50.0):
    return {
        "type": "image_caption",
        "text": text,
        "source_ref": {"page": page, "bbox": [x, y, x + w, y + h]},
        "metadata": {"figure_kind": "illustration"},
    }


def test_small_icon_cluster_merges_into_one_chunk():
    blocks = [_icon_block(1, f"warning icon {i}") for i in range(5)]
    out = _merge_small_repeated_icons(blocks)
    assert len(out) == 1
    assert out[0]["metadata"]["merged_icon_count"] == 5
    for i in range(5):
        assert f"warning icon {i}" in out[0]["text"]


def test_below_min_group_threshold_stays_separate():
    blocks = [_icon_block(1, "icon a"), _icon_block(1, "icon b")]
    out = _merge_small_repeated_icons(blocks, min_group=3)
    assert len(out) == 2  # only 2 tiny icons on the page -> not merged


def test_large_figure_never_merged_even_in_a_cluster():
    blocks = [_icon_block(1, f"icon {i}") for i in range(4)]
    blocks.append(_icon_block(1, "real diagram", w=400.0, h=300.0))
    out = _merge_small_repeated_icons(blocks)
    # the 4 tiny icons merge into 1; the large diagram is untouched
    assert len(out) == 2
    large = next(b for b in out if b["text"] == "real diagram")
    assert "merged_icon_count" not in (large.get("metadata") or {})


def test_different_pages_not_merged_together():
    blocks = [_icon_block(1, "p1 icon a"), _icon_block(1, "p1 icon b"), _icon_block(1, "p1 icon c"),
              _icon_block(2, "p2 icon a"), _icon_block(2, "p2 icon b"), _icon_block(2, "p2 icon c")]
    out = _merge_small_repeated_icons(blocks)
    assert len(out) == 2
    pages = {b["source_ref"]["page"] for b in out}
    assert pages == {1, 2}


def test_non_image_caption_blocks_untouched():
    blocks = [{"type": "text", "text": "hello", "source_ref": {"page": 1}}]
    out = _merge_small_repeated_icons(blocks)
    assert out == blocks


def test_missing_bbox_never_merged():
    blocks = [{"type": "image_caption", "text": f"icon {i}", "source_ref": {"page": 1}} for i in range(5)]
    out = _merge_small_repeated_icons(blocks)
    assert len(out) == 5  # no bbox -> can't size-check -> left alone
