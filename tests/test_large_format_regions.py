"""Tests for the agentic locate-then-zoom region extraction path in large_format.py
(transcribe_large_page_regions and its helpers) — the alternative to blind grid
tiling, added after a live test found blind tiling slow and producing corrupted
bboxes on a real oversized CAD sheet."""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from PIL import Image

from backend.extraction.large_format import (
    _region_crop_box,
    _remap_bbox,
    locate_regions,
    transcribe_large_page_regions,
    transcribe_regions_blocks,
)


def _png_bytes(w=800, h=600, color=(255, 255, 255)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _fake_page(w_pt=2000.0, h_pt=1500.0, png=None):
    page = MagicMock()
    page.rect.width = w_pt
    page.rect.height = h_pt
    pix = MagicMock()
    pix.tobytes.return_value = png or _png_bytes()
    page.get_pixmap.return_value = pix
    return page


# --- _remap_bbox ---------------------------------------------------------------

def test_remap_bbox_valid_normalizes_into_full_page_coords():
    # tile occupies the right half of a 1000x1000 image; bbox is centered in the tile
    out = _remap_bbox([0.0, 0.0, 1.0, 1.0], (500, 0, 1000, 1000), (1000, 1000))
    assert out == [0.5, 0.0, 1.0, 1.0]


def test_remap_bbox_rejects_out_of_range_values():
    assert _remap_bbox([279.12, 373.30, 400.0, 500.0], (0, 0, 1000, 1000), (1000, 1000)) is None


def test_remap_bbox_rejects_bad_shape():
    assert _remap_bbox([0.1, 0.2, 0.3], (0, 0, 1000, 1000), (1000, 1000)) is None
    assert _remap_bbox(None, (0, 0, 1000, 1000), (1000, 1000)) is None
    assert _remap_bbox([], (0, 0, 1000, 1000), (1000, 1000)) is None


def test_remap_bbox_rejects_non_numeric():
    assert _remap_bbox(["a", "b", "c", "d"], (0, 0, 1000, 1000), (1000, 1000)) is None


def test_remap_bbox_rejects_degenerate_box():
    assert _remap_bbox([0.5, 0.5, 0.2, 0.9], (0, 0, 1000, 1000), (1000, 1000)) is None
    assert _remap_bbox([0.5, 0.5, 0.9, 0.5], (0, 0, 1000, 1000), (1000, 1000)) is None


# --- _region_crop_box ------------------------------------------------------------

def test_region_crop_box_adds_padding_within_bounds():
    l, t, r, b = _region_crop_box([0.4, 0.4, 0.6, 0.6], (1000, 1000), pad_frac=0.1)
    assert 0 <= l < 400
    assert 0 <= t < 400
    assert 600 < r <= 1000
    assert 600 < b <= 1000


def test_region_crop_box_clamps_at_image_edges():
    l, t, r, b = _region_crop_box([0.0, 0.0, 0.1, 0.1], (1000, 1000), pad_frac=0.5)
    assert l == 0 and t == 0
    r2, b2 = _region_crop_box([0.9, 0.9, 1.0, 1.0], (1000, 1000), pad_frac=0.5)[2:]
    assert r2 == 1000 and b2 == 1000


# --- locate_regions --------------------------------------------------------------

def test_locate_regions_parses_valid_json_array():
    reply = json.dumps([
        {"type": "table", "label": "title_block", "description": "Title block",
         "bbox": [0.7, 0.0, 1.0, 0.15]},
        {"type": "view", "label": "view_A-A", "description": "Section view",
         "bbox": [0.0, 0.0, 0.6, 0.8]},
    ])
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value=reply):
        regions = locate_regions(page, {})
    assert len(regions) == 2
    assert regions[0]["label"] == "title_block"
    assert regions[0]["bbox"] == [0.7, 0.0, 1.0, 0.15]
    assert regions[1]["type"] == "view"


def test_locate_regions_accepts_dict_with_regions_key():
    reply = json.dumps({"regions": [
        {"type": "table", "label": "parts_list", "description": "Parts",
         "bbox": [0.0, 0.2, 0.5, 0.9]},
    ]})
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value=reply):
        regions = locate_regions(page, {})
    assert len(regions) == 1
    assert regions[0]["label"] == "parts_list"


def test_locate_regions_filters_invalid_bboxes_keeps_valid():
    reply = json.dumps([
        {"type": "table", "label": "bad", "description": "x", "bbox": [279.12, 373.3, 400.0, 500.0]},
        {"type": "table", "label": "good", "description": "y", "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"type": "table", "label": "degenerate", "description": "z", "bbox": [0.5, 0.5, 0.1, 0.9]},
        {"type": "table", "label": "missing_bbox", "description": "w"},
    ])
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value=reply):
        regions = locate_regions(page, {})
    assert len(regions) == 1
    assert regions[0]["label"] == "good"


def test_locate_regions_returns_empty_on_call_failure():
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image",
               side_effect=RuntimeError("provider down")):
        regions = locate_regions(page, {})
    assert regions == []


def test_locate_regions_returns_empty_on_unparseable_reply():
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value="not json at all"):
        regions = locate_regions(page, {})
    assert regions == []


def test_locate_regions_prefers_locate_vision_config_when_set():
    # Real live finding (4-Aug): the CAD-shadowed free-tier vision model returned
    # bboxes mixing normalized and raw-pixel values on this exact coarse-locate
    # task, so locate_vision lets the ONE locate call per page use a different
    # (more reliable) vision config than the many per-region zoom calls.
    page = _fake_page()
    locate_vision_cfg = {"provider": "openai", "model": "gpt-4o-mini"}
    config = {
        "vision_ocr": {"provider": "openai", "model": "nvidia/nemotron-nano-12b-v2-vl"},
        "extraction": {"large_format": {"locate_vision": locate_vision_cfg}},
    }
    with patch("backend.extraction.large_format.describe_image", return_value="[]") as m:
        locate_regions(page, config)
    assert m.call_args[0][2] == {"vision": locate_vision_cfg}


def test_locate_regions_falls_back_to_vision_ocr_when_locate_vision_unset():
    page = _fake_page()
    vision_ocr_cfg = {"provider": "openai", "model": "nvidia/nemotron-nano-12b-v2-vl"}
    config = {"vision_ocr": vision_ocr_cfg}
    with patch("backend.extraction.large_format.describe_image", return_value="[]") as m:
        locate_regions(page, config)
    assert m.call_args[0][2] == {"vision": vision_ocr_cfg}


def test_locate_regions_caps_render_resolution_for_huge_sheet():
    # A huge sheet (E-size, ~44x34in) at default locate_dpi=100 would be >4000px;
    # locate_max_px must cap the actual render call's dpi so it fits in one image.
    page = _fake_page(w_pt=3168.0, h_pt=2448.0)  # 44in x 34in in points (72pt/in)
    with patch("backend.extraction.large_format.describe_image", return_value="[]") as m:
        locate_regions(page, {"extraction": {"large_format": {"locate_dpi": 100, "locate_max_px": 2000}}})
    called_dpi = page.get_pixmap.call_args.kwargs["dpi"]
    assert called_dpi <= 100
    w_px = 3168.0 * called_dpi / 72.0
    assert w_px <= 2000 + 1  # allow rounding


# --- transcribe_regions_blocks ----------------------------------------------------

def test_transcribe_regions_blocks_stamps_region_metadata_and_remaps_bbox():
    regions = [{"type": "table", "label": "title_block", "description": "Title block",
                "bbox": [0.0, 0.0, 0.5, 0.5]}]
    reply = json.dumps([{"type": "table", "text": "Drawing No: ABC123",
                         "bbox": [0.1, 0.1, 0.9, 0.9], "confidence": 0.9}])
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value=reply):
        blocks = transcribe_regions_blocks(page, {}, regions, "detail prompt")
    assert len(blocks) == 1
    b = blocks[0]
    assert b["metadata"]["region_label"] == "title_block"
    assert b["metadata"]["region_description"] == "Title block"
    assert b["bbox"] is not None
    assert all(0.0 <= v <= 1.0 for v in b["bbox"])


def test_transcribe_regions_blocks_skips_region_on_call_failure():
    regions = [{"type": "table", "label": "r1", "description": "d",
                "bbox": [0.0, 0.0, 0.5, 0.5]}]
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image",
               side_effect=RuntimeError("down")):
        blocks = transcribe_regions_blocks(page, {}, regions, "detail prompt")
    assert blocks == []


def test_transcribe_regions_blocks_one_bad_region_does_not_abort_others():
    regions = [
        {"type": "table", "label": "bad", "description": "d", "bbox": [0.0, 0.0, 0.5, 0.5]},
        {"type": "table", "label": "good", "description": "d", "bbox": [0.5, 0.5, 1.0, 1.0]},
    ]
    good_reply = json.dumps([{"type": "table", "text": "ok", "bbox": [0.0, 0.0, 1.0, 1.0]}])
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image",
               side_effect=[RuntimeError("down"), good_reply]):
        blocks = transcribe_regions_blocks(page, {}, regions, "detail prompt")
    assert len(blocks) == 1
    assert blocks[0]["metadata"]["region_label"] == "good"


# --- transcribe_large_page_regions (orchestrator) ---------------------------------

def test_transcribe_large_page_regions_appends_region_index_block():
    locate_reply = json.dumps([
        {"type": "table", "label": "title_block", "description": "Title block",
         "bbox": [0.7, 0.0, 1.0, 0.15]},
    ])
    detail_reply = json.dumps([{"type": "table", "text": "Drawing No: X",
                                "bbox": [0.0, 0.0, 1.0, 1.0]}])
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image",
               side_effect=[locate_reply, detail_reply]):
        blocks = transcribe_large_page_regions(page, {}, "detail prompt")
    kinds = [b.get("metadata", {}).get("kind") for b in blocks]
    assert "region_index" in kinds
    index_block = next(b for b in blocks if b["metadata"].get("kind") == "region_index")
    assert len(index_block["metadata"]["regions"]) == 1
    assert "title_block" in index_block["text"]
    # the actual transcribed content block is also present
    assert any(b.get("text") == "Drawing No: X" for b in blocks)


def test_transcribe_large_page_regions_falls_back_to_blind_tiling_when_no_regions():
    page = _fake_page()
    with patch("backend.extraction.large_format.describe_image", return_value="[]"), \
         patch("backend.extraction.large_format.transcribe_large_page_blocks",
               return_value=[{"text": "fallback block"}]) as mock_fallback:
        blocks = transcribe_large_page_regions(page, {}, "detail prompt")
    mock_fallback.assert_called_once()
    assert blocks == [{"text": "fallback block"}]


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
