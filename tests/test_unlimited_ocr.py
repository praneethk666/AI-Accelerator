"""Tests for backend/extraction/unlimited_ocr.py — the det-tag parser and the
local-engine HTTP client. Real sample fixtures below are trimmed excerpts from
Unlimited-OCR's ACTUAL output on the real servo manual (24-Jul GPU bake-off,
pages 50 and 61), not synthetic examples, so the rowspan-denormalization test
covers the exact shape this project's real data produces.
"""
from unittest.mock import MagicMock, patch

import httpx

from backend.extraction.unlimited_ocr import det_to_blocks, transcribe_page_local, transcribe_table_local
from backend.extraction.vision_ocr import transcribe_page_blocks


# ── det_to_blocks: type mapping ─────────────────────────────────────────────────

def test_header_and_page_number_are_dropped_as_page_furniture():
    raw = (
        "<|det|>header [812, 9, 938, 38]<|/det|>TOYODA"
        "<|det|>text [157, 112, 728, 126]<|/det|>Real content line."
        "<|det|>page_number [527, 951, 556, 965]<|/det|>5-2"
    )
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert [b["type"] for b in blocks] == ["text"]
    assert blocks[0]["text"] == "Real content line."


def test_image_marker_is_dropped_figure_handling_is_a_separate_pipeline():
    raw = "<|det|>image [331, 109, 457, 162]<|/det|><|det|>text [1,1,2,2]<|/det|>hi"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert [b["type"] for b in blocks] == ["text"]


def test_title_maps_to_heading():
    raw = "<|det|>title [98, 86, 281, 105]<|/det|>5.1 Alarm List"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert blocks[0]["type"] == "heading"
    assert blocks[0]["text"] == "5.1 Alarm List"


def test_unknown_type_falls_back_to_text_fail_safe():
    raw = "<|det|>caption [1,2,3,4]<|/det|>Some unrecognized tag type."
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert blocks[0]["type"] == "text"


def test_empty_content_between_tags_is_dropped():
    raw = "<|det|>text [1,2,3,4]<|/det|>   <|det|>text [5,6,7,8]<|/det|>real"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert len(blocks) == 1
    assert blocks[0]["text"] == "real"


def test_det_bbox_is_kept_in_metadata_not_source_ref():
    raw = "<|det|>text [157, 112, 728, 126]<|/det|>content"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    b = blocks[0]
    assert b["source_ref"]["bbox"] is None  # unverified coordinate space -> not trusted
    assert b["metadata"]["det_bbox_raw"] == [157.0, 112.0, 728.0, 126.0]


def test_malformed_bbox_does_not_crash_and_content_is_kept():
    raw = "<|det|>text [not, numbers]<|/det|>still real content"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert blocks[0]["text"] == "still real content"
    assert blocks[0]["metadata"]["det_bbox_raw"] is None


# ── det_to_blocks: table with rowspan (real p050 excerpt, trimmed) ─────────────

_P050_TABLE_EXCERPT = (
    "<|det|>table [108, 288, 913, 868]<|/det|>"
    "<table><tr><td>Alarm name</td><td>Code</td><td>Alarm details, detecting and "
    "clearing method</td><td>Indication</td><td>Remarks</td></tr>"
    "<tr><td rowspan=\"3\">Power element error</td><td rowspan=\"3\">11H</td>"
    "<td>Power module error</td><td rowspan=\"3\"><img></td><td>SU</td></tr>"
    "<tr><td>Detect power module error before powering on main circuit power "
    "supply.</td><td>DB</td></tr>"
    "<tr><td>Clear with alarm clear command.</td><td>During servo-offNC-reset</td></tr>"
    "</table>"
)


def test_rowspan_cells_are_denormalized_into_every_row_they_cover():
    blocks = det_to_blocks(_P050_TABLE_EXCERPT, "doc1", 50, "manual.pdf")
    assert len(blocks) == 1
    tbl = blocks[0]
    assert tbl["type"] == "table"
    data = tbl["table_data"]
    assert data["headers"] == [
        "Alarm name", "Code", "Alarm details, detecting and clearing method",
        "Indication", "Remarks",
    ]
    assert len(data["rows"]) == 3
    # every row is self-contained: "Power element error" / "11H" repeated down all 3
    # rows, not left blank after the first — this is the whole point of the parser.
    for row in data["rows"]:
        assert row[0] == "Power element error"
        assert row[1] == "11H"
    assert data["rows"][0][2] == "Power module error"
    assert data["rows"][0][4] == "SU"
    assert data["rows"][1][2] == "Detect power module error before powering on main circuit power supply."
    assert data["rows"][1][4] == "DB"
    assert data["rows"][2][4] == "During servo-offNC-reset"


def test_rowspan_img_placeholder_cell_denormalizes_too_without_crashing():
    blocks = det_to_blocks(_P050_TABLE_EXCERPT, "doc1", 50, "manual.pdf")
    data = blocks[0]["table_data"]
    # the rowspan="3"><img></td> cell has no text -> denormalizes to "" every row,
    # not an exception and not a misaligned column.
    assert all(row[3] == "" for row in data["rows"])


# ── det_to_blocks: MULTIPLE rowspan columns + a later group (real p059 excerpt,
# trimmed) — the bug found live 27-Jul that _P050_TABLE_EXCERPT above didn't catch
# because it only has ONE spanning column (Code) whose value differs per group; this
# real alarm-code table also spans the Indication/Remarks columns AND has
# continuation rows with only ONE new <td> (just the details cell) — leaving those
# trailing spanning columns un-visited by a naive column-skip for that row, so their
# pending carry-down survived stale into the NEXT alarm group once that group's own
# column also carried a genuine rowspan. ─────────────────────────────────────────

_P059_TWO_GROUP_EXCERPT = (
    "<|det|>table [93, 76, 915, 699]<|/det|>"
    "<table><tr><td>Alarm name</td><td>Code</td><td>Alarm details, detecting and "
    "clearing method</td><td>Indication</td><td>Remarks</td></tr>"
    "<tr><td rowspan=\"3\">Motor model setting error</td><td rowspan=\"3\">F7H</td>"
    "<td>Motor model setting inconsistency detected.</td><td rowspan=\"3\"><img></td>"
    "<td rowspan=\"3\">SUDBAt initializationRe-power on</td></tr>"
    "<tr><td>Detect error when motor code being set is not the one allowable for "
    "the amplifier you use.</td></tr>"
    "<tr><td>Clear by re-powering on.</td></tr>"
    "<tr><td rowspan=\"4\">Parameter error 1</td><td rowspan=\"4\">F8H</td>"
    "<td>System parameter error detected.</td><td rowspan=\"4\"><img></td>"
    "<td rowspan=\"4\">SUDBAlwaysRe-power on</td></tr>"
    "<tr><td>Detect error when system parameter re-written, any errors in data "
    "settings.</td></tr>"
    "<tr><td>Clear by re-powering on.</td></tr>"
    "<tr><td>This error occurs when writing parameters into new amplifier.This is "
    "not abnormal.</td></tr>"
    "</table>"
)


def test_later_group_does_not_inherit_an_earlier_groups_stale_rowspan_value():
    # Real bug: once a continuation row has FEWER <td>s than actively-spanning
    # columns (here: only the details cell, while Indication+Remarks are ALSO under
    # rowspan), a naive column-skip never visits/decrements those trailing columns
    # for that row -- so their pending value survived stale into the NEXT group,
    # which got fed "Motor model setting error"'s own leftover Indication/Remarks
    # values instead of its own once the column indices happened to realign.
    blocks = det_to_blocks(_P059_TWO_GROUP_EXCERPT, "doc1", 59, "manual.pdf")
    data = blocks[0]["table_data"]
    assert data["headers"] == [
        "Alarm name", "Code", "Alarm details, detecting and clearing method",
        "Indication", "Remarks",
    ]
    assert len(data["rows"]) == 7  # 3 rows for group 1, 4 for group 2
    for row in data["rows"]:
        assert len(row) == 5  # never drifts wider than the real header count

    group1 = data["rows"][:3]
    for row in group1:
        assert row[0] == "Motor model setting error"
        assert row[1] == "F7H"
        assert row[4] == "SUDBAt initializationRe-power on"

    group2 = data["rows"][3:]
    for row in group2:
        assert row[0] == "Parameter error 1"
        assert row[1] == "F8H"
        # the real bug: this used to come out as group 1's leftover value instead
        assert row[4] == "SUDBAlwaysRe-power on"


# ── det_to_blocks: simple (non-rowspan) table, real p061 excerpt ───────────────

def test_simple_table_without_spans_parses_headers_and_rows():
    raw = (
        "<|det|>table [130, 217, 890, 311]<|/det|>"
        "<table><tr><td></td><td>Factor 1</td><td>Factor 2</td></tr>"
        "<tr><td>Occurred by turning on control power supply.</td><td>○</td>"
        "<td></td></tr></table>"
    )
    blocks = det_to_blocks(raw, "doc1", 61, "manual.pdf")
    data = blocks[0]["table_data"]
    assert data["headers"] == ["", "Factor 1", "Factor 2"]
    assert data["rows"] == [["Occurred by turning on control power supply.", "○", ""]]


def test_unparseable_table_markup_is_dropped_not_emitted_as_raw_html():
    raw = "<|det|>table [1,2,3,4]<|/det|><table></table>"
    blocks = det_to_blocks(raw, "doc1", 50, "manual.pdf")
    assert blocks == []


# ── transcribe_page_local ────────────────────────────────────────────────────────

def test_missing_endpoint_raises_before_making_any_network_call():
    try:
        transcribe_page_local(MagicMock(), {"vision_ocr": {"engine": "local"}})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "local_endpoint" in str(e)


def test_successful_call_sends_api_key_header_and_returns_text():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"text": "<|det|>text [1,2,3,4]<|/det|>hi"}
    fake_resp.raise_for_status.return_value = None

    with patch("backend.extraction.orientation.upright_png", return_value=b"png-bytes"), \
         patch("httpx.post", return_value=fake_resp) as mock_post:
        result = transcribe_page_local(MagicMock(), {
            "vision_ocr": {
                "engine": "local",
                "local_endpoint": "http://gpu-box/infer",
                "local_api_key": "secret123",
            }
        })

    assert result == "<|det|>text [1,2,3,4]<|/det|>hi"
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-API-Key"] == "secret123"
    assert kwargs["files"]["image"][2] == "image/png"


def test_call_without_api_key_configured_sends_no_auth_header():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"text": "ok"}
    fake_resp.raise_for_status.return_value = None

    with patch("backend.extraction.orientation.upright_png", return_value=b"png-bytes"), \
         patch("httpx.post", return_value=fake_resp) as mock_post:
        transcribe_page_local(MagicMock(), {
            "vision_ocr": {"engine": "local", "local_endpoint": "http://gpu-box/infer"}
        })

    _, kwargs = mock_post.call_args
    assert "X-API-Key" not in kwargs["headers"]


# ── transcribe_page_blocks: engine dispatch ─────────────────────────────────────

def test_default_engine_is_vlm_and_uses_markdown_path():
    with patch("backend.extraction.vision_ocr.transcribe_page",
               return_value="plain text") as mock_vlm, \
         patch("backend.extraction.unlimited_ocr.transcribe_page_local") as mock_local:
        blocks = transcribe_page_blocks(MagicMock(), {}, "doc1", 5, "manual.pdf")

    mock_vlm.assert_called_once()
    mock_local.assert_not_called()
    assert blocks[0]["text"] == "plain text"


def test_engine_local_routes_to_unlimited_ocr_and_never_touches_the_vlm():
    cfg = {"vision_ocr": {"engine": "local"}}
    with patch("backend.extraction.vision_ocr.transcribe_page") as mock_vlm, \
         patch("backend.extraction.unlimited_ocr.transcribe_page_local",
               return_value="<|det|>text [1,2,3,4]<|/det|>hi") as mock_local:
        blocks = transcribe_page_blocks(MagicMock(), cfg, "doc1", 5, "manual.pdf")

    mock_local.assert_called_once()
    mock_vlm.assert_not_called()
    assert blocks[0]["text"] == "hi"


# ── transcribe_table_local (per-table-crop escalation path) ────────────────────

def test_transcribe_table_local_extracts_just_the_table_and_denormalizes_rowspan():
    raw = (
        "<|det|>header [1,1,2,2]<|/det|>TOYODA"
        "<|det|>table [1,2,3,4]<|/det|>"
        "<table><tr><td>Code</td><td>Name</td></tr>"
        "<tr><td rowspan=\"2\">F7H</td><td>Motor error</td></tr>"
        "<tr><td>Detect error.</td></tr></table>"
        "<|det|>page_number [1,1,2,2]<|/det|>5-1"
    )
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"text": raw}
    fake_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = transcribe_table_local(b"crop-bytes", {
            "vision_ocr": {"local_endpoint": "http://gpu-box/infer", "local_api_key": "k"}
        })

    assert result["headers"] == ["Code", "Name"]
    assert result["rows"] == [["F7H", "Motor error"], ["F7H", "Detect error."]]
    # header/page_number furniture around the table must not leak into the result
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["engine"] == "unlimited_ocr"


def test_transcribe_table_local_retries_with_crop_mode_false_on_cuda_oom():
    # Real fix, 27-Jul: ocr_server.py returns a distinguishable 503 specifically for
    # a CUDA-OOM-shaped failure (large/dense crop under crop_mode=True's patch-tiling
    # path) so the client can retry with a less memory-hungry codepath instead of
    # immediately giving up to the paid VLM fallback.
    oom_resp = MagicMock()
    oom_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=MagicMock(status_code=503))

    good_raw = (
        "<|det|>table [1,2,3,4]<|/det|>"
        "<table><tr><td>Code</td><td>Name</td></tr>"
        "<tr><td>F7H</td><td>Motor error</td></tr></table>"
    )
    good_resp = MagicMock()
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"text": good_raw}

    with patch("httpx.post", side_effect=[oom_resp, good_resp]) as mock_post:
        result = transcribe_table_local(b"crop-bytes", {
            "vision_ocr": {"local_endpoint": "http://gpu-box/infer", "local_api_key": "k"}
        })

    assert result["headers"] == ["Code", "Name"]
    assert mock_post.call_count == 2
    _, retry_kwargs = mock_post.call_args  # the retry (second/last) call
    assert retry_kwargs["data"]["crop_mode"] is False


def test_transcribe_table_local_reraises_non_oom_http_errors_without_retry():
    err_resp = MagicMock()
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock(status_code=500))

    with patch("httpx.post", return_value=err_resp) as mock_post:
        raised = False
        try:
            transcribe_table_local(b"crop-bytes", {"vision_ocr": {"local_endpoint": "http://gpu-box/infer"}})
        except httpx.HTTPStatusError:
            raised = True
    assert raised
    mock_post.assert_called_once()  # a non-OOM error must NOT trigger a retry


def test_transcribe_table_local_returns_none_when_no_table_tag_present():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"text": "<|det|>text [1,2,3,4]<|/det|>not a table"}
    fake_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=fake_resp):
        result = transcribe_table_local(b"crop-bytes", {
            "vision_ocr": {"local_endpoint": "http://gpu-box/infer"}
        })

    assert result is None
