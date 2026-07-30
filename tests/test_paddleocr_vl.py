"""Tests for backend/extraction/paddleocr_vl.py — the PaddleOCR-VL-1.6 local
table-OCR client. Real sample fixture below is a trimmed excerpt of
PaddleOCR-VL's ACTUAL output on the real servo manual's page-2 safety table
(27-Jul GPU bake-off), not a synthetic example — confirms the shared
_parse_html_table (built for Unlimited-OCR's HTML) handles PaddleOCR-VL's HTML
shape unchanged, including the inline style= attributes it adds that
Unlimited-OCR's own HTML doesn't have.
"""
from unittest.mock import patch

from backend.extraction.paddleocr_vl import transcribe_table_paddleocr_vl


_REAL_PADDLEOCR_VL_TABLE_HTML = (
    "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    "<tr><td colspan=\"2\">Safety precautions</td><td colspan=\"2\">Symbols</td></tr>"
    "<tr><td rowspan=\"2\">Danger</td>"
    "<td rowspan=\"2\">Indicates an imminently hazardous situation which, if "
    "incorrectly operated, will result in death or serious injury.</td>"
    "<td style='text-align: center;'><img src=\"imgs/x.jpg\" alt=\"Image\" /></td>"
    "<td style='text-align: center;'>Danger, injury</td></tr>"
    "<tr><td style='text-align: center;'><img src=\"imgs/y.jpg\" alt=\"Image\" /></td>"
    "<td style='text-align: center;'>Electrical shock</td></tr>"
    "</table>"
)


def test_transcribe_table_paddleocr_vl_parses_real_response_html():
    with patch("backend.extraction.paddleocr_vl._call_paddleocr_vl_server",
               return_value=_REAL_PADDLEOCR_VL_TABLE_HTML) as mock_call:
        result = transcribe_table_paddleocr_vl(b"cropped-png-bytes", {"vision_ocr": {}})

    mock_call.assert_called_once_with(b"cropped-png-bytes", {})
    assert result["headers"] == ["Safety precautions", "Safety precautions", "Symbols", "Symbols"]
    assert result["rows"] == [
        ["Danger", "Indicates an imminently hazardous situation which, if incorrectly "
                    "operated, will result in death or serious injury.", "", "Danger, injury"],
        ["Danger", "Indicates an imminently hazardous situation which, if incorrectly "
                    "operated, will result in death or serious injury.", "", "Electrical shock"],
    ]


def test_transcribe_table_paddleocr_vl_returns_none_for_unparseable_response():
    with patch("backend.extraction.paddleocr_vl._call_paddleocr_vl_server",
               return_value="no table markup here, just prose"):
        result = transcribe_table_paddleocr_vl(b"cropped-png-bytes", {"vision_ocr": {}})

    assert result is None


def test_transcribe_table_paddleocr_vl_passes_vision_ocr_config_through():
    cfg = {"vision_ocr": {"local_paddleocr_vl_endpoint": "http://gpu-box:8084/infer",
                           "local_timeout_s": 60}}
    with patch("backend.extraction.paddleocr_vl._call_paddleocr_vl_server",
               return_value=_REAL_PADDLEOCR_VL_TABLE_HTML) as mock_call:
        transcribe_table_paddleocr_vl(b"png", cfg)

    mock_call.assert_called_once_with(
        b"png", {"local_paddleocr_vl_endpoint": "http://gpu-box:8084/infer",
                 "local_timeout_s": 60})


# ── _call_paddleocr_vl_server: real HTTP client behavior ────────────────────────

def test_call_paddleocr_vl_server_raises_when_endpoint_not_configured():
    from backend.extraction.paddleocr_vl import _call_paddleocr_vl_server
    try:
        _call_paddleocr_vl_server(b"png", {})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_call_paddleocr_vl_server_sends_api_key_header_when_configured():
    from unittest.mock import MagicMock
    from backend.extraction.paddleocr_vl import _call_paddleocr_vl_server
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"text": "<table><tr><td>x</td></tr></table>"}
    cfg = {"local_paddleocr_vl_endpoint": "http://gpu-box:8084/infer",
           "local_paddleocr_vl_api_key": "secret123"}
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        _call_paddleocr_vl_server(b"png", cfg)

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"X-API-Key": "secret123"}


if __name__ == "__main__":
    test_transcribe_table_paddleocr_vl_parses_real_response_html()
    test_transcribe_table_paddleocr_vl_returns_none_for_unparseable_response()
    test_transcribe_table_paddleocr_vl_passes_vision_ocr_config_through()
    test_call_paddleocr_vl_server_raises_when_endpoint_not_configured()
    test_call_paddleocr_vl_server_sends_api_key_header_when_configured()
    print("paddleocr_vl tests passed")
