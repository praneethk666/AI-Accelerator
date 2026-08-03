"""Tests for the deployment.force_document_type bypass
(backend/categorize/categorize_tool.py) — lets a narrow-scope deployment (a
customer whose documents are always one known type) skip the categorize
LLM/vision call entirely, since classification would land on the same answer
every time anyway.
"""
from unittest.mock import patch

from backend.categorize.categorize_tool import CategorizeTool, _forced_result


def _config(force_document_type=None, force_industry=None, force_route=None):
    cfg = {
        "type_to_route": {"manual": "text_default", "cad_drawing": "cad_route"},
        "default_industry": "general",
    }
    dep = {}
    if force_document_type:
        dep["force_document_type"] = force_document_type
    if force_industry:
        dep["force_industry"] = force_industry
    if force_route:
        dep["force_route"] = force_route
    if dep:
        cfg["deployment"] = dep
    return cfg


def test_forced_result_none_when_not_configured():
    assert _forced_result("manual.pdf", _config()) is None


def test_forced_result_uses_type_to_route_when_force_route_unset():
    result = _forced_result("manual.pdf", _config(force_document_type="cad_drawing"))
    assert result["document_type"] == "cad_drawing"
    assert result["route"] == "cad_route"
    assert result["industry"] == "general"  # falls back to default_industry
    assert result["confidence"] == 1.0
    assert result["file_type"] == "pdf"


def test_forced_result_explicit_route_and_industry_override_defaults():
    result = _forced_result(
        "manual.pdf",
        _config(force_document_type="manual", force_industry="automotive", force_route="diagram_heavy"),
    )
    assert result["route"] == "diagram_heavy"
    assert result["industry"] == "automotive"


def test_run_skips_classification_entirely_when_forced():
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config(force_document_type="manual", force_industry="automotive")

    with patch("backend.categorize.categorize_tool.categorize") as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_not_called()
    assert result["document_type"] == "manual"
    assert result["industry"] == "automotive"
    assert result["confidence"] == 1.0


def test_run_calls_real_categorize_when_not_forced():
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config()  # no deployment override

    fake_result = {"route": "text_default", "document_type": "report",
                    "industry": "general", "confidence": 0.7, "reasoning": "x", "errors": []}
    with patch("backend.categorize.categorize_tool.categorize", return_value=fake_result) as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_called_once()
    assert result["document_type"] == "report"


if __name__ == "__main__":
    test_forced_result_none_when_not_configured()
    test_forced_result_uses_type_to_route_when_force_route_unset()
    test_forced_result_explicit_route_and_industry_override_defaults()
    test_run_skips_classification_entirely_when_forced()
    test_run_calls_real_categorize_when_not_forced()
    print("categorize force-override tests passed")
