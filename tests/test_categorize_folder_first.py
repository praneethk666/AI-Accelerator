"""Tests for two config-driven-client-customization additions to
backend/categorize/categorize_tool.py:

1. `_folder_result` -- a tier-1 resolution step, checked between the existing
   `_forced_result` global bypass and the full vision/LLM `categorize()` call.
   When deployment.corpus_root is set and the file's own folder path matches a
   known category (backend/categorize/folder_router.py, already built/tested,
   but never wired in before this), document_type/route resolve deterministically
   and the vision call is skipped entirely -- same payoff as force_document_type,
   but per-file instead of one global constant.
2. `_apply_industry_override` / deployment.industry -- a general, always-applied
   constant industry override, independent of force_document_type (the real gap
   fixed here: force_industry previously had NO effect unless force_document_type
   was ALSO set, which doesn't work for a corpus with multiple real document_types).

Real folder fixtures reused from tests/test_folder_router.py's own ROOT/path
shapes (validated there against the actual Toyoda TNGA tree), not invented here.
"""
import os
from unittest.mock import patch

from backend.categorize.categorize_tool import (
    CategorizeTool,
    _apply_industry_override,
    _folder_result,
)

ROOT = "/corpus/TNGA"


def _config(corpus_root=None, industry=None, folder_category_keywords=None,
            force_document_type=None):
    cfg = {
        "type_to_route": {"manual": "text_default", "cad_drawing": "cad_route"},
        "default_industry": "general",
    }
    dep = {}
    if corpus_root:
        dep["corpus_root"] = corpus_root
    if industry:
        dep["industry"] = industry
    if force_document_type:
        dep["force_document_type"] = force_document_type
    if dep:
        cfg["deployment"] = dep
    if folder_category_keywords is not None:
        cfg["categorization"] = {"folder_category_keywords": folder_category_keywords}
    return cfg


# ---------------------------------------------------------------------------
# _folder_result
# ---------------------------------------------------------------------------

def test_folder_result_none_when_corpus_root_unset():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    assert _folder_result(p, _config()) is None


def test_folder_result_none_when_corpus_root_is_unresolved_placeholder():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    assert _folder_result(p, _config(corpus_root="${CORPUS_ROOT}")) is None


def test_folder_result_resolves_manual_from_real_folder_scheme():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    result = _folder_result(p, _config(corpus_root=ROOT))
    assert result["document_type"] == "manual"
    assert result["route"] == "text_default"
    assert result["confidence"] == 0.9      # deterministic structural match, not 1.0 (explicit force)
    assert result["file_type"] == "pdf"


def test_folder_result_none_when_no_folder_match():
    p = os.path.join(ROOT, "SOME_MACHINE", "99_Unrecognized folder", "x.pdf")
    assert _folder_result(p, _config(corpus_root=ROOT)) is None


def test_folder_result_uses_client_extra_keywords():
    # a folder convention the built-in list has no entry for at all
    p = os.path.join(ROOT, "SOME_MACHINE", "Field Service Manual", "x.pdf")
    cfg = _config(corpus_root=ROOT, folder_category_keywords=[["field service manual", "manual"]])
    result = _folder_result(p, cfg)
    assert result["document_type"] == "manual"


# ---------------------------------------------------------------------------
# CategorizeTool.run() -- tier ordering (forced > folder > vision)
# ---------------------------------------------------------------------------

def test_run_skips_vision_when_folder_resolves():
    tool = CategorizeTool()
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    state = {"file_path": p}
    config = _config(corpus_root=ROOT)

    with patch("backend.categorize.categorize_tool.categorize") as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_not_called()
    assert result["document_type"] == "manual"
    assert result["confidence"] == 0.9


def test_run_falls_through_to_vision_when_no_folder_match():
    tool = CategorizeTool()
    p = os.path.join(ROOT, "SOME_MACHINE", "99_Unrecognized folder", "x.pdf")
    state = {"file_path": p}
    config = _config(corpus_root=ROOT)

    fake_result = {"route": "text_default", "document_type": "report",
                    "industry": "general", "confidence": 0.7, "reasoning": "x", "errors": []}
    with patch("backend.categorize.categorize_tool.categorize", return_value=fake_result) as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_called_once()
    assert result["document_type"] == "report"


def test_run_falls_through_to_vision_when_no_corpus_root_configured():
    # unchanged default behaviour for the majority of deployments (no
    # folder-structured corpus at all)
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config()  # no deployment block at all

    fake_result = {"route": "text_default", "document_type": "report",
                    "industry": "general", "confidence": 0.7, "reasoning": "x", "errors": []}
    with patch("backend.categorize.categorize_tool.categorize", return_value=fake_result) as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_called_once()
    assert result["document_type"] == "report"


def test_forced_result_takes_precedence_over_folder_result():
    # explicit force_document_type (tier 0) wins even when the folder would
    # ALSO resolve -- _forced_result is checked first in CategorizeTool.run().
    tool = CategorizeTool()
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    state = {"file_path": p}
    config = _config(corpus_root=ROOT, force_document_type="cad_drawing")

    with patch("backend.categorize.categorize_tool.categorize") as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_not_called()
    assert result["document_type"] == "cad_drawing"   # forced, not folder-derived "manual"


# ---------------------------------------------------------------------------
# deployment.industry -- independent constant, decoupled from
# force_document_type (the real gap this closes: force_industry alone
# previously had NO effect unless force_document_type was ALSO set).
# ---------------------------------------------------------------------------

def test_apply_industry_override_sets_industry_when_configured():
    result = {"industry": "general"}
    out = _apply_industry_override(result, {"deployment": {"industry": "manufacturing"}})
    assert out["industry"] == "manufacturing"


def test_apply_industry_override_noop_when_unset():
    result = {"industry": "general"}
    out = _apply_industry_override(result, {"deployment": {}})
    assert out["industry"] == "general"


def test_deployment_industry_overrides_vision_result_industry():
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config(industry="manufacturing")

    fake_result = {"route": "text_default", "document_type": "report",
                    "industry": "electronics", "confidence": 0.7, "reasoning": "x", "errors": []}
    with patch("backend.categorize.categorize_tool.categorize", return_value=fake_result):
        result = tool.run(state, config)

    assert result["industry"] == "manufacturing"   # overridden, not vision's "electronics"


def test_deployment_industry_works_without_force_document_type():
    # the actual bug: deployment.industry must take effect on its own, without
    # also requiring force_document_type to be set (impossible for a corpus
    # that genuinely mixes manuals/CAD/parts lists under one deployment).
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config(industry="manufacturing")  # no force_document_type

    fake_result = {"route": "text_default", "document_type": "report",
                    "industry": "general", "confidence": 0.7, "reasoning": "x", "errors": []}
    with patch("backend.categorize.categorize_tool.categorize", return_value=fake_result) as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_called_once()   # normal classification still ran (no force)
    assert result["industry"] == "manufacturing"   # but industry is still pinned


def test_deployment_industry_applies_on_top_of_forced_result():
    tool = CategorizeTool()
    state = {"file_path": "manual.pdf"}
    config = _config(industry="manufacturing", force_document_type="manual")

    with patch("backend.categorize.categorize_tool.categorize") as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_not_called()
    assert result["document_type"] == "manual"
    assert result["industry"] == "manufacturing"


def test_deployment_industry_applies_on_top_of_folder_result():
    tool = CategorizeTool()
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "x.pdf")
    state = {"file_path": p}
    config = _config(corpus_root=ROOT, industry="manufacturing")

    with patch("backend.categorize.categorize_tool.categorize") as mock_categorize:
        result = tool.run(state, config)

    mock_categorize.assert_not_called()
    assert result["document_type"] == "manual"
    assert result["industry"] == "manufacturing"
