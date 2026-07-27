"""Tests for classifier.py's _score_industry_from_text — the keyword-based industry
fallback used when vision doesn't confidently name one. Real bug found 27-Jul on a
genuine Toyoda grinding-machine manual: first-match-wins picked "automotive" off a
single generic keyword ("torque", 18 occurrences) while zero manufacturing keywords
matched at all. Fixed to score every industry by distinct keyword count and require
at least 2 distinct matches before committing to one.
"""
import json
import os
import tempfile
from unittest.mock import patch

import fitz

from backend.categorize.classifier import _score_industry_from_text, _industry_keyword_evidence, categorize
from backend.core.config import load_config


_INDUSTRY_KEYWORDS = {
    "automotive": ["toyota", "ford", "bmw", "vehicle", "engine", "torque",
                   "transmission", "chassis", "automotive"],
    "manufacturing": ["assembly", "drawing", "tolerance", "weld", "machining",
                      "fixture", "jig", "bom", "part number"],
    "finance": ["balance sheet", "revenue", "profit", "ledger"],
    "general": [],
}


def test_single_generic_keyword_is_not_enough_to_pick_an_industry():
    # The exact real failure: "torque" alone (a generic mechanical term, not
    # automotive-specific) used to trigger "automotive" on a non-automotive manual.
    text = "Set the servo motor torque to 15 Nm per the specification table."
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) is None


def test_two_distinct_keywords_are_enough():
    text = "Adjust chassis alignment and check the transmission fluid level."
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) == "automotive"


def test_scores_all_industries_not_just_the_first_match():
    # "torque" (automotive) appears, but manufacturing has 3 distinct matches —
    # manufacturing must win even though automotive is listed first in the dict.
    text = ("Apply torque per spec. Assembly drawing shows the fixture jig "
            "tolerance for this weld.")
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) == "manufacturing"


def test_repeated_occurrences_of_one_keyword_still_only_count_once():
    # 18 occurrences of "torque" is still ONE distinct keyword, not enough alone.
    text = "torque torque torque torque torque torque torque torque torque"
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) is None


def test_no_matches_anywhere_returns_none():
    assert _score_industry_from_text("The sky is blue today.", _INDUSTRY_KEYWORDS) is None


def test_empty_text_returns_none():
    assert _score_industry_from_text("", _INDUSTRY_KEYWORDS) is None


# ── _industry_keyword_evidence: full per-industry breakdown ────────────────────

def test_evidence_lists_every_industry_with_at_least_one_hit():
    text = "Apply torque per spec, then check the assembly drawing tolerance."
    evidence = _industry_keyword_evidence(text, _INDUSTRY_KEYWORDS)
    assert evidence["automotive"] == ["torque"]
    assert set(evidence["manufacturing"]) == {"assembly", "drawing", "tolerance"}
    assert "finance" not in evidence  # no hits -> not present at all


def test_evidence_is_empty_dict_when_nothing_matches():
    assert _industry_keyword_evidence("The sky is blue.", _INDUSTRY_KEYWORDS) == {}


# ── categorize(): keyword-evidence corroboration wired into the vision prompt ──

def _make_pdf(pages_text: list[str]) -> str:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


def test_industry_corroborated_by_keyword_evidence_is_accepted_even_with_weak_visual_evidence():
    # Page 1 (what vision "sees") is generic; pages 2-30 (what the free keyword
    # scan covers) carry real electronics vocabulary — mirrors the exact real
    # scenario found 27-Jul on the servo manual.
    path = _make_pdf([
        "MAINTENANCE MANUAL\nSafety precautions and general information.",
        "Check the circuit voltage. Inspect the resistor and capacitor. Signal check.",
    ])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "manual",
        "industry": "electronics",
        "industry_evidence": "",  # deliberately weak/no quotable phrase from page 1
        "confidence": 0.6,
        "reasoning": "Looks like a technical manual.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response):
            result = categorize(file_path=path, state={}, config=config,
                                 deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    assert result["industry"] == "electronics"
    assert "corroborated by document-wide keyword evidence" in result["reasoning"]


def test_industry_without_keyword_or_text_corroboration_is_rejected():
    # Vision claims an industry that's neither corroborated by the keyword scan
    # NOR quotable from the (short) visible text -> must NOT be accepted blindly.
    path = _make_pdf(["Hello world, this is a short generic document."])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "report",
        "industry": "finance",
        "industry_evidence": "some made up phrase",
        "confidence": 0.5,
        "reasoning": "Guessed finance.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response):
            result = categorize(file_path=path, state={}, config=config,
                                 deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    assert result["industry"] != "finance"


if __name__ == "__main__":
    test_single_generic_keyword_is_not_enough_to_pick_an_industry()
    test_two_distinct_keywords_are_enough()
    test_scores_all_industries_not_just_the_first_match()
    test_repeated_occurrences_of_one_keyword_still_only_count_once()
    test_no_matches_anywhere_returns_none()
    test_empty_text_returns_none()
    test_evidence_lists_every_industry_with_at_least_one_hit()
    test_evidence_is_empty_dict_when_nothing_matches()
    test_industry_corroborated_by_keyword_evidence_is_accepted_even_with_weak_visual_evidence()
    test_industry_without_keyword_or_text_corroboration_is_rejected()
    print("categorize industry-scoring tests passed")
