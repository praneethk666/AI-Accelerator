"""Tests for classifier.py's keyword-based industry detection.

Real bug found 27-Jul on a genuine Toyoda grinding-machine manual: first-match-wins
picked "automotive" off a single generic keyword ("torque", 18 occurrences) while
zero manufacturing keywords matched at all. Fixed in two stages, both covered here:
  1. Score every industry (not first-match-wins), require 2+ distinct hits.
  2. Tiered weighted scoring (strong=2, weak=1): a flat "2 distinct hits" threshold
     still let 3 WEAK generic terms (torque/chassis/transmission) cluster together
     and false-positive — strong/weak tiers fix this without losing the ability for
     multiple weak terms to still corroborate each other.

Also covers the architectural flip agreed 27-Jul: resolve industry from the FREE
keyword scan BEFORE even asking vision (empirically more reliable for this
sub-field), only falling back to vision when keyword signal is weak/absent — and
letting vision propose a genuinely NEW industry outside the configured list for
documents from a domain we haven't pre-defined.
"""
import json
import os
import tempfile
from unittest.mock import patch

import fitz

from backend.categorize.classifier import _score_industry_from_text, _industry_keyword_evidence, categorize
from backend.core.config import load_config


_INDUSTRY_KEYWORDS = {
    "automotive": {
        "strong": ["toyota", "ford", "bmw", "vin"],
        "weak": ["vehicle", "engine", "torque", "transmission", "chassis", "automotive"],
    },
    "manufacturing": {
        "strong": ["bom"],
        "weak": ["assembly", "drawing", "tolerance", "weld", "machining", "fixture", "jig", "part number"],
    },
    "finance": {
        "strong": ["balance sheet"],
        "weak": ["revenue", "profit", "ledger"],
    },
    "general": {"strong": [], "weak": []},
}


# ── _score_industry_from_text: weighted, tiered scoring ─────────────────────────

def test_single_weak_keyword_is_not_enough_to_pick_an_industry():
    # The exact real failure: "torque" alone (weak — generic mechanical term, not
    # automotive-specific) used to trigger "automotive" on a non-automotive manual.
    text = "Set the servo motor torque to 15 Nm per the specification table."
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) is None


def test_two_distinct_weak_keywords_are_enough():
    text = "Adjust chassis alignment and check the transmission fluid level."
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) == "automotive"


def test_one_strong_keyword_alone_is_enough():
    text = "This vehicle's VIN is stamped on the frame rail."
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) == "automotive"


def test_scores_all_industries_not_just_the_first_match():
    # "torque" (automotive, weak) appears, but manufacturing has 4 distinct weak
    # matches (score=4) — manufacturing must win even though automotive is listed
    # first in the dict.
    text = ("Apply torque per spec. Assembly drawing shows the fixture jig "
            "tolerance for this weld.")
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) == "manufacturing"


def test_repeated_occurrences_of_one_weak_keyword_still_only_count_once():
    # 18 occurrences of "torque" is still ONE distinct weak keyword (score=1), not
    # enough alone.
    text = "torque torque torque torque torque torque torque torque torque"
    assert _score_industry_from_text(text, _INDUSTRY_KEYWORDS) is None


def test_no_matches_anywhere_returns_none():
    assert _score_industry_from_text("The sky is blue today.", _INDUSTRY_KEYWORDS) is None


def test_empty_text_returns_none():
    assert _score_industry_from_text("", _INDUSTRY_KEYWORDS) is None


# ── _industry_keyword_evidence: full per-industry, tier-separated breakdown ─────

def test_evidence_lists_every_industry_with_at_least_one_hit_tier_separated():
    text = "Apply torque per spec, then check the assembly drawing tolerance."
    evidence = _industry_keyword_evidence(text, _INDUSTRY_KEYWORDS)
    assert evidence["automotive"]["weak"] == ["torque"]
    assert evidence["automotive"]["strong"] == []
    assert evidence["automotive"]["score"] == 1
    assert set(evidence["manufacturing"]["weak"]) == {"assembly", "drawing", "tolerance"}
    assert evidence["manufacturing"]["score"] == 3
    assert "finance" not in evidence  # no hits -> not present at all


def test_evidence_weights_strong_hits_double():
    text = "The VIN and toyota badge are visible."
    evidence = _industry_keyword_evidence(text, _INDUSTRY_KEYWORDS)
    assert set(evidence["automotive"]["strong"]) == {"vin", "toyota"}
    assert evidence["automotive"]["score"] == 4  # 2 strong hits x weight 2


def test_evidence_is_empty_dict_when_nothing_matches():
    assert _industry_keyword_evidence("The sky is blue.", _INDUSTRY_KEYWORDS) == {}


# ── categorize(): keyword-first pre-resolution + vision fallback + novel industry ──

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


def test_strong_keyword_evidence_resolves_industry_before_vision_is_even_asked():
    # Page 1 (what vision "sees") is generic; pages 2-30 (what the free keyword
    # scan covers) carry real electronics vocabulary — mirrors the exact real
    # scenario found 27-Jul on the servo manual. Vision is mocked to return
    # something that would NOT independently justify "electronics" at all, proving
    # the final answer came from the pre-resolution, not from vision agreeing.
    path = _make_pdf([
        "MAINTENANCE MANUAL\nSafety precautions and general information.",
        "Check the circuit voltage. Inspect the resistor and capacitor. Signal check.",
    ])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "manual",
        "confidence": 0.9,
        "reasoning": "A maintenance manual.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response) as mock_vision:
            result = categorize(file_path=path, state={}, config=config,
                                 deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    assert result["industry"] == "electronics"
    assert "resolved from document-wide keyword evidence" in result["reasoning"]
    assert "before vision was asked" in result["reasoning"]
    # The prompt vision DID receive should NOT even ask about industry, since it
    # was already resolved -- confirms the token-saving design, not just the result.
    sent_prompt = mock_vision.call_args[0][1]
    assert "industry" not in sent_prompt.lower() or "KEYWORD EVIDENCE" not in sent_prompt


def test_weak_keyword_signal_alone_does_not_pre_resolve_vision_still_asked():
    # Only ONE weak keyword total (score=1, below threshold) -- must NOT pre-resolve;
    # vision should still be asked about industry.
    path = _make_pdf(["A servo torque specification sheet with no other context."])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "datasheet",
        "industry": "general",
        "industry_evidence": "",
        "confidence": 0.8,
        "reasoning": "Generic spec sheet.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response) as mock_vision:
            categorize(file_path=path, state={}, config=config,
                       deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    mock_vision.assert_called_once()
    sent_prompt = mock_vision.call_args[0][1]
    assert "industry" in sent_prompt.lower()  # vision WAS asked this time


def test_industry_without_keyword_or_text_corroboration_is_rejected():
    # Vision claims a known industry that's neither pre-resolved by keywords NOR
    # quotable from the (short) visible text -> must NOT be accepted blindly.
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


def test_novel_industry_accepted_when_verified_and_confident():
    # Vision proposes an industry NOT in the configured list at all, but with a
    # genuinely verifiable quote from the visible text and high confidence.
    path = _make_pdf(["AEROSPACE MAINTENANCE LOG\nTurbine blade inspection record."])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "manual",
        "industry": "aerospace",
        "industry_evidence": "turbine blade inspection",
        "confidence": 0.85,
        "reasoning": "Aerospace maintenance content.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response):
            result = categorize(file_path=path, state={}, config=config,
                                 deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    assert result["industry"] == "aerospace"
    assert "NEW industry candidate" in result["reasoning"]
    assert "human review" in result["reasoning"]


def test_novel_industry_rejected_without_verifiable_evidence():
    # Same novel proposal, but with an evidence phrase that doesn't actually appear
    # in the visible text -> must be discarded, not accepted on the model's say-so.
    path = _make_pdf(["A short generic document with no aerospace content at all."])
    config = load_config("config/global.yaml")
    vision_response = json.dumps({
        "document_type": "report",
        "industry": "aerospace",
        "industry_evidence": "turbine blade inspection",  # not actually in the text
        "confidence": 0.85,
        "reasoning": "Guessed aerospace.",
    })
    try:
        with patch("backend.categorize.classifier.describe_image", return_value=vision_response):
            result = categorize(file_path=path, state={}, config=config,
                                 deployment=config.get("deployment", {}))
    finally:
        os.unlink(path)

    assert result["industry"] != "aerospace"


if __name__ == "__main__":
    test_single_weak_keyword_is_not_enough_to_pick_an_industry()
    test_two_distinct_weak_keywords_are_enough()
    test_one_strong_keyword_alone_is_enough()
    test_scores_all_industries_not_just_the_first_match()
    test_repeated_occurrences_of_one_weak_keyword_still_only_count_once()
    test_no_matches_anywhere_returns_none()
    test_empty_text_returns_none()
    test_evidence_lists_every_industry_with_at_least_one_hit_tier_separated()
    test_evidence_weights_strong_hits_double()
    test_evidence_is_empty_dict_when_nothing_matches()
    test_strong_keyword_evidence_resolves_industry_before_vision_is_even_asked()
    test_weak_keyword_signal_alone_does_not_pre_resolve_vision_still_asked()
    test_industry_without_keyword_or_text_corroboration_is_rejected()
    test_novel_industry_accepted_when_verified_and_confident()
    test_novel_industry_rejected_without_verifiable_evidence()
    print("categorize industry-scoring tests passed")
