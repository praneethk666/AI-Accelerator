"""Tests for classifier.py's _score_industry_from_text — the keyword-based industry
fallback used when vision doesn't confidently name one. Real bug found 27-Jul on a
genuine Toyoda grinding-machine manual: first-match-wins picked "automotive" off a
single generic keyword ("torque", 18 occurrences) while zero manufacturing keywords
matched at all. Fixed to score every industry by distinct keyword count and require
at least 2 distinct matches before committing to one.
"""
from backend.categorize.classifier import _score_industry_from_text


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


if __name__ == "__main__":
    test_single_generic_keyword_is_not_enough_to_pick_an_industry()
    test_two_distinct_keywords_are_enough()
    test_scores_all_industries_not_just_the_first_match()
    test_repeated_occurrences_of_one_keyword_still_only_count_once()
    test_no_matches_anywhere_returns_none()
    test_empty_text_returns_none()
    print("categorize industry-scoring tests passed")
