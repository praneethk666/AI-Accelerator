"""Tests for backend.categorize.redaction_detect -- real finding, 3-Aug: the
spindle assembly CAD drawing's (MS03AAA789AB) own parts table has its "No. /
Parts No. / Q'ty" column values blanked out as "***" / "**-*********-*" in the
source PDF. No OCR/VLM improvement fixes this -- the data genuinely isn't there.
This detects it so it can be flagged instead of silently hallucinated or
confusingly echoed back as literal asterisks.

Run: pytest tests/test_redaction_detect.py
"""
from backend.categorize.redaction_detect import (
    is_redacted_table,
    is_redacted_text,
    tag_blocks_with_redaction,
)


def test_real_spindle_assembly_parts_table_detected():
    # exact values read off the real drawing's title-block parts table
    table = {"headers": ["No.", "Parts No.", "Q'ty"],
             "rows": [["***", "**-*********-*", "*"]]}
    assert is_redacted_table(table) is True


def test_normal_populated_table_not_flagged():
    table = {"headers": ["No.", "Parts No.", "Q'ty"],
             "rows": [["1", "MKR2002800AB", "2"], ["2", "MS03AAA789AB", "1"]]}
    assert is_redacted_table(table) is False


def test_mixed_table_below_threshold_not_flagged():
    # only 1 of 6 data cells is placeholder -- real data, not a redacted table
    table = {"headers": ["a", "b"],
             "rows": [["1", "2"], ["3", "4"], ["***", "6"]]}
    assert is_redacted_table(table, threshold=0.8) is False


def test_empty_table_not_flagged():
    assert is_redacted_table(None) is False
    assert is_redacted_table({"headers": ["a"], "rows": []}) is False
    assert is_redacted_table({"headers": ["a"], "rows": [["", ""]]}) is False


def test_markdown_bold_and_real_hyphenated_values_not_false_positives():
    # "**bold**" has real letters; a real part number w/ hyphens has real digits
    table = {"headers": ["a", "b"],
             "rows": [["**bold text**", "MS03-AAA-789"]]}
    assert is_redacted_table(table) is False


def test_redacted_text_block_detected():
    assert is_redacted_text("*** **-*********-*** ***") is True


def test_normal_text_not_flagged():
    assert is_redacted_text("Torque to 12 N*m and check for leaks.") is False


def test_empty_text_not_flagged():
    assert is_redacted_text("") is False
    assert is_redacted_text(None) is False


# ---------------------------------------------------------------------------
# tag_blocks_with_redaction
# ---------------------------------------------------------------------------

def test_tags_redacted_table_block_with_metadata():
    blocks = [{
        "type": "table", "text": "| No. | Parts No. |",
        "table_data": {"headers": ["No.", "Parts No."], "rows": [["***", "**-*-*"]]},
        "metadata": {},
    }]
    tag_blocks_with_redaction(blocks)
    assert blocks[0]["metadata"]["redacted"] is True
    assert "redacted" in blocks[0]["metadata"]["redaction_reason"].lower() or \
           "blanked" in blocks[0]["metadata"]["redaction_reason"].lower()


def test_leaves_normal_table_untouched():
    blocks = [{
        "type": "table", "text": "| a | b |",
        "table_data": {"headers": ["a"], "rows": [["real value"]]},
        "metadata": {},
    }]
    tag_blocks_with_redaction(blocks)
    assert "redacted" not in blocks[0]["metadata"]


def test_tags_redacted_text_block():
    blocks = [{"type": "text", "text": "*** *** ***", "metadata": {}}]
    tag_blocks_with_redaction(blocks)
    assert blocks[0]["metadata"]["redacted"] is True


def test_skips_non_dict_entries_without_crashing():
    blocks = [{"type": "table", "text": "x", "table_data": None, "metadata": {}}, None]
    tag_blocks_with_redaction(blocks)   # must not raise
