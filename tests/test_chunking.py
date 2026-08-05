"""chunk_tool tests (no infra).  Run: python tests/test_chunking.py  (or pytest)

Asserts the contract: size-splitting + overlap, heading merge, atomic
tables/captions, skipped non-content, source_ref carried, and that every
emitted chunk is a valid Chunk (schema conformance).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chunking.chunk_tool import chunk_blocks, _try_extract_alarm_table_chunks
from backend.core.schemas import Chunk


def _text_block(text, **kw):
    b = {"block_id": "b", "document_id": "d1", "type": "text", "text": text}
    b.update(kw)
    return b


def test_text_splits_by_size_with_overlap():
    # one long sentence (no punctuation) -> hard word-split into size windows
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_blocks([_text_block(text)], size=10, overlap=3)
    assert len(chunks) > 1  # actually split
    assert all(c["token_count"] <= 10 for c in chunks)  # respects size
    # consecutive chunks share words -> overlap working
    first_words = set(chunks[0]["text"].split())
    second_words = set(chunks[1]["text"].split())
    assert first_words & second_words


def test_heading_merges_into_next_text():
    blocks = [
        {"type": "heading", "text": "Section 1", "document_id": "d1"},
        _text_block("body text here"),
    ]
    chunks = chunk_blocks(blocks, size=400)
    assert len(chunks) == 1  # heading did NOT become its own chunk
    assert chunks[0]["text"].startswith("Section 1")
    assert "body text here" in chunks[0]["text"]


def test_table_is_atomic_and_keeps_table_data():
    td = {"headers": ["a"], "rows": [["1"]]}
    blocks = [{"type": "table", "text": "tbl", "table_data": td, "document_id": "d1"}]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["table_data"] == td


def test_image_caption_atomic_with_image_path():
    blocks = [
        {
            "type": "image_caption",
            "text": "a bar chart",
            "metadata": {"image_path": "uploads/x.jpg"},
            "document_id": "d1",
        }
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["image_path"] == "uploads/x.jpg"


def _image_block(bbox, figure_kind, text="a figure"):
    return {
        "type": "image_caption",
        "text": text,
        "document_id": "d1",
        "metadata": {"image_path": "uploads/x.jpg", "figure_kind": figure_kind},
        "source_ref": {"filename": "manual.pdf", "page": 1, "bbox": bbox},
    }


def test_logo_excluded_regardless_of_size():
    chunks = chunk_blocks([_image_block([0, 0, 400, 300], "logo")])
    assert chunks == []


def test_small_icon_with_uncertain_kind_excluded():
    # validated live: warning/hazard icons and a page-header logo were all under
    # 35pt in their shorter side and classified "illustration" or "unknown"
    chunks = chunk_blocks([_image_block([0, 0, 30, 28], "illustration")])
    assert chunks == []


def test_small_crop_with_known_content_kind_kept():
    # a small crop the VLM confidently called a real content kind survives —
    # only UNCERTAIN small crops are excluded, not all small crops
    chunks = chunk_blocks([_image_block([0, 0, 30, 28], "photo")])
    assert len(chunks) == 1


def test_large_illustration_kept():
    # the real "11H" alarm indication diagram: 76x43pt, classified "illustration"
    chunks = chunk_blocks([_image_block([0, 0, 76, 43], "illustration")])
    assert len(chunks) == 1


def test_large_unknown_kept_fail_safe():
    # classification failed/rate-limited (kind=unknown) but the crop is large —
    # don't drop a possibly-real diagram just because the VLM call didn't confirm it
    chunks = chunk_blocks([_image_block([0, 0, 200, 200], "unknown")])
    assert len(chunks) == 1


def _big_table_block(n_rows=40):
    headers = ["Item", "Unit", "Value"]
    rows = [[f"spec_{i}", "mm", f"{i * 10}"] for i in range(n_rows)]
    return {
        "type": "table",
        "text": "\n".join(" | ".join(r) for r in [headers] + rows),  # long enough to force a split
        "table_data": {"headers": headers, "rows": rows},
        "document_id": "d1",
        "source_ref": {"filename": "spec.pdf", "page": 5},
    }


def test_large_table_splits_into_row_groups_with_repeated_header():
    chunks = chunk_blocks([_big_table_block(40)], size=60)
    assert len(chunks) > 1  # too big for one chunk -> split
    for i, c in enumerate(chunks):
        assert c["table_data"]["headers"] == ["Item", "Unit", "Value"]  # header repeated
        assert "Item" in c["text"]  # rendered markdown carries the header too
        assert c["source_ref"]["table_part"] == i + 1
        assert c["source_ref"]["table_parts"] == len(chunks)
        assert c["source_ref"]["filename"] == "spec.pdf"  # original source_ref preserved
    # every row appears exactly once across the parts (no loss, no duplication)
    all_rows = [r for c in chunks for r in c["table_data"]["rows"]]
    assert len(all_rows) == 40


def test_small_table_unaffected_by_split_large_tables():
    td = {"headers": ["a"], "rows": [["1"]]}
    blocks = [{"type": "table", "text": "tbl", "table_data": td, "document_id": "d1"}]
    chunks = chunk_blocks(blocks, split_large_tables=True)
    assert len(chunks) == 1
    assert chunks[0]["table_data"] == td
    assert "table_part" not in (chunks[0]["source_ref"] or {})


def test_split_large_tables_false_keeps_old_atomic_behavior():
    chunks = chunk_blocks([_big_table_block(40)], size=60, split_large_tables=False)
    assert len(chunks) == 1
    assert len(chunks[0]["table_data"]["rows"]) == 40


def test_orphaned_heading_before_table_merges_into_table_chunk():
    # heading immediately followed by a table, no body text between them — the
    # common case in table-heavy manuals ("2.1 Servo motor" -> straight into its
    # spec table). The heading must NOT become its own near-empty chunk.
    td = {"headers": ["a"], "rows": [["1"]]}
    blocks = [
        {"type": "heading", "text": "2.1 Servo motor", "document_id": "d1"},
        {"type": "table", "text": "tbl", "table_data": td, "document_id": "d1"},
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1  # no separate heading-only chunk
    assert chunks[0]["text"].startswith("2.1 Servo motor")
    assert chunks[0]["table_data"] == td


def test_chained_orphaned_headings_merge_into_one_lead():
    # nested headings with nothing between them ("2." -> "2.1 Servo motor" ->
    # "2.1.1 General specifications") should all merge into a single lead,
    # not each become their own orphan.
    td = {"headers": ["a"], "rows": [["1"]]}
    blocks = [
        {"type": "heading", "text": "2.", "document_id": "d1"},
        {"type": "heading", "text": "2.1 Servo motor", "document_id": "d1"},
        {"type": "heading", "text": "2.1.1 General specifications", "document_id": "d1"},
        {"type": "table", "text": "tbl", "table_data": td, "document_id": "d1"},
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert "2." in chunks[0]["text"]
    assert "2.1 Servo motor" in chunks[0]["text"]
    assert "2.1.1 General specifications" in chunks[0]["text"]


def test_trailing_orphaned_heading_is_dropped():
    # a heading with nothing after it anywhere (end of document) has nothing to
    # introduce -> dropped, not emitted as a bare chunk.
    blocks = [
        _text_block("some real body text here", source_ref={"page": 1}),
        {"type": "heading", "text": "Appendix", "document_id": "d1"},
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert "Appendix" not in chunks[0]["text"]


def test_orphaned_heading_still_merges_normally_with_following_text():
    # sanity check: the normal (non-orphaned) case is unaffected by this change.
    blocks = [
        {"type": "heading", "text": "Section 1", "document_id": "d1"},
        _text_block("body text here"),
    ]
    chunks = chunk_blocks(blocks, size=400)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Section 1")
    assert "body text here" in chunks[0]["text"]


def test_dot_leader_table_cell_does_not_produce_oversized_chunk():
    # a table-of-contents dot-leader cell ("Title . . . . . . . page") extracted
    # verbatim can balloon to thousands of characters in a SINGLE row (the
    # smallest splittable unit) -- validated live: one such cell crashed the
    # embedder with a 64 GiB self-attention allocation. Must come out small.
    dot_leader = "2. Specifications " + ". " * 2000 + " 2-1"
    td = {"headers": ["toc"], "rows": [[dot_leader]]}
    blocks = [{"type": "table", "text": dot_leader, "table_data": td, "document_id": "d1"}]
    chunks = chunk_blocks(blocks, size=400)
    assert len(chunks) == 1
    assert chunks[0]["token_count"] < 100  # collapsed, nowhere near the raw ~2000
    assert ". . . . . . . . . ." not in chunks[0]["text"]  # long run gone
    assert "2. Specifications" in chunks[0]["text"]  # real content preserved
    assert "2-1" in chunks[0]["text"]


def test_non_content_blocks_skipped():
    blocks = [{"type": "page_metrics", "text": "ignore me", "document_id": "d1"}]
    assert chunk_blocks(blocks) == []


def test_empty_text_produces_no_chunk():
    assert chunk_blocks([_text_block("   ")]) == []


def test_source_ref_carried_through():
    src = {"filename": "x.pdf", "page": 12}
    chunks = chunk_blocks([_text_block("hello world", source_ref=src)])
    assert chunks[0]["source_ref"] == src


# ── _try_extract_alarm_table_chunks: real bug found live, 28-Jul, on the servo
# manual's page-54 alarm table (F80H-F8CH). _RowspanTableParser (unlimited_ocr.py,
# used by both Unlimited-OCR and PaddleOCR-VL) denormalizes rowspan cells --
# repeating the alarm name/code into EVERY physical row a rowspan spans, by
# design (never blank). The old continuation-row check only treated a BLANK
# first column as "same alarm, merge" -- so a real 3-line "Alarm details" cell
# produced 3 separate duplicate alarm chunks instead of 1. ──────────────────────

def _alarm_block(headers, rows):
    return {"type": "table", "document_id": "d1",
            "table_data": {"headers": headers, "rows": rows},
            "source_ref": {"filename": "x.pdf", "page": 54}}


def test_alarm_rowspan_denormalized_continuation_rows_merge_into_one_chunk():
    # Real shape: PaddleOCR-VL's rowspan-denormalized output for one alarm whose
    # "Alarm details" cell wraps across 3 physical rows -- name/code repeated on
    # every row (never blank), Indication/Remarks also repeated.
    headers = ["Alarm name", "Code", "Alarm details, detecting and clearing method",
               "Indication", "Remarks"]
    rows = [
        ["No response from sensor", "82H",
         "State that no serial communication from sensor detected.", "8<=>2", "SU"],
        ["No response from sensor", "82H",
         "Detect the state no response from sensor.", "8<=>2", "SU"],
        ["No response from sensor", "82H",
         "Clear with re-powering on.", "8<=>2", "SU"],
    ]
    chunks = _try_extract_alarm_table_chunks(_alarm_block(headers, rows), "d1", lambda r: r)
    assert chunks is not None
    assert len(chunks) == 1  # all 3 physical rows are ONE alarm, not 3
    text = chunks[0]["text"]
    assert "State that no serial communication from sensor detected." in text
    assert "Detect the state no response from sensor." in text
    assert "Clear with re-powering on." in text
    # the repeated Indication/Remarks value shouldn't be piled up 3x
    assert text.count("8<=>2") == 1
    assert text.count("SU") == 1


def test_alarm_rows_with_distinct_ids_stay_separate_chunks():
    headers = ["Alarm name", "Code", "Details"]
    rows = [
        ["Sensor error 1", "83H", "Internal sensor failure detected."],
        ["Sensor error 2", "84H", "Internal sensor failure detected."],
    ]
    chunks = _try_extract_alarm_table_chunks(_alarm_block(headers, rows), "d1", lambda r: r)
    assert len(chunks) == 2
    assert chunks[0]["tags"]["alarm_name"] == "Sensor error 1"
    assert chunks[1]["tags"]["alarm_name"] == "Sensor error 2"


def test_alarm_continuation_row_with_old_blank_style_still_merges():
    # The ORIGINAL supported shape (non-denormalized upstream: continuation rows
    # leave the identity column blank) must keep working unchanged.
    headers = ["Alarm name", "Code", "Details"]
    rows = [
        ["Sensor error 1", "83H", "Internal sensor failure detected."],
        ["", "", "Clear with re-powering on."],
    ]
    chunks = _try_extract_alarm_table_chunks(_alarm_block(headers, rows), "d1", lambda r: r)
    assert len(chunks) == 1
    assert "Clear with re-powering on." in chunks[0]["text"]


def test_every_chunk_is_a_valid_chunk_schema():
    blocks = [
        {"type": "heading", "text": "H", "document_id": "d1"},
        _text_block("some body text", source_ref={"filename": "x.pdf", "page": 1}),
        {"type": "table", "text": "t", "table_data": {"headers": [], "rows": []}, "document_id": "d1"},
    ]
    for c in chunk_blocks(blocks):
        Chunk(**c)  # raises TypeError if any key is not a valid Chunk field


def test_redacted_table_flag_propagates_from_block_metadata_into_chunk():
    # real finding, 3-Aug: redaction_detect.py tags block metadata, but nothing
    # copied that through into the chunk the answerer actually sees -- fixed in
    # _make_chunk, mirroring how image_path is already selectively pulled through.
    blocks = [{
        "type": "table", "text": "| No. | Parts No. |", "document_id": "d1",
        "table_data": {"headers": ["No."], "rows": [["***"]]},
        "metadata": {"redacted": True, "redaction_reason": "blanked out in source"},
    }]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["redacted"] is True
    assert chunks[0]["redaction_reason"] == "blanked out in source"
    Chunk(**chunks[0])  # still a valid Chunk schema


def test_non_redacted_block_produces_chunk_without_the_flag():
    blocks = [{"type": "text", "text": "normal content", "document_id": "d1", "metadata": {}}]
    chunks = chunk_blocks(blocks)
    assert "redacted" not in chunks[0] or chunks[0]["redacted"] is False
    Chunk(**chunks[0])


def test_validate_repaired_rows_allows_synthesis_connector_words():
    # Same intent as an origin/main test that targeted the now-superseded
    # validate_repaired_chunks (no row_index anchoring) -- ported to the real,
    # currently-used _validate_repaired_rows so the coverage isn't lost.
    # Ordinary synthesis-connector words ("indicates", "detected",
    # "recommended", "checking") must not trip the traceability check just for
    # being connective prose -- they're in _REPAIR_STOPWORDS precisely so a
    # correct repair using only its own row's real content isn't rejected.
    from backend.chunking.chunk_tool import _validate_repaired_rows
    headers = ["Alarm Code", "Meaning", "Corrective Action"]
    rows = [["8bhh", "Overheating detected", "Check cooling fan and replace if damaged"]]
    parsed = [{
        "row_index": 0,
        "chunk_text": "Context indicates event for Alarm Code 8bhh where overheating was "
                      "detected. Recommended action includes checking cooling fan and "
                      "replacing component.",
        "structured": {"Alarm Code": "8bhh", "Meaning": "Overheating detected",
                       "Corrective Action": "Check cooling fan and replace if damaged"},
    }]
    ordered, err = _validate_repaired_rows(parsed, headers, rows, "Section 5.2 Alarms",
                                           "Preceding section text")
    assert err == ""
    assert ordered == parsed


if __name__ == "__main__":
    test_text_splits_by_size_with_overlap()
    test_heading_merges_into_next_text()
    test_table_is_atomic_and_keeps_table_data()
    test_large_table_splits_into_row_groups_with_repeated_header()
    test_small_table_unaffected_by_split_large_tables()
    test_split_large_tables_false_keeps_old_atomic_behavior()
    test_orphaned_heading_before_table_merges_into_table_chunk()
    test_chained_orphaned_headings_merge_into_one_lead()
    test_trailing_orphaned_heading_is_dropped()
    test_orphaned_heading_still_merges_normally_with_following_text()
    test_dot_leader_table_cell_does_not_produce_oversized_chunk()
    test_image_caption_atomic_with_image_path()
    test_non_content_blocks_skipped()
    test_empty_text_produces_no_chunk()
    test_source_ref_carried_through()
    test_alarm_rowspan_denormalized_continuation_rows_merge_into_one_chunk()
    test_alarm_rows_with_distinct_ids_stay_separate_chunks()
    test_alarm_continuation_row_with_old_blank_style_still_merges()
    test_every_chunk_is_a_valid_chunk_schema()
    test_validate_repaired_chunks_with_synthesis_connectors()
    print("chunking tests passed")

