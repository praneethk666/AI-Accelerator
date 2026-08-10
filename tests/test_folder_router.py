"""Tests for backend.categorize.folder_router.route_from_path -- derives
document_type + machine/doc_category/component tags from a corpus's own folder
structure (validated live against the real Toyoda TNGA manual tree, 3-Aug).

Run: pytest tests/test_folder_router.py
"""
import logging
import os
from unittest.mock import patch

from backend.categorize.folder_router import route_from_path, tag_blocks_with_folder_info

ROOT = "/corpus/TNGA"
GD_ROOT = "/corpus/GD"


def test_none_when_file_outside_corpus_root():
    assert route_from_path("/somewhere/else/file.pdf", ROOT) is None


def test_none_when_file_sits_directly_at_root():
    assert route_from_path(os.path.join(ROOT, "file.pdf"), ROOT) is None


def test_none_when_no_category_keyword_matches():
    p = os.path.join(ROOT, "SOME_MACHINE", "99_Unrecognized folder", "file.pdf")
    assert route_from_path(p, ROOT) is None


def test_instruction_manual_matched_as_manual():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER", "3.INSTRUCTION MANUAL", "EN", "x.pdf")
    info = route_from_path(p, ROOT)
    assert info["force_document_type"] == "manual"
    assert info["machine"] == "120_CYLINDRICAL GRINDER"
    assert info["doc_category"] == "3.INSTRUCTION MANUAL"
    assert info["component"] == "EN"


def test_machine_assembly_drawing_matched_as_cad_with_component():
    p = os.path.join(ROOT, "OP200_TIGG300_HONING MC_JTEKT", "04_Machine assembly drawing",
                     "04_Spindlehead", "part.pdf")
    info = route_from_path(p, ROOT)
    assert info["force_document_type"] == "cad_drawing"
    assert info["component"] == "04_Spindlehead"


def test_no_component_when_file_sits_directly_in_category_folder():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER",
                     "1.Expendable parts drawing Expendable parts drawing", "x.pdf")
    info = route_from_path(p, ROOT)
    assert info["component"] is None


def test_cad_keyword_never_forced_on_a_non_pdf_file():
    # real bug caught before launch: cad_drawing/circuit_diagram map to a
    # route_extractor override that REPLACES the file_type extractor entirely
    # (backend/pipeline/graph.py) -- forcing it on an .xlsx would route the file
    # into cad_extract.py, which opens files via fitz and would just crash.
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER",
                     "1.Expendable parts drawing Expendable parts drawing", "bom.xlsx")
    info = route_from_path(p, ROOT)
    assert info["force_document_type"] is None
    # machine/doc_category/component tags survive even though the force doesn't
    assert info["machine"] == "120_CYLINDRICAL GRINDER"
    assert info["doc_category"] is not None


def test_circuit_folder_downgraded_to_manual_when_pdf_has_real_text_layer():
    p = os.path.join(ROOT, "OP200_TIGG300_HONING MC_JTEKT",
                     "05-2_PDF for printing drawings", "Electric circuit diagram(MAIN).pdf")
    with patch("backend.categorize.folder_router._has_text_layer", return_value=True):
        info = route_from_path(p, ROOT)
    assert info["force_document_type"] == "manual"


def test_circuit_folder_stays_circuit_diagram_when_no_text_layer():
    p = os.path.join(ROOT, "OP200_TIGG300_HONING MC_JTEKT",
                     "05-2_PDF for printing drawings", "scanned_print.pdf")
    with patch("backend.categorize.folder_router._has_text_layer", return_value=False):
        info = route_from_path(p, ROOT)
    assert info["force_document_type"] == "circuit_diagram"


def test_text_layer_check_skipped_when_undetermined_keeps_original_type():
    p = os.path.join(ROOT, "OP200_TIGG300_HONING MC_JTEKT",
                     "05-2_PDF for printing drawings", "missing.pdf")
    with patch("backend.categorize.folder_router._has_text_layer", return_value=None):
        info = route_from_path(p, ROOT)
    assert info["force_document_type"] == "circuit_diagram"


# ---------------------------------------------------------------------------
# GD/BF-5 scheme -- ADDED 10-Aug. Real folder names surveyed directly from the
# corpus, a completely different numbering/naming convention from TNGA above.
# ---------------------------------------------------------------------------

def test_parts_change_procedure_manual_matched_as_manual():
    p = os.path.join(GD_ROOT, "BF-5", "08-Parts change procedure manual", "x.pdf")
    info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "manual"
    assert info["machine"] == "BF-5"


def test_lubricating_manual_matched_as_manual():
    p = os.path.join(GD_ROOT, "BF-5", "10-Equipment Oiling & Lubricating Manual", "x.pdf")
    info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "manual"


def test_electrical_circuit_diagram_matched_as_circuit_diagram():
    p = os.path.join(GD_ROOT, "BF-5", "Electrical circuit diagram", "x.pdf")
    with patch("backend.categorize.folder_router._has_text_layer", return_value=None):
        info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "circuit_diagram"


def test_pneumatic_circuit_diagram_matched_as_circuit_diagram():
    p = os.path.join(GD_ROOT, "BF-5", "Pneumatic circuit diagram", "x.pdf")
    with patch("backend.categorize.folder_router._has_text_layer", return_value=None):
        info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "circuit_diagram"


def test_manufactured_parts_drawing_matched_as_cad_drawing():
    p = os.path.join(GD_ROOT, "BF-5", "Manufactured parts drawing（製作品）", "x.pdf")
    info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "cad_drawing"


def test_purchased_parts_drawing_matched_as_cad_drawing():
    p = os.path.join(GD_ROOT, "BF-5", "Purchased parts drawing（追加工図）", "x.pdf")
    info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "cad_drawing"


def test_cycle_line_diagram_matched_as_schematic():
    p = os.path.join(GD_ROOT, "BF-5", "Cycle line diagram", "x.pdf")
    info = route_from_path(p, GD_ROOT)
    assert info["force_document_type"] == "schematic"


def test_bilingual_japanese_only_sibling_folder_is_a_safe_noop():
    # Real corpus shape: "01-Table of contents" and "01-目次" exist as SIBLING
    # folders for the same content. The Japanese-only one has no English
    # substring to match any keyword -- must return None (fall through to
    # normal categorize), never raise or silently mis-tag.
    p = os.path.join(GD_ROOT, "BF-5", "01-目次", "x.pdf")
    assert route_from_path(p, GD_ROOT) is None


def test_unmatched_numbered_category_logs_a_diagnostic(caplog):
    # A real, not-yet-mapped scheme should be discoverable in logs, not a
    # silent, unmeasured gap.
    p = os.path.join(GD_ROOT, "BF-5", "20-SomeFutureCategoryNotYetMapped", "x.pdf")
    with caplog.at_level(logging.INFO, logger="backend.categorize.folder_router"):
        assert route_from_path(p, GD_ROOT) is None
    assert any("matched no keyword" in r.message for r in caplog.records)


def test_non_numbered_unmatched_folder_does_not_log():
    # Only NUMBERED category-folder-shaped segments are worth flagging -- an
    # ordinary unmatched folder name (no leading number) is expected/common
    # and shouldn't spam logs.
    p = os.path.join(GD_ROOT, "BF-5", "Miscellaneous", "x.pdf")
    assert route_from_path(p, GD_ROOT) is None


# ---------------------------------------------------------------------------
# tag_blocks_with_folder_info
# ---------------------------------------------------------------------------

def _block(text="x"):
    return {"type": "text", "text": text, "metadata": {}}


def test_tag_blocks_stamps_machine_doc_category_component():
    p = os.path.join(ROOT, "OP200_TIGG300_HONING MC_JTEKT", "04_Machine assembly drawing",
                     "04_Spindlehead", "part.pdf")
    blocks = [_block(), _block()]
    tag_blocks_with_folder_info(blocks, p, ROOT)
    for b in blocks:
        assert b["metadata"]["folder"] == {
            "machine": "OP200_TIGG300_HONING MC_JTEKT",
            "doc_category": "04_Machine assembly drawing",
            "component": "04_Spindlehead",
        }


def test_tag_blocks_noop_when_path_not_recognized():
    blocks = [_block()]
    out = tag_blocks_with_folder_info(blocks, "/somewhere/else/x.pdf", ROOT)
    assert "folder" not in blocks[0]["metadata"]
    assert out is blocks


def test_tag_blocks_omits_null_component_key():
    p = os.path.join(ROOT, "120_CYLINDRICAL GRINDER",
                     "1.Expendable parts drawing Expendable parts drawing", "x.pdf")
    blocks = [_block()]
    tag_blocks_with_folder_info(blocks, p, ROOT)
    assert "component" not in blocks[0]["metadata"]["folder"]
