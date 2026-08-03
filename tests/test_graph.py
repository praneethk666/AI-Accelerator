"""LangGraph pipeline tests.  Run:  python tests/test_graph.py   (or:  pytest)

Proves the four backbone guarantees: runs end-to-end, steps toggle, routes
branch, and one failing tool never kills the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState
from backend.pipeline.graph import build_pipeline
from tests.stub_tools import STUB_NAMES


@pytest.fixture(autouse=True)
def _no_cancellation_db_check(monkeypatch):
    # Each node now calls check_cancelled(document_id), which looks the document
    # row up in Postgres — these are hermetic orchestration tests with a fake
    # document_id ("d1") that was never inserted, so without this they'd all
    # raise IngestionCancelledError as if the (nonexistent) doc had been deleted.
    monkeypatch.setattr("backend.pipeline.graph.check_cancelled", lambda document_id: None)

ALL_STEPS = [
    "categorize",
    "page_profile",
    "pdf_extraction",
    "vision_enrichment",
    "chunk",
    "enrich_chunks",
    "embed",
    "index",
]
GATES = {"vision_enrichment": ["diagram_heavy"]}


def _registry():
    reg = ToolRegistry()
    for stub in STUB_NAMES:
        reg.register(stub)
    return reg


def _has_caption(state: PipelineState) -> bool:
    return any(b.get("type") == "image_caption" for b in state.get("blocks", []))


def test_runs_end_to_end():
    cfg = PipelineConfig.from_dict(
        {"route": "diagram_heavy", "steps": ALL_STEPS, "route_gates": GATES}
    )
    out = build_pipeline(_registry(), cfg).invoke({"document_id": "default"})
    assert out["errors"] == []
    assert (
        out["chunks"] and out["chunks"][0]["indexed"] is True
    )  # reached the last step


def test_steps_toggle_via_config():
    # vision_enrichment dropped from steps -> its node never runs, even on diagram route
    steps = [s for s in ALL_STEPS if s != "vision_enrichment"]
    cfg = PipelineConfig.from_dict({"route": "diagram_heavy", "steps": steps})
    out = build_pipeline(_registry(), cfg).invoke({"document_id": "default"})
    assert not _has_caption(out)


def test_route_branches_on_gate():
    # same steps, different route -> gate keeps/skips vision_enrichment
    diagram = PipelineConfig.from_dict(
        {"route": "diagram_heavy", "steps": ALL_STEPS, "route_gates": GATES}
    )
    text = PipelineConfig.from_dict(
        {"route": "text_default", "steps": ALL_STEPS, "route_gates": GATES}
    )
    assert _has_caption(
        build_pipeline(_registry(), diagram).invoke({"document_id": "default"})
    )
    assert not _has_caption(
        build_pipeline(_registry(), text).invoke({"document_id": "default"})
    )


EXTRACTORS = {
    "pdf": "pdf_extraction",
    "excel": "excel_extraction",
    "ppt": "ppt_extraction",
    "image": "image_extraction",
}


def _vias(state: PipelineState) -> list:
    return [b.get("via") for b in state.get("blocks", [])]


def test_extract_placeholder_dispatches_on_file_type():
    # `extract` expands to all extractors; only the one matching file_type runs
    cfg = PipelineConfig.from_dict(
        {
            "route": "text_default",
            "steps": ["categorize", "extract", "chunk"],
            "extractors": EXTRACTORS,
        }
    )
    graph = build_pipeline(_registry(), cfg)
    for file_type, _ in EXTRACTORS.items():
        out = graph.invoke({"document_id": "default", "file_type": file_type})
        assert _vias(out) == [file_type]  # exactly the matching extractor ran


def test_extract_skips_all_when_file_type_unknown():
    # no extractor matches -> none run, graph still completes (graceful)
    cfg = PipelineConfig.from_dict(
        {
            "route": "text_default",
            "steps": ["categorize", "extract", "chunk"],
            "extractors": EXTRACTORS,
        }
    )
    out = build_pipeline(_registry(), cfg).invoke(
        {"document_id": "default", "file_type": "cad"}
    )
    assert _vias(out) == []
    assert out["errors"] == []


class _ViaStub:
    """Minimal extractor stub that tags which extractor ran."""

    def __init__(self, name):
        self.name = name

    def run(self, state, config):
        state.setdefault("blocks", []).append({"type": "text", "via": self.name})
        return state


def _pdf_registry():
    reg = _registry()
    for n in ("pdf_digital", "scanned_pdf", "mixed_pdf", "cad_extract"):
        reg.register(_ViaStub(n))
    return reg


PDF_CFG = {
    "route": "text_default",
    "steps": ["categorize", "extract", "chunk"],
    "extractors": {"excel": "excel_extraction", "ppt": "ppt_extraction"},
    "pdf_extractors": {
        "digital": "pdf_digital",
        "scanned": "scanned_pdf",
        "mixed": "mixed_pdf",
    },
    "route_extractors": {"cad_route": "cad_extract", "circuit_route": "cad_extract"},
}


def test_pdf_dispatches_on_pdf_kind():
    # file_type=pdf -> the extractor matching state["pdf_kind"] runs, alone
    graph = build_pipeline(_pdf_registry(), PipelineConfig.from_dict(PDF_CFG))
    for kind, tool in [
        ("digital", "pdf_digital"),
        ("scanned", "scanned_pdf"),
        ("mixed", "mixed_pdf"),
    ]:
        out = graph.invoke(
            {"document_id": "default", "file_type": "pdf", "pdf_kind": kind}
        )
        assert _vias(out) == [tool]


def test_route_extractor_overrides_file_type():
    # on cad_route, cad_extract runs and the pdf extractors are suppressed
    cfg = dict(PDF_CFG, route="cad_route")
    out = build_pipeline(_pdf_registry(), PipelineConfig.from_dict(cfg)).invoke(
        {"document_id": "default", "file_type": "pdf", "pdf_kind": "digital",
         "route": "cad_route"}
    )
    assert _vias(out) == ["cad_extract"]


def test_non_pdf_still_dispatches_on_file_type():
    # excel/ppt unaffected by the pdf/route machinery
    out = build_pipeline(_pdf_registry(), PipelineConfig.from_dict(PDF_CFG)).invoke(
        {"document_id": "default", "file_type": "excel"}
    )
    assert _vias(out) == ["excel"]


def test_failing_tool_degrades_gracefully():
    class Boom:
        name = "boom"

        def run(self, state, config):
            raise RuntimeError("kaboom")

    reg = _registry()
    reg.register(Boom())
    cfg = PipelineConfig.from_dict(
        {"route": "text_default", "steps": ["boom", "chunk"]}
    )
    out = build_pipeline(reg, cfg).invoke(
        {"document_id": "default", "blocks": [{"text": "x"}]}
    )
    assert any("boom: kaboom" in e for e in out["errors"])  # error captured
    assert "chunks" in out  # graph kept going


if __name__ == "__main__":
    test_runs_end_to_end()
    test_steps_toggle_via_config()
    test_route_branches_on_gate()
    test_extract_placeholder_dispatches_on_file_type()
    test_extract_skips_all_when_file_type_unknown()
    test_failing_tool_degrades_gracefully()
    print("graph tests passed")
