"""LangGraph pipeline tests.  Run:  python tests/test_graph.py   (or:  pytest)

Proves the four backbone guarantees: runs end-to-end, steps toggle, routes
branch, and one failing tool never kills the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState
from backend.pipeline.graph import build_pipeline
from backend.pipeline.stub_tools import STUB_NAMES

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
    out = build_pipeline(_registry(), cfg).invoke({"document_id": "d1"})
    assert out["errors"] == []
    assert (
        out["chunks"] and out["chunks"][0]["indexed"] is True
    )  # reached the last step


def test_steps_toggle_via_config():
    # vision_enrichment dropped from steps -> its node never runs, even on diagram route
    steps = [s for s in ALL_STEPS if s != "vision_enrichment"]
    cfg = PipelineConfig.from_dict({"route": "diagram_heavy", "steps": steps})
    out = build_pipeline(_registry(), cfg).invoke({"document_id": "d1"})
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
        build_pipeline(_registry(), diagram).invoke({"document_id": "d1"})
    )
    assert not _has_caption(
        build_pipeline(_registry(), text).invoke({"document_id": "d1"})
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
        out = graph.invoke({"document_id": "d1", "file_type": file_type})
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
        {"document_id": "d1", "file_type": "cad"}
    )
    assert _vias(out) == []
    assert out["errors"] == []


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
        {"document_id": "d1", "blocks": [{"text": "x"}]}
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
