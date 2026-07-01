"""PipelineConfig tests.  Run:  python tests/test_config.py   (or:  python3)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import DEFAULT_ROUTE, PipelineConfig, load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_YAML = os.path.join(REPO_ROOT, "config", "pipeline.example.yaml")
GLOBAL_YAML = os.path.join(REPO_ROOT, "config", "global.yaml")


def test_global_yaml_loads_and_has_pipeline_profiles():
    # guards against YAML typos in the real config (e.g. missing space after a key)
    cfg = load_config(GLOBAL_YAML)
    assert cfg["ingestion"]["steps"][0] == "categorize"
    assert "vision_enrichment" in cfg["ingestion"]["route_gates"]
    assert cfg["query"]["steps"] == ["query_planner", "retrieval", "answerer"]
    assert cfg["pdf_extractors"]["digital"] == "docling_pdf"
    assert cfg["route_extractors"]["cad_route"] == "cad_extract"
    # the categorization block parses (had a missing-space bug)
    assert "manufacturing" in cfg["categorization"]["industry_keywords"]


def test_from_yaml_reads_route_and_steps():
    cfg = PipelineConfig.from_yaml(EXAMPLE_YAML)
    assert cfg.route == "diagram_heavy"
    assert cfg.steps[0] == "categorize"
    assert "vision_enrichment" in cfg.steps


def test_section_returns_tool_settings():
    # A tool reads only its own block; the graph never hardcodes these values.
    cfg = PipelineConfig.from_yaml(EXAMPLE_YAML)
    assert cfg.section("vision")["dpi"] == 150
    assert cfg.section("chunking")["size"] == 800


def test_missing_fields_use_safe_defaults():
    cfg = PipelineConfig.from_dict({})
    assert cfg.route == DEFAULT_ROUTE  # unknown route -> safe-state default
    assert cfg.steps == []  # no steps -> nothing runs, no crash
    assert cfg.section("vision") == {}  # missing section -> empty, not error


def test_raw_passthrough_is_preserved():
    # Tool-specific keys survive untouched, so new settings never need a core edit.
    cfg = PipelineConfig.from_dict({"steps": ["chunk"], "custom": {"k": 1}})
    assert cfg.raw["custom"]["k"] == 1


if __name__ == "__main__":
    test_from_yaml_reads_route_and_steps()
    test_section_returns_tool_settings()
    test_missing_fields_use_safe_defaults()
    test_raw_passthrough_is_preserved()
    print("config tests passed")
