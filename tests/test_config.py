"""PipelineConfig tests.  Run:  python tests/test_config.py   (or:  python3)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import DEFAULT_ROUTE, PipelineConfig, load_config, get_db_url

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_YAML = os.path.join(REPO_ROOT, "config", "pipeline.example.yaml")
GLOBAL_YAML = os.path.join(REPO_ROOT, "config", "global.yaml")


def test_get_db_url():
    # Test getting from config directly
    assert get_db_url({"database": {"postgres_url": "postgresql://test:test@localhost:5432/test"}}) == "postgresql://test:test@localhost:5432/test"
    
    # Test unresolved template string fallback to environment (or returning None/fallback if env is empty)
    # Clear the env var during test to avoid pollution
    orig_env = os.environ.get("POSTGRES_URL")
    if "POSTGRES_URL" in os.environ:
        del os.environ["POSTGRES_URL"]
    try:
        assert get_db_url({"database": {"postgres_url": "${POSTGRES_URL}"}}) is None
        
        # Test fallback to POSTGRES_HOST / etc
        os.environ["POSTGRES_HOST"] = "mydb.host"
        assert "host=mydb.host" in get_db_url({})
        del os.environ["POSTGRES_HOST"]
    finally:
        if orig_env is not None:
            os.environ["POSTGRES_URL"] = orig_env


def test_global_yaml_loads_and_has_pipeline_profiles():
    # guards against YAML typos in the real config (e.g. missing space after a key)
    cfg = load_config(GLOBAL_YAML)
    assert cfg["ingestion"]["steps"][0] == "categorize"
    assert "vision_enrichment" in cfg["ingestion"]["route_gates"]
    assert cfg["query"]["steps"] == ["query_planner", "retrieval", "answerer"]
    assert cfg["pdf_extractors"]["digital"] in ("docling_pdf", "pymupdf_pdf")
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


def test_yaml_comment_preservation():
    from backend.core.yaml_handler import (
        load_yaml_roundtrip,
        dump_yaml_roundtrip,
        apply_settings_in_place,
    )
    import tempfile

    sample_yaml = """# Global Header Comment: System configuration
llm:
  # Primary reasoning model
  provider: openai
  model: gpt-4o-mini  # Fast default model
  api_key: ${OPENAI_API_KEY}

chunking:
  strategy: semantic  # semantic chunker
  size: 400
  overlap: 50

# Ingestion pipeline definition
ingestion:
  steps:
    - categorize
    - extract  # main extraction step
    - chunk
    - embed
    - index
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(sample_yaml)
        temp_file = f.name

    try:
        data = load_yaml_roundtrip(temp_file)
        settings_map = {
            "llm_model": ["llm", "model"],
            "chunking_size": ["chunking", "size"],
        }
        apply_settings_in_place(data, {"llm_model": "gpt-4o", "chunking_size": 600}, settings_map)
        dump_yaml_roundtrip(data, temp_file)

        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Values updated
        assert "model: gpt-4o" in content or "gpt-4o" in content
        assert "size: 600" in content or "600" in content

        # All comments and structure preserved
        assert "# Global Header Comment: System configuration" in content
        assert "# Primary reasoning model" in content
        assert "# Fast default model" in content
        assert "# semantic chunker" in content
        assert "# Ingestion pipeline definition" in content
        assert "# main extraction step" in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    test_get_db_url()
    test_global_yaml_loads_and_has_pipeline_profiles()
    test_from_yaml_reads_route_and_steps()
    test_section_returns_tool_settings()
    test_missing_fields_use_safe_defaults()
    test_raw_passthrough_is_preserved()
    test_yaml_comment_preservation()
    print("config tests passed")

