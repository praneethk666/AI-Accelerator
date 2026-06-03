"""Render the config-driven pipeline graph to a Mermaid PNG.

- builds the graph from a config + the stub tools, then draws it
- the picture reflects the CONFIG: toggled-off steps vanish, route_gates show as
  conditional (dotted) edges -> a real check that "config drives the graph"
- NOTE: draw_mermaid_png() sends the graph STRUCTURE (tool names + edges, no doc
  content) to the mermaid.ink web API, so it needs internet

Run from repo root:
    python scripts/visualize_graph.py                         # example config -> docs/pipeline_graph.png
    python scripts/visualize_graph.py <config.yaml> <out.png>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import PipelineConfig
from backend.core.registry import ToolRegistry
from backend.pipeline.graph import build_pipeline
from backend.pipeline.stub_tools import STUB_NAMES

DEFAULT_CONFIG = "config/pipeline.example.yaml"
DEFAULT_OUT = "docs/pipeline_graph.png"


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    reg = ToolRegistry()
    for stub in STUB_NAMES:
        reg.register(stub)

    cfg = PipelineConfig.from_yaml(config_path)
    graph = build_pipeline(reg, cfg).get_graph()

    png = graph.draw_mermaid_png()  # -> mermaid.ink
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(png)
    print(f"wrote {out_path}  (route={cfg.route}, steps={len(cfg.steps)})")


if __name__ == "__main__":
    main()
