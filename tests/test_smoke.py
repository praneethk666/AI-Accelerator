"""Smoke test: a dummy tool flows through the stub runner and updates state.
Run from the repo root:  python tests/test_smoke.py   (or:  pytest)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.tool import PipelineState
from backend.pipeline.graph import run_pipeline


class EchoTool:
    name = "echo"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append({"type": "text", "text": "hello"})
        return state


def test_pipeline_runs():
    tools = {"echo": EchoTool()}
    config = {"steps": ["echo"]}
    # document_id "default" is exempted from check_cancelled's DB lookup (see
    # backend/core/tool.py) — no mock needed.
    out = run_pipeline(tools, {"document_id": "default"}, config)
    assert out["blocks"][0]["text"] == "hello"
    assert out["errors"] == []


if __name__ == "__main__":
    test_pipeline_runs()
    print("smoke test passed")
