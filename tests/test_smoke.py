"""Smoke test: a dummy tool flows through the stub runner and updates state.
Run from the repo root:  python tests/test_smoke.py   (or:  pytest)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

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
    # document_id "d1" is a fake id never inserted into Postgres — without this,
    # each node's check_cancelled(document_id) call would treat the "missing" row
    # as a cancelled ingestion and raise.
    with patch("backend.pipeline.graph.check_cancelled"):
        out = run_pipeline(tools, {"document_id": "d1"}, config)
    assert out["blocks"][0]["text"] == "hello"
    assert out["errors"] == []


if __name__ == "__main__":
    test_pipeline_runs()
    print("smoke test passed")
