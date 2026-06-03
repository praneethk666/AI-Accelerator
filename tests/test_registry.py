"""ToolRegistry tests.  Run:  python tests/test_registry.py   (or:  python3)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.registry import ToolRegistry
from backend.core.tool import PipelineState


class DummyTool:
    """Exactly what a teammate writes to plug in: a name + a run()."""

    name = "dummy"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        return state


def test_register_and_get():
    reg = ToolRegistry()
    tool = reg.register(DummyTool())
    assert reg.get("dummy") is tool
    assert "dummy" in reg
    assert reg.names() == ["dummy"]


def test_get_unknown_returns_none():
    reg = ToolRegistry()
    assert reg.get("nope") is None
    assert "nope" not in reg


def test_duplicate_name_raises():
    # Duplicate name = wiring mistake -> fail loud, don't silently shadow.
    reg = ToolRegistry()
    reg.register(DummyTool())
    try:
        reg.register(DummyTool())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate tool name")


def test_missing_name_raises():
    reg = ToolRegistry()

    class Nameless:
        name = ""

        def run(self, state, config):
            return state

    try:
        reg.register(Nameless())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty tool name")


if __name__ == "__main__":
    test_register_and_get()
    test_get_unknown_returns_none()
    test_duplicate_name_raises()
    test_missing_name_raises()
    print("registry tests passed")
