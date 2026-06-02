"""Tool registry:

The integration contract: write a Tool (a `name` + `run`), register it,
add its name to `config.steps`. No edits to anyone else's code.
"""
from __future__ import annotations

from backend.core.tool import Tool


class ToolRegistry:
    """Maps tool name -> instance so the graph can look tools up by config.steps."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        # Register a tool under its `.name`; returns it for convenience.

        name = getattr(tool, "name", None)
        if not name:
            raise ValueError(f"tool {tool!r} has no non-empty 'name' attribute")
        if name in self._tools:
            raise ValueError(f"a tool named '{name}' is already registered")
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Shared default registry for tools that self-register on import; wiring code may
# also make its own for isolation/tests.
default_registry = ToolRegistry()
