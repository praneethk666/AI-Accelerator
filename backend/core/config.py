from __future__ import annotations
from dataclasses import dataclass, field


# Safe-state default when route is not declared. Matches the .md rules:
#       low confidence / unknown -> text_default, never wrong-and-confident.
DEFAULT_ROUTE = "text_default"


def load_config(path: str) -> dict:
    # Load a pipeline profile YML into a plain dict.
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class PipelineConfig:
    """A parsed pipeline profile.

    Attributes:
        route: The profile's declared route (e.g. "diagram_heavy"). Drives the
            graph's conditional edge. Defaults to DEFAULT_ROUTE if absent.
        steps: Ordered tool names to run. A tool not listed here simply does not
            run — this is how steps toggle on/off without code changes.
        raw: The full config dict. Tools read their own sections from here, so
            new settings never require a change to this core module.
    """

    route: str = DEFAULT_ROUTE
    steps: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Build a PipelineConfig from an already-loaded config dict."""
        data = data or {}
        return cls(
            route=data.get("route", DEFAULT_ROUTE),
            steps=list(data.get("steps", [])),
            raw=data,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load and parse a pipeline profile YAML in one call."""
        return cls.from_dict(load_config(path))

    def section(self, name: str, default: dict | None = None) -> dict:
        """Return a tool's own settings block (e.g. config.section("vision")).

        Returns an empty dict (or the given default) when the section is absent,
        so a tool can read its settings without guarding for missing keys.
        """
        value = self.raw.get(name, default)
        return value if isinstance(value, dict) else (default or {})
