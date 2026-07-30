"""Load a pipeline config into a plain dict.

Supports ${VAR_NAME} substitution in YAML values — resolved from the
environment. Missing vars are left as the literal ${VAR_NAME} string so
the caller can detect and report them.

Usage:
    from backend.core.config import load_config, PipelineConfig
    config = load_config("config/global.yaml")
    pc = PipelineConfig.from_yaml("config/pipeline.example.yaml")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Safe-state default when route is not declared. Matches the .md rules:
#       low confidence / unknown -> text_default, never wrong-and-confident.
DEFAULT_ROUTE = "text_default"


def load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), raw)
    return yaml.safe_load(raw)


def get_db_url(config: dict) -> str | None:
    """Extract postgres database URL from configuration or environment.
    
    Checks config['database']['postgres_url'] first, then falls back to env POSTGRES_URL,
    and finally to individual POSTGRES_* env vars.
    """
    db_url = config.get("database", {}).get("postgres_url")
    if not db_url or "${" in str(db_url):
        db_url = os.getenv("POSTGRES_URL")
    
    if not db_url:
        host = os.getenv("POSTGRES_HOST")
        if host:
            port = os.getenv("POSTGRES_PORT", "5432")
            dbname = os.getenv("POSTGRES_DB", "accelerator")
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            db_url = f"host={host} port={port} dbname={dbname} user={user} password={password}"
            
    return db_url



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
