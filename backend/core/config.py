"""Load a pipeline config into a plain dict.

Supports ${VAR_NAME} substitution in YAML values — resolved from the
environment. Missing vars are left as the literal ${VAR_NAME} string so
the caller can detect and report them.

Usage:
    from backend.core.config import load_config
    config = load_config("config/global.yaml")
"""
from __future__ import annotations
import os
import re


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = f.read()
    raw = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), raw)
    return yaml.safe_load(raw)
