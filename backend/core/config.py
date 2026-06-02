"""Load a pipeline profile into a plain dict.

No magic numbers in code — read settings from a profile here.
"""
from __future__ import annotations


def load_config(path: str) -> dict:
    import yaml  # pyyaml (see requirements.txt)
    with open(path) as f:
        return yaml.safe_load(f)
