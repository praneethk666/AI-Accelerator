"""Lossless round-trip YAML handler preserving comments, structure, and formatting.

Uses ruamel.yaml to ensure that edits to config files (such as from the
frontend Settings UI) do not wipe out comments (#), blank lines, or key order.
"""

from __future__ import annotations

import io
import os
from typing import Any

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    HAS_RUAMEL = True
except ImportError:
    import yaml
    HAS_RUAMEL = False
    CommentedMap = dict
    CommentedSeq = list


def get_yaml_instance():
    """Create a configured ruamel.yaml instance for round-trip operations."""
    if not HAS_RUAMEL:
        return None
    yaml_obj = YAML(typ="rt")  # round-trip
    yaml_obj.preserve_quotes = True
    yaml_obj.width = 4096  # Prevent unnecessary line-wrapping of long strings
    yaml_obj.indent(mapping=2, sequence=4, offset=2)
    return yaml_obj


def load_yaml_roundtrip(path_or_stream: str | io.IOBase) -> CommentedMap | dict:
    """Load YAML file or string while preserving comments and structure."""
    if HAS_RUAMEL:
        yaml_obj = get_yaml_instance()
        if isinstance(path_or_stream, str):
            if os.path.isfile(path_or_stream):
                with open(path_or_stream, "r", encoding="utf-8") as f:
                    data = yaml_obj.load(f)
            else:
                data = yaml_obj.load(path_or_stream)
        else:
            data = yaml_obj.load(path_or_stream)
        return data if data is not None else CommentedMap()
    else:
        if isinstance(path_or_stream, str):
            if os.path.isfile(path_or_stream):
                with open(path_or_stream, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(path_or_stream)
        else:
            data = yaml.safe_load(path_or_stream)
        return data if data is not None else {}


def dump_yaml_roundtrip(data: Any, target_path_or_stream: str | io.IOBase | None = None) -> str | None:
    """Dump YAML data while preserving comments and formatting.
    
    If target_path_or_stream is a file path (str), writes to that file and returns None.
    If target_path_or_stream is a stream, writes to the stream and returns None.
    If None, returns the YAML text as a string.
    """
    if HAS_RUAMEL:
        yaml_obj = get_yaml_instance()
        if target_path_or_stream is None:
            buf = io.StringIO()
            yaml_obj.dump(data, buf)
            return buf.getvalue()
        elif isinstance(target_path_or_stream, str):
            with open(target_path_or_stream, "w", encoding="utf-8") as f:
                yaml_obj.dump(data, f)
            return None
        else:
            yaml_obj.dump(data, target_path_or_stream)
            return None
    else:
        if target_path_or_stream is None:
            return yaml.dump(data, sort_keys=False, allow_unicode=True)
        elif isinstance(target_path_or_stream, str):
            with open(target_path_or_stream, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            return None
        else:
            yaml.dump(data, target_path_or_stream, sort_keys=False, allow_unicode=True)
            return None


def apply_settings_in_place(
    raw: CommentedMap | dict,
    settings: dict,
    settings_map: dict[str, list[str]],
    optional_override_keys: set[str] | None = None,
) -> CommentedMap | dict:
    """Apply form settings to a CommentedMap in-place without destroying comments.
    
    Args:
        raw: The round-trip loaded YAML structure (CommentedMap).
        settings: Flat dict of setting key -> value from the UI.
        settings_map: Mapping of flat key -> nested path in YAML.
        optional_override_keys: Set of keys where blank/None means remove the key.
    """
    optional_override_keys = optional_override_keys or set()

    for key, path in settings_map.items():
        if key not in settings:
            continue
        val = settings[key]

        # Optional override left blank => remove key so step inherits default
        if key in optional_override_keys and (val is None or val == ""):
            cur = raw
            for k in path[:-1]:
                if not isinstance(cur, (dict, CommentedMap)) or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if isinstance(cur, (dict, CommentedMap)):
                cur.pop(path[-1], None)
            continue

        if val is None:
            continue

        cur = raw
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], (dict, CommentedMap)):
                cur[k] = CommentedMap()
            cur = cur[k]
        cur[path[-1]] = val

    # Handle structured settings in-place to avoid wiping comments on parent maps
    if "vision_prompts" in settings and isinstance(settings["vision_prompts"], dict):
        if "vision" not in raw or not isinstance(raw["vision"], (dict, CommentedMap)):
            raw["vision"] = CommentedMap()
        
        existing_prompts = raw["vision"].get("prompt")
        if isinstance(existing_prompts, (dict, CommentedMap)):
            # Mutate existing keys and add new ones in-place
            new_prompts = settings["vision_prompts"]
            # Remove deleted prompt keys
            for k in list(existing_prompts.keys()):
                if k not in new_prompts:
                    del existing_prompts[k]
            for k, v in new_prompts.items():
                existing_prompts[k] = v
        else:
            raw["vision"]["prompt"] = settings["vision_prompts"]

    if "ingestion_steps" in settings and isinstance(settings["ingestion_steps"], list):
        if "ingestion" not in raw or not isinstance(raw["ingestion"], (dict, CommentedMap)):
            raw["ingestion"] = CommentedMap()
        raw["ingestion"]["steps"] = settings["ingestion_steps"]

    if "route_gates" in settings and isinstance(settings["route_gates"], dict):
        if "ingestion" not in raw or not isinstance(raw["ingestion"], (dict, CommentedMap)):
            raw["ingestion"] = CommentedMap()
        raw["ingestion"]["route_gates"] = settings["route_gates"]

    return raw
