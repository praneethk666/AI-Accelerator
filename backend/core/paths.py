"""Shared helper for recovering the user-facing filename from a disk path.

The API saves uploads as "{document_id}_{original_filename}" (main.py's /upload
and /files/stage) so concurrent uploads never collide on disk. Every extractor
independently derives a "filename" for citations/blocks from that disk path —
without stripping the prefix, citations show the raw UUID-prefixed disk name
instead of what the user actually uploaded.
"""
from __future__ import annotations

import os
import re

_ID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_"
)


def display_filename(file_path: str) -> str:
    """basename(file_path) with a leading '{uuid}_' storage prefix stripped, if
    present. Safe on paths that never had one (e.g. test fixtures) — returns the
    plain basename unchanged."""
    return _ID_PREFIX_RE.sub("", os.path.basename(file_path), count=1)
