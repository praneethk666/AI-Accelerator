"""Utility functions to save PageProfile and NormalizedBlock lists as JSON."""

import os
import json
from dataclasses import asdict, is_dataclass
from typing import Any, List, Union

from backend.core.schemas import PageProfile, NormalizedBlock


def _default_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for non‑serializable objects.
    Handles bytes (if any) and dataclasses.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, bytes):
        # Convert bytes to a placeholder string (or base64 if needed)
        return f"<binary: {len(obj)} bytes>"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_page_profiles(
    profiles: List[PageProfile],
    pdf_path: str,
    output_dir: str = "output/page_profiles"
) -> str:
    """
    Save a list of PageProfile objects to a JSON file.

    Args:
        profiles: List of PageProfile dataclass instances.
        pdf_path: Path to the original PDF (used to derive filename).
        output_dir: Directory where the JSON file will be saved.

    Returns:
        The path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, f"{pdf_name}_page_profiles.json")

    data = [asdict(p) for p in profiles]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer)

    print(f"Saved page profiles to: {out_path}")
    return out_path


def save_blocks(
    blocks: List[NormalizedBlock],
    pdf_path: str,
    output_dir: str = "output/blocks"
) -> str:
    """
    Save a list of NormalizedBlock objects to a JSON file.

    Args:
        blocks: List of NormalizedBlock dataclass instances.
        pdf_path: Path to the original PDF (used to derive filename).
        output_dir: Directory where the JSON file will be saved.

    Returns:
        The path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, f"{pdf_name}_blocks.json")

    data = [asdict(b) for b in blocks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer)

    print(f"Saved blocks to: {out_path}")
    return out_path