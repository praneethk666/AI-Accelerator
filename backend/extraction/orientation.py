"""Page-orientation correction for the VLM render path.

Two kinds of rotated pages:
  - DIGITAL with a /Rotate flag  -> fitz `get_pixmap` already renders upright; nothing
    to do (the text layer is also fine).
  - SCANNED / image-only rotated 90/180/270 (a sideways scan, NO /Rotate metadata) ->
    the rendered image is genuinely sideways. A VLM reads it far better upright, so we
    detect the orientation and rotate the rendered PNG before sending it.

Detection uses Tesseract OSD (`tesseract --psm 0`, the binary ships on most boxes and
is the reliable 0/90/180/270 detector). It's best-effort: any failure -> return the
image unchanged (the VLM prompt still instructs orientation-awareness as a backstop).
Gate with config `extraction.deskew` (default on). Only worth running on pages we send
to the VLM that lack a good text layer.
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

_TESS = shutil.which("tesseract")
_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")


def _osd_rotate_degrees(png_bytes: bytes) -> int:
    """Degrees to rotate the image CLOCKWISE to make it upright (0/90/180/270), per
    Tesseract OSD. 0 if tesseract is unavailable or can't decide."""
    if not _TESS:
        return 0
    try:
        proc = subprocess.run(
            [_TESS, "stdin", "stdout", "--psm", "0"],
            input=png_bytes, capture_output=True, timeout=20,
        )
        m = _ROTATE_RE.search(proc.stdout.decode("utf-8", "ignore"))
        return int(m.group(1)) % 360 if m else 0
    except Exception as e:
        logger.debug("orientation: OSD failed (%s)", e)
        return 0


def correct_png_orientation(png_bytes: bytes, enabled: bool = True) -> bytes:
    """Return an upright PNG. Detects rotation via OSD and rotates if needed; returns the
    input unchanged on no rotation / any failure (never raises)."""
    if not enabled or not _TESS:
        return png_bytes
    deg = _osd_rotate_degrees(png_bytes)
    if deg == 0:
        return png_bytes
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        # Tesseract "Rotate" is the clockwise correction; PIL rotates counter-clockwise.
        img = img.rotate(-deg, expand=True)
        out = io.BytesIO()
        img.save(out, format="PNG")
        logger.info("orientation: rotated page %d° to upright", deg)
        return out.getvalue()
    except Exception as e:
        logger.debug("orientation: rotate failed (%s)", e)
        return png_bytes


def upright_png(page, dpi: int, config: dict | None = None) -> bytes:
    """Render a fitz page to PNG (fitz applies any /Rotate), then OSD-correct a sideways
    SCAN. Used by the VLM transcription paths so rotated invoices/scans read upright."""
    enabled = ((config or {}).get("extraction") or {}).get("deskew", True)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    # Only spend OSD on pages without a usable text layer (digital pages are already
    # upright via /Rotate and have correct text); cheap guard.
    try:
        if len((page.get_text() or "").strip()) > 20:
            return png
    except Exception:
        pass
    return correct_png_orientation(png, enabled=enabled)
