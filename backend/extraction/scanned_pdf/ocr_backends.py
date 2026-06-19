"""Pluggable OCR backends for scanned pages.

Two engines behind one interface (config: ocr.engine = "surya" | "paddle"):

  surya  — Surya (torch-based). ONE rec(full_page=True) call returns both the text
           AND the layout (figure/table/picture regions), replacing PaddleOCR + the
           DocLayout-YOLO/contour detector. Better layout + 90+ languages, and
           torch-based so no paddle<->torch allocator conflict. Needs the
           `llama-server` binary (brew install llama.cpp) for its foundation model.
  paddle — PaddleOCR text + DocLayout-YOLO/contour regions (the original path,
           kept as a fallback).

Each backend exposes surya_page(pil) / paddle is handled in scanned.py. The result
is a SuryaPage(text, text_boxes, regions) that scanned.py adapts into blocks.
"""
from __future__ import annotations

import html as _html
import logging
import re
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Platform matters for OCR. Surya spawns llama-server via fork+exec; on macOS a
# fork from a non-main thread aborts the process (os_unfair_lock SIGKILL), so the
# recognizer MUST be constructed on the main thread at startup. On Linux a
# fork-from-thread is tolerated, so pre-warming there is only a cold-start
# optimization, not a correctness requirement.
IS_MACOS = sys.platform == "darwin"

# Surya layout labels that are visual (sent to vision enrichment), not body text.
_VISUAL_LABELS = {"picture", "figure", "table", "chart", "form"}
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class SuryaPage:
    text: str = ""
    text_boxes: list = field(default_factory=list)   # [(x1,y1,x2,y2), ...]
    regions: list = field(default_factory=list)      # visual region bboxes


# ── Surya backend ───────────────────────────────────────────────────────────
_surya_rec = None


def _get_surya_recognizer():
    global _surya_rec
    if _surya_rec is None:
        # warm torch first (paddle<->torch ordering safety, harmless here)
        from backend.core.models import warm_up
        warm_up()
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        _surya_rec = RecognitionPredictor(SuryaInferenceManager())
        logger.info("Surya recognizer loaded (full_page layout+OCR)")
    return _surya_rec


def warm_surya() -> None:
    """Construct the Surya recognizer NOW, on the caller's thread.

    Surya launches llama-server via fork+exec when the recognizer is built. On
    macOS, forking from a worker thread aborts the process (os_unfair_lock
    SIGKILL), so this MUST be called at startup on the MAIN thread — never lazily
    from an ingestion worker thread. After this, per-page calls only run inference
    against the already-spawned server (no further fork). On Linux this is just a
    cold-start optimization (see IS_MACOS)."""
    _get_surya_recognizer()


def surya_page(pil_image) -> SuryaPage:
    """Run Surya full-page recognition: text lines + visual regions in one pass."""
    rec = _get_surya_recognizer()
    page = rec([pil_image], full_page=True)[0]

    out = SuryaPage()
    text_parts = []
    for block in page.blocks:
        bbox = _poly_to_bbox(block.polygon)
        if bbox is None:
            continue
        label = (block.label or "").lower()
        if label in _VISUAL_LABELS:
            out.regions.append(bbox)
            continue
        txt = _html_to_text(block.html or "")
        if txt:
            text_parts.append(txt)
            out.text_boxes.append(bbox)
    out.text = "\n".join(text_parts)
    return out


def _html_to_text(s: str) -> str:
    """Surya returns recognized text as HTML — strip tags to plain text."""
    text = _html.unescape(_TAG_RE.sub(" ", s or ""))
    return re.sub(r"\s+", " ", text).strip()


def _poly_to_bbox(polygon):
    """[[x,y],...] corner points -> (x1,y1,x2,y2). None if unusable."""
    try:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    except Exception:
        return None
