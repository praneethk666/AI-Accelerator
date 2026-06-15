"""Pluggable OCR backends for scanned pages."""
from __future__ import annotations

import html as _html
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_VISUAL_LABELS = {"picture", "figure", "table", "chart", "form"}
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class SuryaPage:
    text: str = ""
    text_boxes: list = field(default_factory=list)
    regions: list = field(default_factory=list)


_surya_rec = None


def _get_surya_recognizer():
    global _surya_rec
    if _surya_rec is None:
        # Force the default (torch) recognition backend unless the user
        # has explicitly opted into llamacpp AND has the binary available.
        backend = os.environ.get("SURYA_BACKEND", "").lower()
        if backend == "transformers":
            import shutil
            if shutil.which("llama-server") is None and not os.environ.get("LLAMA_CPP_BINARY"):
                logger.warning(
                    "SURYA_BACKEND=transformers but llama-server not found; "
                    "falling back to default torch backend."
                )
                os.environ.pop("SURYA_BACKEND", None)

        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        _surya_rec = RecognitionPredictor(SuryaInferenceManager())
        logger.info("Surya recognizer loaded")
    return _surya_rec


def surya_page(pil_image) -> SuryaPage:
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
    text = _html.unescape(_TAG_RE.sub(" ", s or ""))
    return re.sub(r"\s+", " ", text).strip()


def _poly_to_bbox(polygon):
    try:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    except Exception:
        return None