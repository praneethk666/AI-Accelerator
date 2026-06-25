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

def _patch_surya_stop_process_early() -> None:
    try:
        import surya.inference.backends.spawn as _spawn_mod
        _spawn_mod._stop_process = lambda pid, name: None
        logger.debug("surya._stop_process patched to no-op at import time")
    except Exception:
        pass  # surya not installed or different version — safe to ignore
 
_patch_surya_stop_process_early()

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
    _patch_surya_stop_process()

def _patch_surya_stop_process() -> None:
    """Monkey-patch surya's _stop_process to not block on exit.

    Root cause
    ----------
    surya/inference/backends/spawn.py registers an atexit _cleanup closure that
    calls _stop_process(pid, backend). _stop_process sends SIGTERM then loops
    20 × time.sleep(0.5) = 10 s waiting for llama-server to die. When CTRL+C
    arrives while that sleep is running Python prints:

        Exception ignored in atexit callback: KeyboardInterrupt

    This happens in EVERY process that called warm_surya() — both the main
    FastAPI process and the OCR subprocess.

    Why gc/atexit.unregister approaches fail
    -----------------------------------------
    _cleanup is a closure created fresh inside attach_or_spawn(). It gets
    registered via atexit.register(_cleanup) and there is no stored reference
    to pass to atexit.unregister(). The gc walk finds the object but
    atexit.unregister() uses __eq__ not identity, and function equality in
    CPython falls back to identity — so unregister only works if you pass the
    exact same object gc found, which the test above confirmed does work BUT
    only when warm_surya() is called in that process. If Surya attaches to an
    existing llama-server (sentinel file present from a prior run), a NEW
    _cleanup is registered every attach — our gc walk misses it because it
    runs before the new attach.

    The reliable fix
    ----------------
    Patch _stop_process itself in the spawn module to a no-op. _cleanup still
    runs at exit (we can't stop that without clearing all atexit handlers), but
    it calls our patched _stop_process which returns instantly instead of
    sleeping 10 s. No KeyboardInterrupt, no delay, no noise.

    The OS automatically sends SIGHUP to llama-server (child process) when the
    parent exits, so llama-server is cleaned up regardless.
    """
    try:
        import surya.inference.backends.spawn as _spawn_mod

        def _instant_stop(pid: int, name: str) -> None:
            """No-op replacement: OS reaps the child process on parent exit."""
            pass

        _spawn_mod._stop_process = _instant_stop
        logger.info(
            "Patched surya._stop_process → no-op "
            "(llama-server will be reaped by OS on exit)"
        )
    except Exception as e:
        logger.debug("Could not patch surya._stop_process: %s", e)

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
