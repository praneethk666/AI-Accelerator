"""Self-hosted PaddleOCR-VL-1.6 as an alternative per-table-crop transcription
engine, alongside Unlimited-OCR (backend/extraction/unlimited_ocr.py) — but
pointed at its OWN server process/port (scripts/paddleocr_vl_server.py), NOT
the shared scripts/ocr_server.py.

Why a separate server: real finding, live-attempted, 28-Jul — torch 2.10.0
(ocr_server.py's stack) and paddlepaddle-gpu 3.3.1 pin mutually exclusive
nvidia-*-cu12 versions (12.8-family vs 12.6-family) with zero overlap;
installing both in the SAME venv leaves one unable to even import (torch's own
`libtorch_cuda.so: undefined symbol: ncclCommShrink` once paddlepaddle-gpu
downgrades the shared nvidia-nccl-cu12 package underneath it). A hard
packaging conflict, not a config gap — hence its own process, own port
(8084), own systemd service.

Why this engine at all: live bake-off (27-Jul) on the 3 hardest real tables
Unlimited-OCR's own retry ladder (base_size ladder + row-split, see
unlimited_ocr.py) still couldn't fully solve on this T4 — page 2 (rowspan-
heavy safety table) and page 26 (18-row list table) both OOM identically at
every base_size (a decoder/output-length bottleneck, not input-resolution),
and page 18 (a 44-row/5-section spec table) came back from Unlimited-OCR with
an entire row ("Operating humidity") silently dropped, cascading a 6-row
label/value misalignment that _has_fully_empty_column's safety net can't
catch (every cell still has plausible text, just paired with the wrong
label). PaddleOCR-VL-1.6 transcribed all three of these PERFECTLY, live, on
the same GPU. Its two-stage architecture (PP-DocLayoutV3 layout localization,
then PaddleOCR-VL-1.6-0.9B element recognition WITHIN already-fixed regions)
is the likely reason: it avoids the documented single-stage-VLM weakness of
implicitly reconstructing row/column alignment from a long generated HTML
token stream, which is exactly where Unlimited-OCR's row drop happened.

Output format differs from Unlimited-OCR: this server's /infer response is
raw HTML for the table region directly (real <table>/<tr>/<td rowspan=..>
markup — no <|det|> wrapper, see unlimited_ocr.py's _DET_RE), so this module
feeds it straight to the SAME _parse_html_table used there — confirmed live
to handle PaddleOCR-VL's HTML shape unchanged, including inline style=
attributes it adds that Unlimited-OCR's own HTML doesn't.

Only wired into per-table-crop escalation so far (see
extraction.docling.local_table_ocr_engine in docling_extract.py) — the same
integration point unlimited_ocr's transcribe_table_local uses. Not (yet) an
option for vision_ocr.engine's whole-page rescue path.
"""
from __future__ import annotations

import logging

import httpx

from backend.extraction.unlimited_ocr import _parse_html_table

logger = logging.getLogger(__name__)


def _call_paddleocr_vl_server(png_bytes: bytes, cfg: dict) -> str:
    """POST an already-rendered/cropped PNG to scripts/paddleocr_vl_server.py
    and return its raw text (HTML for a table crop)."""
    endpoint = cfg.get("local_paddleocr_vl_endpoint")
    if not endpoint:
        raise ValueError(
            "extraction.docling.local_table_ocr_engine is 'paddleocr_vl' but "
            "vision_ocr.local_paddleocr_vl_endpoint is not set (point it at "
            "scripts/paddleocr_vl_server.py running on the GPU box)")
    timeout = float(cfg.get("local_timeout_s", 90))
    headers = {}
    api_key = cfg.get("local_paddleocr_vl_api_key")
    if api_key:
        headers["X-API-Key"] = api_key
    resp = httpx.post(endpoint, files={"image": ("table.png", png_bytes, "image/png")},
                       headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["text"]


def transcribe_table_paddleocr_vl(png_bytes: bytes, config: dict) -> dict | None:
    """Send an already-cropped table-region PNG to the self-hosted PaddleOCR-VL
    server and return {headers, rows}.

    No OOM retry ladder here (unlike transcribe_table_local) — PaddleOCR-VL's
    two-stage pipeline hasn't shown the decoder-length-bound OOM Unlimited-OCR
    hits on long/dense tables in any of tonight's live testing (3/3 hardest
    known real cases succeeded on the first attempt). If that changes on a
    larger corpus, add a retry ladder here the same way transcribe_table_local
    does — nothing here assumes it will never be needed."""
    cfg = config.get("vision_ocr") or {}
    raw = _call_paddleocr_vl_server(png_bytes, cfg)
    table_data = _parse_html_table(raw)
    if table_data is None:
        logger.warning("paddleocr_vl: no parseable <table> in response, discarding")
    return table_data
