"""PaddleOCR PP-StructureV3 inference server — drop-in replacement for unlimited_ocr_server.py

Runs on port 8083 for parallel testing alongside existing Unlimited-OCR on 8082.
Same /infer API shape. Returns clean pipe-separated text (no det tags).

Expected speed: ~1-3s/page on T4 vs ~56s/page for Unlimited-OCR.

EC2 Setup (separate venv to avoid torch/paddle conflicts):
    python3.11 -m venv ~/paddleocr_test
    source ~/paddleocr_test/bin/activate
    pip install paddlepaddle-gpu paddleocr fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

Run:
    UNLIMITED_OCR_API_KEY=<same-key> python paddleocr_server.py
    # or via systemd: see paddleocr.service

Switch pipeline to test: in .env change UNLIMITED_OCR_ENDPOINT to http://100.49.250.232:8083/infer
"""

import io
import logging
import os
import time
from html.parser import HTMLParser

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddleocr_server")

API_KEY = os.environ.get("UNLIMITED_OCR_API_KEY")
if not API_KEY:
    logger.warning("UNLIMITED_OCR_API_KEY not set — /infer is UNAUTHENTICATED")

app = FastAPI()
_state: dict = {}


# ── HTML table → pipe-separated plain text (mirrors unlimited_ocr.py) ─────────

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._in_cell = True

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append(" ".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._row:
            self._rows.append(self._row)

    def rows(self):
        return self._rows


def _html_to_pipe(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    return "\n".join(" | ".join(r) for r in p.rows() if any(c.strip() for c in r))


def _regions_to_text(regions: list) -> str:
    """Convert PPStructure result list to clean plain text, tables as pipe rows."""
    parts: list[str] = []
    # Sort top-to-bottom for reading order
    for region in sorted(regions, key=lambda x: x.get("bbox", [0, 0, 0, 0])[1]):
        rtype = region.get("type", "").lower()
        res = region.get("res", {})

        if rtype == "table":
            html = ""
            if isinstance(res, dict):
                html = res.get("html", "")
            elif isinstance(res, str):
                html = res
            t = _html_to_pipe(html)
            if t:
                parts.append(t)

        elif rtype in ("text", "title", "reference", "figure_caption", "header", "footer"):
            if isinstance(res, list):
                lines: list[str] = []
                for item in res:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    text_conf = item[1]
                    if isinstance(text_conf, (list, tuple)) and text_conf:
                        txt = text_conf[0]
                        if isinstance(txt, str) and txt.strip():
                            lines.append(txt)
                if lines:
                    parts.append("\n".join(lines))

    return "\n".join(parts)


# ── startup: load model ────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    logger.info("Loading PaddleOCR PP-Structure...")
    t0 = time.time()
    try:
        from paddleocr import PPStructure
        # table=True: enable table structure recognition
        # layout=True: enable layout analysis (text/table/figure blocks)
        import paddle
        gpu_available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        _state["engine"] = PPStructure(
            show_log=False,
            use_gpu=gpu_available,
            lang="en",
            table=True,
            layout=True,
        )
        logger.info("PPStructure loaded (gpu=%s)", gpu_available)
        logger.info("PPStructure loaded in %.1fs", time.time() - t0)
    except Exception as e:
        logger.error("PPStructure load failed: %s — falling back to plain PaddleOCR", e)
        from paddleocr import PaddleOCR
        import paddle
        gpu_available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        _state["engine"] = PaddleOCR(
            use_gpu=gpu_available,
            lang="en",
            use_angle_cls=True,
            show_log=False,
        )
        _state["plain_mode"] = True
        logger.info("PaddleOCR (plain) loaded in %.1fs", time.time() - t0)


# ── /infer endpoint (same shape as unlimited_ocr_server.py) ───────────────────

@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(None),
    crop_mode: bool = Form(True),   # accepted for API compatibility, ignored
    invoice_mode: bool = Form(False),  # accepted for API compatibility, ignored
):
    """Drop-in for unlimited_ocr_server /infer. Returns {text, elapsed_s}."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing/invalid X-API-Key")
    if "engine" not in _state:
        raise HTTPException(503, "model still loading")

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image upload")

    # PIL → numpy BGR for PaddleOCR
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    img_array = np.array(pil_img)[:, :, ::-1]  # RGB→BGR

    t0 = time.time()
    try:
        if _state.get("plain_mode"):
            # Fallback: plain OCR, no table structure
            result = _state["engine"].ocr(img_array, cls=True)
            lines: list[str] = []
            for page in (result or []):
                for item in (page or []):
                    if item and len(item) >= 2 and item[1] and item[1][0].strip():
                        lines.append(item[1][0])
            text = "\n".join(lines)
        else:
            result = _state["engine"](img_array)
            text = _regions_to_text(result or [])
    except Exception as exc:
        logger.exception("infer error")
        raise HTTPException(500, str(exc)) from exc

    elapsed = time.time() - t0
    logger.info("infer: %.2fs, chars=%d", elapsed, len(text))
    return {"text": text, "elapsed_s": elapsed}


@app.get("/health")
def health():
    return {"status": "ok", "engine": "paddleocr", "loaded": "engine" in _state}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
