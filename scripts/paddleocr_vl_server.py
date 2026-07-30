"""Standalone PaddleOCR-VL-1.6 inference server — deliberately a SEPARATE
process/port from scripts/ocr_server.py (Unlimited-OCR + Surya), not a third
engine bolted onto it.

Why separate: real finding, live-attempted, 28-Jul — torch 2.10.0 (ocr_server.py's
stack) and paddlepaddle-gpu 3.3.1 pin mutually exclusive nvidia-*-cu12 versions
(12.8-family vs 12.6-family, confirmed via pip's own conflict report both
directions) with zero overlap. Installing both in the SAME venv leaves one of
the two unable to even import — confirmed live: torch's own
`libtorch_cuda.so: undefined symbol: ncclCommShrink` the moment paddlepaddle-gpu
downgraded the shared nvidia-nccl-cu12 package underneath it. This is a hard
packaging conflict, not a config/effort gap — no amount of pinning fixes it,
since both packages pin exact (`==`) versions of the same shared libraries.

Why this engine at all: live bake-off (27-Jul) on the 3 hardest real tables
Unlimited-OCR's own OOM-retry-ladder + row-split fallback still couldn't fully
solve on this T4 — 2 that OOM identically at every base_size (a decoder/output-
length bottleneck, not input-resolution) and 1 where Unlimited-OCR silently
DROPPED a row, cascading a 6-row label/value misalignment past the
fully-empty-column safety net. PaddleOCR-VL-1.6's two-stage pipeline
(PP-DocLayoutV3 layout localization, then PaddleOCR-VL-1.6-0.9B element
recognition WITHIN already-fixed regions) transcribed all three PERFECTLY.

Setup on the GPU box (own venv, NOT shared with ocr_server.py's):
    python3 -m venv ~/paddleocrvl_test
    ~/paddleocrvl_test/bin/pip install paddlepaddle-gpu==3.3.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    ~/paddleocrvl_test/bin/pip install paddleocr==3.7.0 paddlex==3.7.2 \
        fastapi "uvicorn[standard]" python-multipart
    # Needs `ccache` for first-run JIT kernel compilation, or it silently takes
    # 14+ minutes (100% CPU, 0% GPU, zero log progress — looks hung, isn't).
    # yum/dnf had no package on Amazon Linux 2023; installed via prebuilt binary
    # from ccache's GitHub releases instead.

Run via systemd (see paddleocr-vl-server.service). Port 8084.
"""
import gc
import logging
import os
import tempfile
import threading
import time

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddleocr_vl_server")

# Same shared secret convention as ocr_server.py — nginx puts this behind a
# public EC2 IP. Can reuse the same value as UNLIMITED_OCR_API_KEY or set a
# distinct one; either way it must match the client's local_api_key config.
API_KEY = os.environ.get("PADDLEOCR_VL_API_KEY")
if not API_KEY:
    logger.warning("PADDLEOCR_VL_API_KEY not set — /infer is UNAUTHENTICATED. "
                    "Fine for localhost-only testing, not fine once nginx exposes this.")

app = FastAPI()
_state: dict = {}
_lock = threading.Lock()


def _ensure_loaded() -> None:
    """Load PaddleOCR-VL-1.6 on first use only, not eagerly at startup — same
    lazy-load convention as ocr_server.py's engines."""
    if "pipeline" in _state:
        return
    with _lock:
        if "pipeline" in _state:
            return
        t0 = time.time()
        logger.info("Loading PaddleOCR-VL-1.6 (first use)...")
        from paddleocr import PaddleOCRVL
        _state["pipeline"] = PaddleOCRVL()
        logger.info("PaddleOCR-VL-1.6 loaded in %.1fs", time.time() - t0)


def _infer(tmp_path: str) -> str:
    """Real finding, 27-Jul: on the 3 hardest known real tables from this
    project's servo manual (2 that OOM Unlimited-OCR at every base_size, 1 with
    a silently-dropped row causing cascading misalignment), PaddleOCR-VL-1.6
    transcribed all three perfectly. Output is real HTML
    (<table>/<tr>/<td rowspan=..>), parsed client-side by the SAME
    _parse_html_table used for Unlimited-OCR's det-tagged tables — see
    backend/extraction/paddleocr_vl.py."""
    outputs = list(_state["pipeline"].predict(tmp_path))
    if not outputs:
        return ""
    md = getattr(outputs[0], "markdown", None) or {}
    return (md.get("markdown_texts") or "").strip()


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing/invalid X-API-Key")
    _ensure_loaded()

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image upload")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
        tmp_path = f.name

    t0 = time.time()
    try:
        text = _infer(tmp_path)
    finally:
        os.unlink(tmp_path)
        # Same lesson as ocr_server.py's Unlimited-OCR path, 27-Jul: memory can
        # climb across a sequence of requests unless dead tensors are actually
        # released, not just left for the allocator to notice eventually.
        # gc.collect() first so paddle's own cache-clear has something real to
        # free (breaks lingering refs, e.g. held by a just-raised exception).
        gc.collect()
        import paddle
        paddle.device.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info("infer: %.1fs output_len=%d", elapsed, len(text))
    return {"text": text, "elapsed_s": elapsed, "engine": "paddleocr_vl"}


@app.get("/health")
def health():
    return {"status": "ok", "paddleocr_vl_loaded": "pipeline" in _state}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8084)
