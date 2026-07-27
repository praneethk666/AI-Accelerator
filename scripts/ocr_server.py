"""Combined OCR inference server: Unlimited-OCR (structured document parsing) +
Surya (fast raw text-line OCR), one process, one /infer endpoint, `engine` picks
which model runs per request. Models are LAZY-LOADED per engine on first use, NOT
both loaded eagerly at startup — real finding, 27-Jul, live on a real 105-page
extraction run: with both loaded (~10GB combined baseline, higher than the ~8.3GB
originally measured on small test crops), a few large/dense table crops spiked
Unlimited-OCR's attention computation enough to hit CUDA OOM (13.7GB used, only
864MB free) and fail — while this project's pipeline NEVER actually calls
engine="surya" at all (that's the OTHER project, NEI). Lazy-loading means Surya's
~1.7GB is never allocated unless something actually requests it, permanently
freeing that headroom for Unlimited-OCR here. Replaces the two separate
single-engine servers (unlimited_ocr_server.py, surya_server.py).

Real, validated finding (24-27 Jul) driving which engine to pick:
  - engine="unlimited_ocr": correct reading order + structured <table> HTML with
    proper rowspan denormalization (every row carries its own identifying values,
    not left blank relying on the row above). Slower, ~26-34s/page. Use where
    downstream processing depends on real table structure (AI-Accelerator RAG
    chunking).
  - engine="surya": fast, ~11s/page (3x faster). But returns UNORDERED flat text
    lines — no table/layout structure at all. Validated live: on a real dense
    alarm-code table, row/column order came out scrambled (a Remarks-column value
    printed BEFORE the alarm name/code it belongs to). Use only where downstream
    processing does its own structured extraction from raw text regardless of the
    OCR engine (NEI invoices).

invoice_mode=True (unlimited_ocr only) applies a shorter generation budget suited
to invoice-sized output (~500-1500 tokens vs a full manual page) AND now actually
forces crop_mode=False as part of the same preset — the previous split-script
version left this to the caller to remember, which is exactly the setting that
avoids the tile-boundary data loss we found on dense invoice tables.

Setup on the GPU box (merges both venvs' deps):
    pip install torch==2.10.0 torchvision==0.25.0 transformers==4.57.1 \
        Pillow==12.1.1 einops==0.8.2 addict==2.4.0 easydict==1.13 \
        pymupdf==1.27.2.2 psutil==7.2.2 matplotlib fastapi "uvicorn[standard]" \
        python-multipart surya-ocr==0.17.1

Run via systemd (see ocr-server.service). Port 8008.
"""
import io
import logging
import os
import tempfile
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_server")

UNLIMITED_OCR_MODEL = "baidu/Unlimited-OCR"
# Shared secret across both engines — nginx puts this behind a public EC2 IP.
API_KEY = os.environ.get("UNLIMITED_OCR_API_KEY")
if not API_KEY:
    logger.warning("UNLIMITED_OCR_API_KEY not set — /infer is UNAUTHENTICATED. "
                    "Fine for localhost-only testing, not fine once nginx exposes this.")

app = FastAPI()
_state: dict = {}
_uo_lock = threading.Lock()
_surya_lock = threading.Lock()


def _ensure_unlimited_ocr_loaded() -> None:
    """Load Unlimited-OCR on first use only (double-checked locking so concurrent
    first-requests can't both start loading it)."""
    if "uo_model" in _state:
        return
    with _uo_lock:
        if "uo_model" in _state:  # someone else finished loading while we waited
            return
        t0 = time.time()
        logger.info("Loading %s (first use)...", UNLIMITED_OCR_MODEL)
        _state["uo_tokenizer"] = AutoTokenizer.from_pretrained(
            UNLIMITED_OCR_MODEL, trust_remote_code=True)
        # Try Flash Attention 2 first (20-40% speedup on Ampere/T4); fall back silently.
        for fa2 in (True, False):
            try:
                kwargs = dict(trust_remote_code=True, use_safetensors=True, dtype=torch.bfloat16)
                if fa2:
                    kwargs["attn_implementation"] = "flash_attention_2"
                _state["uo_model"] = AutoModel.from_pretrained(
                    UNLIMITED_OCR_MODEL, **kwargs).eval().cuda()
                logger.info("Unlimited-OCR loaded in %.1fs (flash_attn_2=%s)", time.time() - t0, fa2)
                break
            except Exception as e:
                if not fa2:
                    raise
                logger.warning("flash_attn_2 unavailable (%s), retrying without", e)


def _ensure_surya_loaded() -> None:
    """Load Surya on first use only — NEVER triggered by this project's own
    pipeline (which always sends engine="unlimited_ocr"), only if something else
    (e.g. NEI) actually requests engine="surya" against this same server."""
    if "surya_rec" in _state:
        return
    with _surya_lock:
        if "surya_rec" in _state:
            return
        t0 = time.time()
        logger.info("Loading Surya OCR v0.17 (first use)...")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor, FoundationPredictor
        _state["surya_det"] = DetectionPredictor()
        foundation = FoundationPredictor()
        _state["surya_rec"] = RecognitionPredictor(foundation_predictor=foundation)
        logger.info("Surya loaded in %.1fs", time.time() - t0)


def _infer_unlimited_ocr(tmp_path: str, out_dir: str, crop_mode: bool, invoice_mode: bool) -> str:
    if invoice_mode:
        max_length, no_repeat_ngram, ngram_window = 4096, 15, 50
        crop_mode = False  # part of the invoice preset, not left to caller discipline
    else:
        max_length, no_repeat_ngram, ngram_window = 32768, 30, 100
    # infer()'s return value is None by default (confirmed against the real source,
    # modeling_unlimitedocr.py on HF Hub) — eval_mode=True takes an early-return
    # path that decodes and returns the text directly, compatible with crop_mode.
    result = _state["uo_model"].infer(
        _state["uo_tokenizer"],
        prompt="<image>document parsing.",
        image_file=tmp_path,
        output_path=out_dir,
        base_size=1024,
        image_size=640 if crop_mode else 1024,
        crop_mode=crop_mode,
        eval_mode=True,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram,
        ngram_window=ngram_window,
        temperature=0.1,
    )
    return (result or "").strip()


def _infer_surya(pil_img: Image.Image) -> str:
    results = _state["surya_rec"]([pil_img], det_predictor=_state["surya_det"])
    lines = [line.text for line in results[0].text_lines if line.text.strip()]
    return "\n".join(lines)


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(None),
    engine: str = Form("unlimited_ocr"),
    crop_mode: bool = Form(True),
    invoice_mode: bool = Form(False),
):
    """engine: "unlimited_ocr" (structured <table> HTML + correct reading order,
    ~26-34s/page) or "surya" (flat unordered text lines, no table structure,
    ~11s/page — validated to scramble row/column order on dense real tables).
    Whichever engine is requested is loaded on first use, not eagerly at startup —
    see _ensure_unlimited_ocr_loaded/_ensure_surya_loaded."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing/invalid X-API-Key")
    if engine not in ("unlimited_ocr", "surya"):
        raise HTTPException(400, f"unknown engine {engine!r}, expected 'unlimited_ocr' or 'surya'")
    if engine == "surya":
        _ensure_surya_loaded()
    else:
        _ensure_unlimited_ocr_loaded()

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image upload")

    t0 = time.time()
    if engine == "surya":
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        text = _infer_surya(pil_img)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        out_dir = tempfile.mkdtemp()
        try:
            text = _infer_unlimited_ocr(tmp_path, out_dir, crop_mode, invoice_mode)
        except torch.cuda.OutOfMemoryError as e:
            # Real, confirmed failure mode (27-Jul, live on the servo manual): some
            # large/dense table crops need more peak memory than this T4 has free
            # under crop_mode=True's patch-tiling path. A plain 500 gives the client
            # no way to tell "genuinely too large" apart from any other server error,
            # so it can only fall back straight to the paid VLM call. Returning a
            # distinguishable 503 lets the client retry once with crop_mode=False
            # (a different, less memory-hungry codepath) before paying for VLM.
            logger.warning("infer: CUDA OOM (engine=%s crop_mode=%s): %s", engine, crop_mode, e)
            raise HTTPException(503, "cuda_oom") from e
        finally:
            os.unlink(tmp_path)
            torch.cuda.empty_cache()  # release reserved-but-unallocated memory

    elapsed = time.time() - t0
    logger.info("infer: engine=%s %.1fs output_len=%d", engine, elapsed, len(text))
    return {"text": text, "elapsed_s": elapsed, "engine": engine}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "unlimited_ocr_loaded": "uo_model" in _state,
        "surya_loaded": "surya_rec" in _state,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
