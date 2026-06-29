# OCR Engine Comparison — Surya vs PaddleOCR (scanned PDFs)

**Date:** 2026-06-17
**Test doc:** `argo__1618828868_1-general-info.pdf` (22-page ARGO service manual)
rasterized to an image-only PDF (`scripts/make_scanned.py`, 200 DPI) so it
classifies as `scanned` and exercises the OCR path. Both engines were run on the
**same** input via `scripts/ocr_compare.py` (each in its own process — Surya and
PaddleOCR can't share one cleanly).

Engine is selected by `config/global.yaml -> ocr.engine` (`surya` | `paddle`),
honored in `backend/extraction/scanned_pdf/scanned.py`.

## Results

| Metric | Surya | PaddleOCR |
|---|---|---|
| Text captured (first 5 pp) | 1,607 chars | 1,581 chars (≈ tie) |
| Speed, content page (warm) | ~20–25 s/page | ~7 s/page |
| Full 22-page run | **stalled — 22 min at 0% CPU on cold start, killed** | 267 s, completed cleanly (~12 s/page) |
| Cover / stylized text | "SERVICE MANUAL 8x8 MODELS…" ✅ | "Aryoa SERVICE MANDAL…" ❌ (misread Argo/MANUAL) |
| Reading order & TOC leaders | cleaner, layout-faithful | minor word jumbling on multi-column / TOC |
| Visual regions flagged (5 pp) | 8 (conservative) | 19 (contour false-positives on text) |
| Text boxes (5 pp) | 26 (block-level) | 85 (line-level) |
| Dependencies | torch + **`llama-server`** (llama.cpp) running the surya-2 GGUF | `paddlepaddle` + `paddleocr` (+ DocLayout-YOLO for regions) |

Raw numbers: Surya 5-pg per-page secs `[21.5, 2.1, 25.1, 5.5, 19.7]`;
Paddle 5-pg `[26.5(warmup), 2.5, 7.5, 3.1, 7.2]`.

## Reading

- **Accuracy:** Surya edges it. It read the stylized cover correctly where Paddle
  produced "Aryoa SERVICE MANDAL", and its reading order on multi-column / TOC
  pages is cleaner. Both capture essentially the same *volume* of body text.
- **Layout:** Surya's one-pass layout is more conservative and accurate — it
  flagged 8 visual regions vs Paddle's 19, many of Paddle's being contour
  false-positives that would wrongly fan out to (paid) vision enrichment.
- **Speed:** Paddle is ~3× faster per content page and finished all 22 pages in
  ~4.5 min. Surya runs ~20–25 s per text page when warm.
- **Reliability (the deciding factor):** Surya's first full-document run **hung
  for 22 minutes at 0% CPU** — a cold-start stall waiting on `llama-server`. A
  warm `llama-server` runs fine, but `extract_scanned()` has **no per-page
  timeout**, so a Surya stall blocks the whole ingestion with no recovery.
  (`extract_ocr_text_and_boxes` only falls back to Paddle on an *exception* — a
  hang raises nothing.) Paddle is CPU-bound and always makes progress.

## Recommendation (shipped)

Surya stays the default for its better accuracy — but it is **only stable when
isolated**. In the main backend process (which holds the PyTorch nomic embedder
and runs vision in worker threads) the OCR stack crashed three ways on macOS:
cold-start hang, fork-from-thread abort (Surya spawning `llama-server`), and an
OpenMP segfault (PaddlePaddle's runtime vs Torch's). Those are coexistence bugs,
not Surya bugs — in a clean process Surya runs fine.

> **Superseded (2026-06):** the standalone `scanned_pdf` tool + `ocr_worker.py`
> subprocess described below were **removed** when `docling_pdf` became the single
> unified PDF extractor. OCR now runs **in-process inside `docling_pdf`** (Docling's
> own OCR + the Paddle/Surya engine in `scanned_pdf/scanned.py` for region text).
> The crash analysis below is kept for historical rationale; the isolation it argues
> for is no longer how the code is structured.

So OCR previously ran in an **isolated subprocess** (`scanned_pdf/ocr_worker.py`,
spawned by `scanned_pdf/tool.py`):
- A fresh `spawn` process per scanned doc — its own address space, so Paddle /
  Torch-Surya / `llama-server` can't collide with the backend's torch.
- The worker warms Surya on the child's **main thread** (safe fork), then runs
  `extract_scanned` + `page_profile`, returning plain dicts.
- The parent bounds it with `ocr.subprocess_timeout_s` and detects a child death
  via exit code. If the child crashes or times out, the parent records an error
  and the doc fails gracefully — **the backend never goes down**.
- A per-page `ocr.surya_timeout_s` still guards individual Surya stalls inside
  the child (falling back to Paddle there).

Verified: with the torch embedder loaded in the parent (the exact crash
condition), the isolated child OCR'd cleanly and the parent survived.

Net: Surya = better text; isolation = no native crashes in the backend. Switch
`ocr.engine` to `paddle` if you prefer raw throughput over accuracy — it runs in
the same isolated subprocess.
