"""Isolated OCR worker — runs in a SEPARATE process (multiprocessing 'spawn').

Why a subprocess: scanned-page OCR pulls in heavy native libraries (PaddlePaddle,
PyTorch via Surya + DocLayout-YOLO, and llama-server for Surya). In the main
backend process — which already holds the PyTorch nomic embedder and runs vision
in worker threads — these collide on macOS: a fork from a worker thread aborts
(os_unfair_lock), and a second OpenMP runtime (Paddle vs Torch) segfaults. Running
OCR in its own freshly-spawned process gives each its own address space, so none
of that can reach (let alone crash) the backend. If the child dies, the parent
just records an error and the doc fails gracefully.

The target function runs as the child's MAIN thread, so warming Surya here (which
fork+execs llama-server) is safe. extract_scanned + page_profile then share the
per-page OCR cache within this process.
"""
from __future__ import annotations


def run_scanned_ocr(pdf_path: str, document_id: str, config: dict, out_queue) -> None:
    """Child-process entrypoint. Puts ("ok", blocks, profiles) or ("err", msg) on
    out_queue. Must stay top-level (picklable) for spawn."""
    try:
        from backend.extraction.scanned_pdf.scanned import (
            extract_scanned, set_ocr_engine, set_surya_timeout,
        )
        from backend.extraction.page_profile import page_profile
        from backend.core.schemas import as_dicts

        ocr_cfg = config.get("ocr") or {}
        engine = (ocr_cfg.get("engine") or "surya").lower()
        set_ocr_engine(engine)
        set_surya_timeout(ocr_cfg.get("surya_timeout_s"))

        # Warm Surya on THIS (the child's main) thread before any threaded OCR, so
        # its llama-server fork+exec is safe. Per-page calls then reuse the server.
        if engine == "surya":
            from backend.extraction.scanned_pdf.ocr_backends import warm_surya
            warm_surya()

        min_visual_area = config.get("min_visual_area", 50000)
        blocks = extract_scanned(pdf_path, document_id, min_visual_area=min_visual_area)
        profiles = page_profile(pdf_path)

        out_queue.put(("ok", as_dicts(blocks), as_dicts(profiles)))
    except Exception as exc:  # report, don't crash silently — parent surfaces it
        import traceback
        out_queue.put(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
