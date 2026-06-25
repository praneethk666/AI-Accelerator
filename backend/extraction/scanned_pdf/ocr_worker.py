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

Per-page streaming protocol (child → parent via out_queue):
  ("page",       page_num, [block_dict, ...])  — page completed successfully
  ("page_error", page_num, reason_str)          — page skipped (logged, not fatal)
  ("profiles",   [profile_dict, ...])           — page_profile results
  ("ok",         None, None)                    — all done
  ("err",        message, None)                 — fatal child error
"""
from __future__ import annotations


def run_scanned_ocr(pdf_path: str, document_id: str, config: dict, out_queue) -> None:
    """Child-process entrypoint.

    Streams per-page results on out_queue as they complete so the parent can
    accumulate partial results even if the child dies mid-document. Final
    messages are ("profiles", ...) then ("ok", None, None) on success, or
    ("err", msg, None) on a fatal setup error.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [OCR-worker] %(levelname)s %(name)s: %(message)s",
    )
    _wlog = _logging.getLogger(__name__)

    try:
        import surya.inference.backends.spawn as _surya_spawn
        _surya_spawn._stop_process = lambda pid, name: None
        _wlog.debug("surya._stop_process patched to no-op")
    except Exception:
        pass

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
            _wlog.info("Warming Surya on main thread…")
            warm_surya()
            _wlog.info("Surya warm-up complete")

        min_visual_area = config.get("min_visual_area", 50000)

        # extract_scanned streams ("page", n, blocks) / ("page_error", n, reason)
        # onto out_queue as each page finishes. The parent collects them in real
        # time, so even if we die later, pages 1-N are already delivered.
        _wlog.info("Starting extract_scanned for %s", pdf_path)
        extract_scanned(
            pdf_path=pdf_path,
            document_id=document_id,
            config=config,
            min_visual_area=min_visual_area,
            out_queue=out_queue,
        )
        _wlog.info("extract_scanned complete; running page_profile…")

        profiles = page_profile(pdf_path)
        out_queue.put(("profiles", as_dicts(profiles), None))
        out_queue.put(("ok", None, None))
        _wlog.info("OCR worker finished successfully")

    except Exception as exc:
        import traceback
        msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _wlog.error("Fatal OCR worker error: %s", msg)
        out_queue.put(("err", msg, None))