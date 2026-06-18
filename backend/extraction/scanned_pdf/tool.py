"""Tool for extracting content from scanned PDFs using OCR + YOLO.

OCR runs in an ISOLATED subprocess (see ocr_worker): the native OCR stack
(Paddle / Torch-Surya / llama-server) is unstable when sharing the backend
process with the embedder and vision threads on macOS. Spawning a fresh process
per scanned doc keeps those crashes (fork aborts, OpenMP segfaults) out of the
backend entirely — worst case the child dies and we record an error.
"""
import logging
import multiprocessing as mp
import queue as _queue
import time

from backend.core.tool import Tool, PipelineState
from backend.extraction.scanned_pdf.ocr_worker import run_scanned_ocr

logger = logging.getLogger(__name__)


class ScannedPDFTool(Tool):
    """OCR + layout extraction for scanned PDFs, isolated in a subprocess."""

    name = "scanned_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")
        timeout_s = float((config.get("ocr") or {}).get("subprocess_timeout_s", 1800))

        # 'spawn' (not fork): a fresh interpreter that loads the OCR libs cleanly,
        # with none of the parent's torch/threads inherited.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=run_scanned_ocr, args=(pdf_path, doc_id, config, q), daemon=True
        )
        proc.start()
        logger.info("scanned OCR subprocess started (pid=%s, timeout=%ss)", proc.pid, timeout_s)

        result = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                result = q.get(timeout=2)
                break
            except _queue.Empty:
                if not proc.is_alive():
                    break  # child died (segfault/SIGKILL) without sending a result
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)

        errors = state.setdefault("errors", [])
        if result is None:
            # No result: either it crashed natively or blew the timeout. The
            # backend survives — that's the whole point of isolation.
            code = proc.exitcode
            msg = (f"scanned_pdf: OCR subprocess died (exitcode={code}) — likely a "
                   f"native crash" if code not in (None, 0)
                   else f"scanned_pdf: OCR subprocess timed out after {timeout_s}s")
            logger.error(msg)
            errors.append(msg)
            state["blocks"] = []
            state["page_profiles"] = []
            return state

        status, a, b = result
        if status == "ok":
            state["blocks"] = a
            state["page_profiles"] = b
            logger.info("scanned OCR subprocess ok (%d blocks, %d page profiles)", len(a), len(b))
        else:
            logger.error("scanned OCR subprocess error: %s", a)
            errors.append(f"scanned_pdf: OCR failed in subprocess: {a.splitlines()[0] if a else a}")
            state["blocks"] = []
            state["page_profiles"] = []
        return state
