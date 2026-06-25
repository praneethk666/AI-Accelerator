"""Tool for extracting content from scanned PDFs using OCR + YOLO.

OCR runs in an ISOLATED subprocess (see ocr_worker): the native OCR stack
(Paddle / Torch-Surya / llama-server) is unstable when sharing the backend
process with the embedder and vision threads on macOS. Spawning a fresh process
per scanned doc keeps those crashes (fork aborts, OpenMP segfaults) out of the
backend entirely.

Per-page streaming protocol (child → parent via Queue):
  ("page",       page_num, [block_dict, ...])  — page done, collect immediately
  ("page_error", page_num, reason_str)          — page skipped, log and continue
  ("profiles",   [profile_dict, ...], None)     — page_profile results
  ("ok",         None, None)                    — child finished cleanly
  ("err",        message, None)                 — fatal child error

The parent collects ("page", ...) messages in real time, so even if the child
dies mid-document (SIGKILL from OOM, SIGTERM from timeout), all pages processed
before the death are already in state["blocks"]. The document is never all-or-nothing.
"""
import logging
import multiprocessing as mp
import queue as _queue
import time

from backend.core.tool import Tool, PipelineState
from backend.extraction.scanned_pdf.ocr_worker import run_scanned_ocr

logger = logging.getLogger(__name__)


class ScannedPDFTool(Tool):
    """OCR + layout extraction for scanned PDFs, isolated in a subprocess.

    Streams per-page results from the child so partial extraction succeeds even
    when the subprocess dies before finishing all pages.
    """

    name = "scanned_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")
        timeout_s = float((config.get("ocr") or {}).get("subprocess_timeout_s", 1800))

        # 'spawn' (not fork): a fresh interpreter that loads OCR libs cleanly,
        # with none of the parent's torch/threads inherited.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=run_scanned_ocr,
            args=(pdf_path, doc_id, config, q),
            daemon=True,
        )
        proc.start()
        logger.info(
            "scanned OCR subprocess started (pid=%s, timeout=%ss)", proc.pid, timeout_s
        )

        # ── Streaming collection ───────────────────────────────────────────────
        # Accumulate blocks page-by-page from the queue. This means we keep
        # whatever the child finished before dying — never discard partial work.
        all_blocks: list = []
        all_profiles: list = []
        errors = state.setdefault("errors", [])
        page_errors: list = []
        child_done = False          # saw ("ok") or ("err")
        fatal_error: str | None = None

        deadline = time.time() + timeout_s

        while time.time() < deadline:
            # Drain all currently available messages before checking liveness.
            try:
                msg = q.get(timeout=2)
            except _queue.Empty:
                if not proc.is_alive():
                    # Child exited without sending ("ok") or ("err") — native crash.
                    break
                continue

            kind = msg[0]

            if kind == "page":
                _, page_num, block_dicts = msg
                all_blocks.extend(block_dicts)
                logger.info(
                    "Page %d received: %d blocks (running total: %d)",
                    page_num, len(block_dicts), len(all_blocks),
                )

            elif kind == "page_error":
                _, page_num, reason = msg
                page_errors.append(f"page {page_num}: {reason}")
                logger.warning("Page %d skipped in subprocess: %s", page_num, reason)

            elif kind == "profiles":
                _, profile_dicts, _ = msg
                all_profiles = profile_dicts
                logger.info("Page profiles received (%d pages)", len(all_profiles))

            elif kind == "ok":
                child_done = True
                logger.info("scanned OCR subprocess finished cleanly")
                break

            elif kind == "err":
                _, fatal_error, _ = msg
                child_done = True   # child will exit after sending this
                logger.error("scanned OCR subprocess fatal error: %s", fatal_error)
                break

        # ── Cleanup ───────────────────────────────────────────────────────────
        proc.join(timeout=35)
        if proc.is_alive():
            logger.warning("subprocess still alive after 35 s join — terminating")
            proc.terminate()
            proc.join(5)

        exit_code = proc.exitcode

        # ── Result assembly ───────────────────────────────────────────────────
        if fatal_error:
            short = fatal_error.splitlines()[0] if fatal_error else fatal_error
            errors.append(f"scanned_pdf: OCR failed in subprocess: {short}")

        if not child_done:
            # Subprocess died (crash / OOM) or timed out before sending ("ok").
            if exit_code not in (None, 0):
                msg = (
                    f"scanned_pdf: OCR subprocess died (exitcode={exit_code}) — "
                    f"likely a native crash. Partial results: {len(all_blocks)} blocks "
                    f"from pages processed before the crash."
                )
            else:
                msg = (
                    f"scanned_pdf: OCR subprocess timed out after {timeout_s}s. "
                    f"Partial results: {len(all_blocks)} blocks collected."
                )
            logger.error(msg)
            errors.append(msg)

        if page_errors:
            errors.append(
                f"scanned_pdf: {len(page_errors)} page(s) skipped: "
                + "; ".join(page_errors)
            )

        state["blocks"] = all_blocks
        state["page_profiles"] = all_profiles

        logger.info(
            "scanned_pdf done: %d blocks, %d page profiles, %d errors",
            len(all_blocks), len(all_profiles), len(errors),
        )
        return state