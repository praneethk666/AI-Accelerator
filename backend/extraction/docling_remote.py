"""Remote Docling client — talks to docling_server.py on port 8083.

Drop-in replacement for extract_docling() when config sets
  extraction.docling.mode: remote

Returns the same list[dict] NormalizedBlock schema that the local extraction
returns.  Image_caption blocks (figures) are returned as placeholders with
bbox set and text="[figure]" — the caller does crop + VLM captioning exactly
as it would after local extract_docling().  Complex tables that need VLM/OCR
escalation are flagged with metadata.escalation_hint="vlm_or_local".

Config keys consumed (under extraction.docling):
  server_url:   full URL of the Docling server  (e.g. http://100.49.250.232:8083)
  server_key:   API key for the server          (env: DOCLING_API_KEY)
  table_source: "auto" | "docling" | "pymupdf"  (default: "auto")
  timeout_s:    request timeout in seconds       (default: 600)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import requests

logger = logging.getLogger(__name__)


def _caption_remote_figures(blocks: list[dict], pdf_path: str, document_id: str,
                            filename: str, config: dict, total_pages: int = 0) -> list[dict]:
    """Crop + caption every image_caption placeholder the server returned.

    Real bug found live, 3-Aug: the server only LOCATES figures (bbox, text
    literally "[figure]") -- this module's own docstring claims "the caller does
    crop + VLM captioning exactly as it would after local extract_docling()", but
    nothing ever actually did that. Local mode's captioning loop lives INSIDE
    extract_docling(), which remote mode bypasses entirely via the early return
    above; caption_deferred_figures() (called later in docling_pdf/tool.py) only
    picks up blocks explicitly flagged caption_deferred, which these never were.
    Net effect: every figure from a remote-extracted document stayed a bare,
    unusable "[figure]" placeholder forever -- no crop, no image, no caption.

    Fixed by reusing the exact same _figure_block() gate local mode uses,
    re-cropping from the local pdf_path (still available -- we just uploaded it)
    using the bbox the server already found. Pages with zero returned text (do_ocr
    is off server-side too) get caption_deferred=True instead of an immediate gate
    call, same as local mode's scanned-page handling -- caption_deferred_figures()
    already picks those up downstream, no extra wiring needed."""
    from backend.extraction.docling_pdf.docling_extract import _figure_block

    dcfg = (config.get("extraction") or {}).get("docling") or {}
    defer_ok = bool(dcfg.get("page_rescue", True))
    figure_mode = dcfg.get("figure_caption_mode", "eager")
    lazy_threshold = int(dcfg.get("defer_figures_above_pages", 250) or 250)
    size_lazy = figure_mode == "size_based" and total_pages > lazy_threshold

    page_text: dict = {}
    page_figures: dict = {}
    fig_indices: list[int] = []
    for i, b in enumerate(blocks):
        page_no = (b.get("source_ref") or {}).get("page")
        if b.get("type") in ("text", "heading"):
            t = (b.get("text") or "").strip()
            if t:
                page_text.setdefault(page_no, []).append(t)
        elif b.get("type") == "image_caption" and (b.get("source_ref") or {}).get("bbox"):
            page_figures.setdefault(page_no, []).append(i)
            # Skip figures that are ALREADY resolved (real caption text, not the
            # "[figure]" sentinel, and not still flagged deferred) -- matters on a
            # resumed chunked extraction, where cached blocks reloaded from a prior
            # run may already be fully captioned; re-running the gate on those would
            # just be a wasted VLM call, not a correctness issue, but avoidable.
            already_done = (b.get("text") not in (None, "", "[figure]")
                           and not (b.get("metadata") or {}).get("caption_deferred"))
            if not already_done:
                fig_indices.append(i)

    if not fig_indices:
        return blocks

    def _resolve(i: int):
        b = blocks[i]
        ref = b.get("source_ref") or {}
        page_no, bbox = ref.get("page"), ref.get("bbox")
        others = [blocks[j]["source_ref"]["bbox"] for j in page_figures.get(page_no, []) if j != i]
        no_text = not page_text.get(page_no)
        defer_reason = "scanned_no_text" if no_text else ("large_document_lazy" if size_lazy else None)
        defer = defer_ok and defer_reason is not None
        return i, _figure_block(pdf_path, document_id, page_no, filename, bbox,
                                page_text.get(page_no, []), config,
                                other_figure_bboxes=others, defer=defer,
                                defer_reason=defer_reason)

    workers = max(1, int((config.get("vision") or {}).get("max_concurrency", 1) or 1))
    results: dict[int, dict | None] = {}
    if workers > 1 and len(fig_indices) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from backend.core import usage as _usage
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_usage.copy_ctx().run, _resolve, i) for i in fig_indices]
            for f in futs:
                i, blk = f.result()
                results[i] = blk
    else:
        for i in fig_indices:
            i, blk = _resolve(i)
            results[i] = blk

    out = []
    for i, b in enumerate(blocks):
        if i in results:
            resolved = results[i]
            if resolved is not None:
                out.append(resolved)
            # None -> semantic gate said furniture (logo/banner/etc), drop it
        else:
            out.append(b)
    return out


def _post_extract(endpoint: str, headers: dict, pdf_bytes: bytes, filename: str,
                  document_id: str, table_source: str, timeout: int) -> dict:
    """One /extract HTTP call. Raises RuntimeError on any failure (timeout,
    connection, auth, non-2xx) -- unchanged error handling, just factored out
    so both the single-shot and chunked paths below share it."""
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            files={"pdf": (filename, pdf_bytes, "application/pdf")},
            data={
                "document_id":        document_id,
                "filename":           filename,
                "table_source":       table_source,
                "do_ocr":             "false",
                "do_table_structure": "true",
                "min_picture_pts":    "24.0",
            },
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Docling server timeout after {timeout}s for {filename}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Cannot reach Docling server at {endpoint}: {e}")

    if resp.status_code == 401:
        raise RuntimeError("Docling server: invalid API key (401)")
    if not resp.ok:
        raise RuntimeError(
            f"Docling server returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _pdf_page_count(pdf_path: str) -> int | None:
    """None if it can't be determined (missing file, corrupt, etc) -- caller
    falls back to the single-shot path in that case, same as before this ever
    existed."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


def _extract_page_range_bytes(pdf_path: str, start: int, end: int) -> bytes:
    """1-indexed, inclusive. A genuinely separate small PDF containing just
    these pages, built locally (no server round-trip) so each chunk upload is
    bounded regardless of the source document's total size."""
    import fitz
    src = fitz.open(pdf_path)
    try:
        sub = fitz.open()
        try:
            sub.insert_pdf(src, from_page=start - 1, to_page=end - 1)
            return sub.tobytes()
        finally:
            sub.close()
    finally:
        src.close()


def _already_checkpointed_blocks(document_id: str) -> list[dict]:
    """Blocks a PRIOR (possibly crashed/killed) run of this same document_id
    already checkpointed. Empty list on any DB failure -- resume is a nice-to-
    have, never a hard requirement (worst case: redo the chunk, no data lost)."""
    try:
        from backend.storage.postgres_store import PostgresStore
        pg = PostgresStore()
        try:
            return pg.get_blocks(document_id)
        finally:
            pg.close()
    except Exception:
        return []


def extract_docling_remote(
    pdf_path: str,
    document_id: str,
    config: dict,
    table_source: str = "auto",
    report: dict | None = None,
    on_page: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Send *pdf_path* to the Docling server and return NormalizedBlocks.

    Mirrors the signature of extract_docling() so callers can swap without
    changing their call sites.

    Parameters
    ----------
    pdf_path:     local path to the PDF file to extract
    document_id:  document identifier stamped on every returned block
    config:       project config dict (global.yaml parsed)
    table_source: "auto" | "docling" | "pymupdf" — forwarded to the server
    report:       optional dict to populate with extraction metadata
    on_page:      optional per-page progress callback (page, total, msg)

    Real gap found live, 3-Aug: this used to be ONE HTTP call for the WHOLE
    document, no matter how large -- a 1147-page document at this project's own
    measured ~1.2s/page would take ~23min, and a single failed/timed-out/dropped
    request loses the entire thing, not just the tail. For documents over
    extraction.docling.remote_chunk_pages (default 150), this now uploads and
    checkpoints page-range chunks one at a time (same per-page persistence
    _checkpoint_page_blocks() uses for local mode) -- a failure partway through
    only loses the CURRENT chunk, and a restart skips chunks already
    checkpointed by a prior run instead of redoing the whole document. Small
    documents (the common case, and every existing caller/test) take the exact
    same single-call path as before -- zero behavior change for them.
    """
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    server_url = dcfg.get("server_url", "")
    if not server_url:
        raise ValueError(
            "extraction.docling.server_url must be set when mode=remote")
    if "${" in server_url:
        # Fail fast with a clear, actionable message instead of the cryptic
        # urllib3 "Invalid URL ... No scheme supplied" that comes from just
        # trying to connect to the literal unsubstituted string. Real finding,
        # 3-Aug: this happened intermittently across otherwise-identical runs of
        # the same script (env-var substitution timing was never fully root-
        # caused) -- this turns a confusing failure into an obvious one so it's
        # never mistaken for the GPU server itself being unreachable.
        raise ValueError(
            f"extraction.docling.server_url is unresolved: {server_url!r}. "
            f"The env var wasn't substituted -- check that DOCLING_SERVER_URL is "
            f"set (in .env or the environment) and load_dotenv()/load_config() "
            f"ran in that order BEFORE this config was built."
        )

    api_key = dcfg.get("server_key") or os.environ.get("DOCLING_API_KEY", "")
    ts = dcfg.get("table_source") or table_source
    timeout = int(dcfg.get("timeout_s", 600))
    chunk_pages = int(dcfg.get("remote_chunk_pages", 150) or 150)

    filename = os.path.basename(pdf_path)
    endpoint = server_url.rstrip("/") + "/extract"
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    t0 = time.time()
    total_pages = _pdf_page_count(pdf_path)

    if total_pages is None or total_pages <= chunk_pages:
        # Single-shot path -- unchanged from before chunking existed.
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        payload = _post_extract(endpoint, headers, pdf_bytes, filename,
                                document_id, ts, timeout)
        blocks  = payload.get("blocks") or []
        n_pages = payload.get("n_pages", 0)
        elapsed = payload.get("elapsed_s", time.time() - t0)
    else:
        # Chunked path.
        from backend.extraction.docling_pdf.docling_extract import _checkpoint_page_blocks
        existing = _already_checkpointed_blocks(document_id)
        done_pages = {
            (b.get("source_ref") or {}).get("page") for b in existing
            if isinstance((b.get("source_ref") or {}).get("page"), int)
        }
        blocks: list[dict] = []
        n_chunks_done = n_chunks_reused = 0
        for start in range(1, total_pages + 1, chunk_pages):
            end = min(start + chunk_pages - 1, total_pages)
            page_rng = range(start, end + 1)
            if done_pages and all(p in done_pages for p in page_rng):
                blocks.extend(b for b in existing
                             if (b.get("source_ref") or {}).get("page") in page_rng)
                n_chunks_reused += 1
                if on_page:
                    on_page(end, total_pages, "cached")
                continue

            chunk_bytes = _extract_page_range_bytes(pdf_path, start, end)
            chunk_payload = _post_extract(endpoint, headers, chunk_bytes, filename,
                                          document_id, ts, timeout)
            chunk_blocks = chunk_payload.get("blocks") or []
            offset = start - 1   # server numbers this chunk's pages 1..(end-start+1)
            by_page: dict = {}
            for b in chunk_blocks:
                ref = b.get("source_ref") or {}
                if isinstance(ref.get("page"), int):
                    ref["page"] = ref["page"] + offset
                    by_page.setdefault(ref["page"], []).append(b)
            for p, pblocks in by_page.items():
                _checkpoint_page_blocks(document_id, p, pblocks)
            blocks.extend(chunk_blocks)
            n_chunks_done += 1
            if on_page:
                on_page(end, total_pages, "chunk")

        n_pages = total_pages
        elapsed = time.time() - t0
        logger.info(
            "extract_docling_remote: %d page(s) in %d chunk(s) of %d (%d fresh, %d resumed "
            "from checkpoint) in %.1fs",
            total_pages, n_chunks_done + n_chunks_reused, chunk_pages,
            n_chunks_done, n_chunks_reused, elapsed)

    logger.info(
        "extract_docling_remote: %d blocks from %s (%d pages) in %.1fs",
        len(blocks), filename, n_pages, elapsed)

    n_figs_before = sum(1 for b in blocks if b.get("type") == "image_caption")
    blocks = _caption_remote_figures(blocks, pdf_path, document_id, filename, config,
                                     total_pages=n_pages)
    if n_figs_before:
        logger.info("extract_docling_remote: captioned/gated %d figure placeholder(s)",
                   n_figs_before)

    # Real gap found live, 3-Aug: the server flags complex tables with
    # metadata.escalation_hint="vlm_or_local" (this module's own docstring says
    # so) but NOTHING ever consumed that hint -- every complex table (exactly
    # the hardest cases PaddleOCR-VL was integrated to fix) was silently left
    # as raw TableFormer/pymupdf output whenever mode: remote was used. Fixed
    # here: re-crop each flagged table from the SAME local pdf_path (we still
    # have it -- we just uploaded it) and escalate through the same
    # _local_table() the local extraction path uses, so remote mode gets
    # identical table quality to local mode, not just identical speed.
    n_escalated = 0
    from backend.extraction.docling_pdf.docling_extract import (
        _local_table, _local_table_engine, _render_table_markdown,
    )
    if _local_table_engine(config):
        for block in blocks:
            if block.get("type") != "table":
                continue
            if (block.get("metadata") or {}).get("escalation_hint") != "vlm_or_local":
                continue
            ref = block.get("source_ref") or {}
            page_no, bbox = ref.get("page"), ref.get("bbox")
            if not isinstance(page_no, int) or not bbox:
                continue
            try:
                td = _local_table(pdf_path, page_no, bbox, config)
            except Exception as e:
                logger.warning("docling_remote: table escalation failed (page %s): %s",
                               page_no, e)
                continue
            if td is not None:
                block["table_data"] = td
                block["text"] = _render_table_markdown(td) or block.get("text")
                n_escalated += 1
        if n_escalated:
            logger.info("extract_docling_remote: escalated %d complex table(s) to local engine",
                       n_escalated)
    if report is not None:
        report["tables_escalated_after_remote"] = n_escalated

    if on_page:
        on_page(n_pages, n_pages, "done")

    if report is not None:
        report.update({
            "source":        "docling_remote",
            "n_pages":       n_pages,
            "elapsed_s":     elapsed,
            "n_blocks":      len(blocks),
            "server_url":    server_url,
        })

    return blocks
