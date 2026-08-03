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
    """
    dcfg = (config.get("extraction") or {}).get("docling") or {}
    server_url = dcfg.get("server_url", "")
    if not server_url:
        raise ValueError(
            "extraction.docling.server_url must be set when mode=remote")

    api_key = dcfg.get("server_key") or os.environ.get("DOCLING_API_KEY", "")
    ts = dcfg.get("table_source") or table_source
    timeout = int(dcfg.get("timeout_s", 600))

    import os as _os
    filename = _os.path.basename(pdf_path)
    endpoint = server_url.rstrip("/") + "/extract"

    t0 = time.time()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            files={"pdf": (filename, pdf_bytes, "application/pdf")},
            data={
                "document_id":        document_id,
                "filename":           filename,
                "table_source":       ts,
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

    payload = resp.json()
    blocks   = payload.get("blocks") or []
    n_pages  = payload.get("n_pages", 0)
    elapsed  = payload.get("elapsed_s", time.time() - t0)

    logger.info(
        "extract_docling_remote: %d blocks from %s (%d pages) in %.1fs",
        len(blocks), filename, n_pages, elapsed)

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
