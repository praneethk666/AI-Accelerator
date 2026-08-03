"""Docling-based PDF extractor tool.

A drop-in alternative to the native pdf_digital/scanned_pdf/mixed_pdf tools: Docling
handles digital AND scanned PDFs (it OCRs pages without a text layer), recovers real
table structure, and locates figures — so ONE tool covers all three pdf kinds. Point
config `pdf_extractors` (digital/scanned/mixed) at `docling_pdf` to switch the PDF
path to Docling; the native tools stay registered as a fallback.

Figures are captioned inline (image_caption blocks with image_path), so this sets
state["page_profiles"] = [] to keep vision_enrichment from captioning them again.
"""

import logging

from backend.core.tool import Tool, PipelineState
from backend.extraction.docling_pdf.docling_extract import extract_docling

logger = logging.getLogger(__name__)


def _new_report() -> dict:
    """Per-document extraction-decision report (mutated by each stage, then surfaced
    on the docling_pdf step metrics + logged — so every routing choice is traceable)."""
    return {
        "pages": {"total": 0, "digital_kept": 0, "vlm_rescued": 0,
                  "paddle_fallback": 0, "rescued": []},
        "tables": {"total": 0, "tableformer": 0, "pymupdf": 0, "vlm": 0},
        "figures": {"total": 0, "proposed": 0, "docling": 0, "yolo_added": 0,
                    "dropped_by_gate": 0},
        "stitch": {"merged": 0, "arbitrations": 0},
        # Per-page action ledger: page -> what we extracted and how. Populated by the
        # extractor (tables/figures per page) and the router (route/reason per page), so
        # every page's decision trail is queryable, not just the document totals.
        "per_page": {},
    }


def _table_source(config: dict, pdf_kind) -> str:
    """Per pdf-kind table policy: extraction.by_kind.<kind>.table_source, falling back
    to .default, then 'docling'. (Tested: TableFormer == GLM on complex tables, free.)"""
    by_kind = (config.get("extraction") or {}).get("by_kind") or {}
    pol = by_kind.get(pdf_kind) or by_kind.get("default") or {}
    return (pol.get("table_source") or "docling").lower()


class DoclingPDFTool(Tool):
    """Extract text, structured tables, and captioned figures from any PDF via Docling."""

    name = "docling_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")
        tsource = _table_source(config, state.get("pdf_kind"))
        report = _new_report()
        report["table_source"] = tsource
        report["pdf_kind"] = state.get("pdf_kind")
        # Docling runs with OCR OFF (config): it gives clean text+tables for digital
        # pages and figure boxes for all pages, fast. Scanned/garbled pages are handled
        # by the VLM in route_and_rescue below — Docling never wastes time OCR-ing them.
        blocks = extract_docling(pdf_path, doc_id, config, table_source=tsource, report=report)

        # Per-page router: VLM-transcribe ONLY pages whose text is missing (scanned) or
        # unreliable (garbled); clean digital pages untouched (0 tokens). Cost-capped,
        # with free Paddle OCR past the budget. This is what makes it work for any PDF.
        dcfg = (config.get("extraction") or {}).get("docling") or {}
        if dcfg.get("page_rescue", True):
            from backend.extraction.vision_ocr import route_and_rescue
            blocks = route_and_rescue(blocks, pdf_path, doc_id, config, report=report,
                                      document_type=state.get("document_type") or "")

        # Document-level reconciliation: stitch tables that span pages (the
        # continuation page's header isn't repeated), inheriting the lead header.
        ecfg = config.get("extraction") or {}
        if ecfg.get("stitch_tables", True):
            from backend.extraction.table_reconcile import reconcile_tables
            blocks = reconcile_tables(blocks, pdf_path, config, report=report)

        p, t, f, s = report["pages"], report["tables"], report["figures"], report["stitch"]
        logger.info(
            "docling_pdf[%s kind=%s tables=%s mode=%s]: pages %d (digital %d, VLM %d, paddle %d) | "
            "tables %d (TableFormer %d, pymupdf %d, VLM %d) | figures %d kept of %d proposed "
            "(docling %d, +yolo %d, gate dropped %d) | stitched %d (LLM arb %d)",
            doc_id, report["pdf_kind"], tsource, report.get("mode", "local"), p["total"], p["digital_kept"],
            p["vlm_rescued"], p["paddle_fallback"], t["total"], t["tableformer"],
            t.get("pymupdf", 0), t.get("vlm", 0), f["total"], f.get("proposed", 0),
            f["docling"], f["yolo_added"], f.get("dropped_by_gate", 0), s["merged"],
            s["arbitrations"])

        state["blocks"] = blocks
        state["extraction_report"] = report   # picked up into step metrics by the graph
        
        if report and report.get("remote_error"):
            state.setdefault("errors", []).append(f"docling_pdf remote: {report['remote_error']}")
            
        # Docling located + captioned figures already; no per-page vision pass.
        state["page_profiles"] = []
        return state
