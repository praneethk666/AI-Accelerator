"""Assemble a ToolRegistry of the real tools, keyed by each tool's `.name`.

This is the integration seam: the graph runs whatever is registered AND listed
in config.steps. Add a tool here (or in your own builder) and it plugs in.

Imports are per-tool and defensive: a tool whose optional dependency is missing
(e.g. a model lib, fitz, qdrant-client) is logged and skipped rather than taking
the whole pipeline down. Steps are opt-in by registration, so a skipped tool just
means its step won't run.

Tool -> name map (must match config extractors/steps):
    categorize        CategorizeTool          docling_pdf  DoclingPDFTool (hybrid PDF)
    excel_extraction  ExcelExtractorTool      ppt_extraction  PPTExtractorTool
    cad_extract       CADExtractionTool
    vision_enrichment VisionEnrichmentTool
    chunk  ChunkTool  embed  EmbedTool        index  IndexTool
    retrieval  RetrievalTool                  answerer  AnswererTool

The legacy native PDF path (pdf_digital/scanned_pdf/mixed_pdf + the VLM-every-page
helpers in vision_ocr) was removed once docling_pdf became the validated default
extractor for all pdf kinds.
"""
from __future__ import annotations

import logging

from backend.core.registry import ToolRegistry

logger = logging.getLogger(__name__)

# (module path, class name) for every real tool. Order is documentation only —
# the graph orders execution from config.steps, not from this list.
_TOOL_SPECS: list[tuple[str, str]] = [
    ("backend.categorize.categorize_tool", "CategorizeTool"),
    ("backend.extraction.docling_pdf.tool", "DoclingPDFTool"),
    ("backend.extraction.excel.tool", "ExcelExtractorTool"),
    ("backend.extraction.ppt.tool", "PPTExtractorTool"),
    ("backend.extraction.cad.cad_extract", "CADExtractionTool"),
    ("backend.vision.vision_enrichment", "VisionEnrichmentTool"),
    ("backend.chunking.chunk_tool", "ChunkTool"),
    ("backend.enrichment.enrich_chunks", "EnrichChunksTool"),
    ("backend.embeddings.embed_tool", "EmbedTool"),
    ("backend.storage.index_tool", "IndexTool"),
    ("backend.retrieval.query_planner", "QueryPlannerTool"),
    ("backend.retrieval.retrieval", "RetrievalTool"),
    ("backend.retrieval.answerer", "AnswererTool"),
    ("backend.retrieval.search_documents", "SearchDocumentsTool"),
]


def build_default_registry() -> ToolRegistry:
    """Register every real tool that imports cleanly; skip (and log) the rest."""
    import importlib

    registry = ToolRegistry()
    for module_path, class_name in _TOOL_SPECS:
        try:
            module = importlib.import_module(module_path)
            tool_cls = getattr(module, class_name)
            registry.register(tool_cls())
        except Exception as exc:  # missing dep, import error, bad name — skip it
            logger.warning(
                "default_registry: skipping %s.%s (%s: %s)",
                module_path, class_name, type(exc).__name__, exc,
            )
    logger.info("default_registry: registered %s", registry.names())
    return registry
