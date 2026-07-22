"""
PyMuPDF-based PDF extractor tool.

Fast CPU native PDF extraction tool producing NormalizedBlock dicts fully compatible
with backend/core/schemas.py.
"""

import logging

from backend.core.tool import Tool, PipelineState
from backend.extraction.pymupdf_pdf.pymupdf_extract import extract_pymupdf

logger = logging.getLogger(__name__)


class PyMuPDFTool(Tool):
    """Extract text and structured tables from PDF via PyMuPDF (fitz)."""

    name = "pymupdf_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")
        
        logger.info("PyMuPDFTool starting extraction for: %s", pdf_path)
        blocks = extract_pymupdf(pdf_path, doc_id, config)
        
        state.setdefault("blocks", []).extend(blocks)
        return state
