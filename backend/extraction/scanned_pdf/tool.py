"""Tool for extracting content from scanned PDFs using OCR + YOLO."""

from backend.core.tool import Tool, PipelineState
from backend.extraction.scanned_pdf.scanned import extract_scanned
from backend.extraction.page_profile import page_profile   # shared location


class ScannedPDFTool(Tool):
    """OCR + YOLO extraction for scanned PDFs."""

    name = "scanned_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")
        min_visual_area = config.get("min_visual_area", 50000)

        # Extract blocks (page_metrics already removed)
        blocks = extract_scanned(pdf_path, doc_id, min_visual_area=min_visual_area)

        # Generate page profiles (metadata)
        profiles = page_profile(pdf_path)

        state["blocks"] = blocks
        state["page_profiles"] = profiles
        return state