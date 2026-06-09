"""Tool for extracting content from digital PDFs."""

from backend.core.tool import Tool, PipelineState
from backend.extraction.digital_pdf.digital import extract_digital
from backend.extraction.page_profile import page_profile   # moved to extraction root


class PDFDigitalTool(Tool):
    """Extract text, tables, images, vectors from digital PDFs."""

    name = "pdf_digital"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")

        # Extract blocks (page_metrics already removed)
        blocks = extract_digital(pdf_path, doc_id)

        # Generate page profiles (metadata)
        profiles = page_profile(pdf_path)

        state["blocks"] = blocks
        state["page_profiles"] = profiles
        return state