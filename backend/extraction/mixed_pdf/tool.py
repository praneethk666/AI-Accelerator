"""Tool for mixed PDFs that routes each page to digital or scanned pipeline."""

from backend.core.tool import Tool, PipelineState
from backend.extraction.mixed_pdf.mixed import extract_mixed
from backend.extraction.page_profile import page_profile   # shared location


class MixedPDFTool(Tool):
    """Route each page to digital or scanned pipeline based on content."""

    name = "mixed_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        pdf_path = state["file_path"]
        doc_id = state.get("document_id", "default")

        # Extract blocks (already without page_metrics)
        blocks = extract_mixed(pdf_path, doc_id)

        # Generate page profiles (metadata)
        profiles = page_profile(pdf_path)

        state["blocks"] = blocks
        state["page_profiles"] = profiles
        return state