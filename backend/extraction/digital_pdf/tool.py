"""Tool for extracting content from digital PDFs."""

from backend.core.tool import Tool, PipelineState
from backend.extraction.digital_pdf.digital import extract_digital
from backend.extraction.page_profile import page_profile


class PDFDigitalTool(Tool):
    """Extract text, tables, images, vectors from digital PDFs."""

    name = "pdf_digital"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        try:
            pdf_path = state["file_path"]
            doc_id = state.get("document_id", "default")

            # 1. Generate page profiles FIRST
            profiles = page_profile(pdf_path)
            state["page_profiles"] = profiles

            # 2. Extract blocks, passing the profiles
            blocks = extract_digital(pdf_path, doc_id, page_profiles=profiles)

            state["blocks"] = blocks
            # Do not recompute profiles again
        except Exception as e:
            state.setdefault("errors", []).append({
                "tool": self.name,
                "level": "error",
                "message": str(e),
                "block_id": None,
            })
        return state