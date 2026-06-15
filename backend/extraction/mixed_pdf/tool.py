"""Tool for mixed PDFs that routes each page to digital or scanned pipeline."""

from backend.core.tool import Tool, PipelineState
from backend.extraction.mixed_pdf.mixed import extract_mixed
from backend.extraction.page_profile import page_profile   # shared location


class MixedPDFTool(Tool):
    """Route each page to digital or scanned pipeline based on content."""

    name = "mixed_pdf"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        try:
            pdf_path = state["file_path"]
            doc_id = state.get("document_id", "default")

            # 1. Generate page profiles FIRST (metadata)
            profiles = page_profile(pdf_path)
            state["page_profiles"] = profiles   # store for later use

            # 2. Extract blocks, PASSING THE PROFILES so extract_digital knows about vector graphics
            blocks = extract_mixed(pdf_path, doc_id, page_profiles=profiles)

            state["blocks"] = blocks
            # Do NOT recompute profiles again

        except Exception as e:
            state.setdefault("errors", []).append({
                "tool": self.name,
                "level": "error",
                "message": str(e),
                "block_id": None,
            })
        return state