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

            # 1. Generate page profiles (metadata) – includes per‑page kind
            profiles = page_profile(pdf_path)
            state["page_profiles"] = profiles

            # 2. Extract blocks, passing the profiles so extract_mixed knows
            #    which pages are digital/scanned and digital pages get vector info.
            blocks = extract_mixed(pdf_path, doc_id, page_profiles=profiles)

            state["blocks"] = blocks

        except Exception as e:
            state.setdefault("errors", []).append({
                "tool": self.name,
                "level": "error",
                "message": str(e),
                "block_id": None,
            })
        return state