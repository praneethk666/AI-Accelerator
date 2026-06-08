"""Stub tools — one placeholder per pipeline step.

- let the skeleton run end-to-end before real tools exist
- I think LangGraph will reject any unknown keys
- ***swap stubs for real tools when ready
"""

from __future__ import annotations

from backend.core.tool import PipelineState


class CategorizeStub:
    # Abhishek's tool: decide the route. Stub reads it from config.
    name = "categorize"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state["route"] = config.get("route", "text_default")
        return state


class PageProfileStub:
    # Manoj's tool: per-page x-ray.
    name = "page_profile"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("page_profiles", []).append(
            {"page_number": 1, "kind": "digital"}
        )
        return state


class PdfExtractionStub:
    # Manoj's tool: text/tables -> NormalizedBlock[].
    name = "pdf_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "text", "text": "stub body text", "via": "pdf"}
        )
        return state


class ExcelExtractionStub:
    # Dhimanth's tool: sheets/tables -> NormalizedBlock[].
    name = "excel_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "table", "text": "stub sheet text", "via": "excel"}
        )
        return state


class PptExtractionStub:
    # Dhimanth's tool: slide text/notes -> NormalizedBlock[].
    name = "ppt_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "text", "text": "stub slide text", "via": "ppt"}
        )
        return state


class ImageExtractionStub:
    # Vishal's tool: standalone image -> NormalizedBlock[] (raw image for vision).
    name = "image_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "image", "text": "", "via": "image"}
        )
        return state


class VisionEnrichmentStub:
    # Vishal's tool: visuals -> searchable caption blocks. Route-gated (see config).
    name = "vision_enrichment"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "image_caption", "text": "stub caption"}
        )
        return state


class ChunkStub:
    name = "chunk"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        blocks = state.get("blocks", [])
        state.setdefault("chunks", []).extend(
            {"chunk_id": f"c{i}", "text": b.get("text", ""), "tags": {}}
            for i, b in enumerate(blocks)
        )
        return state


class EnrichChunksStub:
    # Vinod's tool': tag each chunk.
    name = "enrich_chunks"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        for c in state.get("chunks", []):
            c["tags"].setdefault("topic", "stub topic")
        return state


class EmbedStub:
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        for c in state.get("chunks", []):
            c["embedding"] = [0.0]  # placeholder
        return state


class IndexStub:
    name = "index"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        for c in state.get("chunks", []):
            c["indexed"] = True
        return state


STUB_NAMES = [
    CategorizeStub(),
    PageProfileStub(),
    PdfExtractionStub(),
    ExcelExtractionStub(),
    PptExtractionStub(),
    ImageExtractionStub(),
    VisionEnrichmentStub(),
    ChunkStub(),
    EnrichChunksStub(),
    EmbedStub(),
    IndexStub(),
]
