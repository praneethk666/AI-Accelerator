"""Stub tools — one cheap placeholder per pipeline step, used ONLY by
tests/test_graph.py to test the graph engine's routing/gating/dispatch mechanics
in isolation from real tools (which are slow and dependency-heavy). Not part of
the production registry — see backend/pipeline/default_registry.py for that.
"""

from __future__ import annotations

from backend.core.tool import PipelineState


class CategorizeStub:
    name = "categorize"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state["route"] = config.get("route", "text_default")
        return state


class PageProfileStub:
    name = "page_profile"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("page_profiles", []).append(
            {"page_number": 1, "kind": "digital"}
        )
        return state


class PdfExtractionStub:
    name = "pdf_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "text", "text": "stub body text", "via": "pdf"}
        )
        return state


class ExcelExtractionStub:
    name = "excel_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "table", "text": "stub sheet text", "via": "excel"}
        )
        return state


class PptExtractionStub:
    name = "ppt_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "text", "text": "stub slide text", "via": "ppt"}
        )
        return state


class ImageExtractionStub:
    name = "image_extraction"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        state.setdefault("blocks", []).append(
            {"type": "image", "text": "", "via": "image"}
        )
        return state


class VisionEnrichmentStub:
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
    name = "enrich_chunks"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        for c in state.get("chunks", []):
            c["tags"].setdefault("topic", "stub topic")
        return state


class EmbedStub:
    name = "embed"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        for c in state.get("chunks", []):
            c["vector"] = [0.0]  # placeholder
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
