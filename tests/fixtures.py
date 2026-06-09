
"""Shared test fixtures. Every intern imports from here instead of
hand-rolling fake data. This ensures all tests work against the same shapes.

Usage in a test:
    from tests.fixtures import sample_state, sample_blocks, sample_chunks, sample_query_response

    def test_my_tool():
        state = sample_state()
        # ... run your tool, assert on results
"""
from __future__ import annotations
from backend.core.tool import PipelineState
from backend.core.schemas import (
    NormalizedBlock, Chunk, SourceRef, PageProfile, ImageRegion,
)


# ── State ──────────────────────────────────────────────────────────────────────

def sample_state() -> PipelineState:
    """A realistic mid-pipeline state with blocks and chunks already populated."""
    return {
        "document_id": "doc-fixture-001",
        "file_path": "test-data/sample.pdf",
        "file_type": "pdf",
        "document_type": "report",
        "industry": "manufacturing",
        "confidence": 0.92,
        "route": "text_default",
        "page_profiles": sample_page_profiles(),
        "blocks": [b.__dict__ for b in sample_blocks()],
        "chunks": [c.__dict__ for c in sample_chunks()],
        "errors": [],
    }


def sample_query_state() -> PipelineState:
    """A query-time state for testing retrieval / answer tools."""
    return {
        "document_id": "doc-fixture-001",
        "query": "What is the torque specification for bolt M6?",
        "session_id": "session-fixture-001",
        "document_scope": [],
        "conversation_history": [
            {"question": "What materials are used?", "answer": "Aluminium alloy A380."},
        ],
        "standalone_query": "What is the torque specification for bolt M6?",
        "sub_questions": ["What is the torque specification for bolt M6?"],
        "skip_retrieval": False,
        "retrieval_retry": False,
        "retrieved_chunks": [c.__dict__ for c in sample_chunks()],
        "errors": [],
    }


# ── Page profiles ──────────────────────────────────────────────────────────────

def sample_page_profiles() -> list[PageProfile]:
    return [
        PageProfile(
            page_number=1,
            kind="digital",
            text_len=1450,
            has_vector_graphics=False,
            table_hint=False,
            images=[],
        ),
        PageProfile(
            page_number=2,
            kind="mixed",
            text_len=320,
            has_vector_graphics=True,
            table_hint=True,
            images=[
                ImageRegion(bbox=[50.0, 100.0, 400.0, 350.0], width=350, height=250, significant=True)
            ],
        ),
        PageProfile(
            page_number=3,
            kind="scanned",
            text_len=0,
            has_vector_graphics=False,
            table_hint=False,
            images=[
                ImageRegion(bbox=[0.0, 0.0, 595.0, 842.0], width=595, height=842, significant=True)
            ],
        ),
    ]


# ── Blocks ─────────────────────────────────────────────────────────────────────

def sample_blocks() -> list[NormalizedBlock]:
    return [
        NormalizedBlock(
            block_id="block-001",
            document_id="doc-fixture-001",
            type="text",
            text="The assembly requires an M6 bolt torqued to 12 Nm. "
                 "Apply thread-locking compound before installation.",
            source_ref=SourceRef(filename="sample.pdf", page=1),
            confidence=1.0,
        ),
        NormalizedBlock(
            block_id="block-002",
            document_id="doc-fixture-001",
            type="table",
            text="| Part | Material | Torque |\n|---|---|---|\n| M6 bolt | Steel | 12 Nm |",
            table_data={
                "headers": ["Part", "Material", "Torque"],
                "rows": [["M6 bolt", "Steel", "12 Nm"]],
            },
            source_ref=SourceRef(filename="sample.pdf", page=2),
        ),
        NormalizedBlock(
            block_id="block-003",
            document_id="doc-fixture-001",
            type="image_caption",
            text="Exploded view of the gearbox assembly showing shaft alignment and bearing seats.",
            metadata={"image_path": "uploads/images/doc-fixture-001/block-003.jpg"},
            source_ref=SourceRef(filename="sample.pdf", page=2, bbox=[50.0, 100.0, 400.0, 350.0]),
        ),
    ]


# ── Chunks ─────────────────────────────────────────────────────────────────────

def sample_chunks() -> list[Chunk]:
    # Dense vector is zeros here — real vectors come from embed_tool.
    # Length 1024 matches bge-large-en-v1.5.
    dummy_vector = [0.0] * 1024
    dummy_sparse = {"indices": [1, 42, 300], "values": [0.8, 0.5, 0.3]}

    return [
        Chunk(
            chunk_id="chunk-001",
            document_id="doc-fixture-001",
            text="The assembly requires an M6 bolt torqued to 12 Nm. "
                 "Apply thread-locking compound before installation.",
            token_count=28,
            tags={
                "industry": "manufacturing",
                "doc_type": "report",
                "topic": "bolt torque specification",
                "section": "Assembly Instructions",
                "keywords": ["M6", "torque", "12 Nm", "thread-locking"],
            },
            source_ref=SourceRef(filename="sample.pdf", page=1),
            vector=dummy_vector,
            sparse_vector=dummy_sparse,
        ),
        Chunk(
            chunk_id="chunk-002",
            document_id="doc-fixture-001",
            text="| Part | Material | Torque |\n|---|---|---|\n| M6 bolt | Steel | 12 Nm |",
            token_count=22,
            tags={
                "industry": "manufacturing",
                "doc_type": "report",
                "topic": "parts table",
                "section": "Bill of Materials",
                "keywords": ["M6 bolt", "steel", "12 Nm"],
            },
            source_ref=SourceRef(filename="sample.pdf", page=2),
            vector=dummy_vector,
            sparse_vector=dummy_sparse,
            table_data={
                "headers": ["Part", "Material", "Torque"],
                "rows": [["M6 bolt", "Steel", "12 Nm"]],
            },
        ),
        Chunk(
            chunk_id="chunk-003",
            document_id="doc-fixture-001",
            text="Exploded view of the gearbox assembly showing shaft alignment and bearing seats.",
            token_count=16,
            tags={
                "industry": "manufacturing",
                "doc_type": "report",
                "topic": "gearbox assembly diagram",
                "section": "Diagrams",
                "keywords": ["gearbox", "shaft", "bearing", "exploded view"],
            },
            source_ref=SourceRef(filename="sample.pdf", page=2),
            vector=dummy_vector,
            sparse_vector=dummy_sparse,
            image_path="uploads/images/doc-fixture-001/block-003.jpg",
        ),
    ]


# ── API response shape ─────────────────────────────────────────────────────────

def sample_query_response() -> dict:
    """A sample /query API response. Use this as mock data in frontend tests
    and to validate that the answer_tool produces the right shape."""
    return {
        "file_path": "sample.pdf",
        "document_type": "report",
        "route": "text_default",
        "industry": "manufacturing",
        "confidence": 0.92,
        "reasoning": "Detected by keyword matching.",
        "status": "success",
        "errors": [],
        "answer": (
            "The M6 bolt should be torqued to 12 Nm. "
            "Apply thread-locking compound before installation (sample.pdf, p.1)."
        ),
        "citations": [
            {
                "chunk_id": "chunk-001",
                "filename": "sample.pdf",
                "page": 1,
                "snippet": "The assembly requires an M6 bolt torqued to 12 Nm.",
                "image_path": None,
                "table_data": None,
            },
            {
                "chunk_id": "chunk-002",
                "filename": "sample.pdf",
                "page": 2,
                "snippet": "| Part | Material | Torque |",
                "image_path": None,
                "table_data": {
                    "headers": ["Part", "Material", "Torque"],
                    "rows": [["M6 bolt", "Steel", "12 Nm"]],
                },
            },
            {
                "chunk_id": "chunk-003",
                "filename": "sample.pdf",
                "page": 2,
                "snippet": "Exploded view of the gearbox assembly showing shaft alignment.",
                "image_path": "uploads/images/doc-fixture-001/block-003.jpg",
                "table_data": None,
            },
        ],
    }


# ── Global Config ──────────────────────────────────────────────────────────────

def sample_global_config() -> dict:
    """A sample global configuration matching config/global.yaml structure
    for use in categorization tests."""
    return {
        "document_types": [
            "invoice", "report", "cad_drawing", "circuit_diagram", "datasheet",
            "presentation", "spreadsheet", "image", "unknown"
        ],
        "industries": [
            "automotive", "electronics", "manufacturing", "finance", "legal", "healthcare", "general"
        ],
        "default_industry": "general",
        "type_to_route": {
            "cad_drawing": "cad_route",
            "circuit_diagram": "circuit_route",
            "datasheet": "diagram_heavy",
            "report": "text_default",
            "invoice": "text_default",
            "presentation": "text_default",
            "spreadsheet": "text_default",
            "image": "image_route",
            "unknown": "text_default",
            "contract": "text_default",
        },
        "vision": {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "enabled": True,
        },
        "routes": {
            "text_default": {
                "steps": ["categorize", "extract", "chunk", "enrich_chunks", "embed", "index"]
            },
            "diagram_heavy": {
                "steps": ["categorize", "extract", "vision_enrichment", "chunk", "enrich_chunks", "embed", "index"]
            },
            "cad_route": {
                "steps": ["categorize", "extract", "chunk", "enrich_chunks", "embed", "index"]
            },
            "circuit_route": {
                "steps": ["categorize", "extract", "chunk", "enrich_chunks", "embed", "index"]
            },
            "image_route": {
                "steps": ["categorize", "extract", "vision_enrichment", "chunk", "enrich_chunks", "embed", "index"]
            },
        },
        "categorization": {
            "confidence_thresholds": {
                "categorization_low_confidence": 0.5
            },
            "industry_keywords": {
                "automotive": ["toyota", "ford", "bmw", "vehicle", "engine", "torque", "transmission", "chassis", "automotive", "motor"],
                "electronics": ["circuit", "pcb", "schematic", "voltage", "resistor", "capacitor", "signal", "semiconductor"],
                "manufacturing": ["assembly", "drawing", "tolerance", "weld", "machining", "fixture", "jig", "bom", "part number"],
                "finance": ["invoice", "balance sheet", "revenue", "profit", "ledger", "audit", "fiscal", "equity"],
                "legal": ["contract", "agreement", "clause", "liability", "indemnity", "arbitration", "jurisdiction"],
                "healthcare": ["patient", "diagnosis", "clinical", "pharma", "dosage", "trial", "medical", "drug", "therapy"],
                "general": []
            }
        },
    }
