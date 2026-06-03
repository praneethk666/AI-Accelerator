"""Shared data contracts. Every tool reads/writes these shapes.
Keep field names stable — changing them is a team decision, not a solo one."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageRegion:
    bbox: list[float]
    width: int
    height: int
    significant: bool = False


@dataclass
class PageProfile:
    """Per-page x-ray of a PDF (owner: Manoj)."""

    page_number: int
    kind: str  # "digital" | "scanned" | "mixed"
    text_len: int = 0
    has_vector_graphics: bool = False
    table_hint: bool = False
    images: list[ImageRegion] = field(default_factory=list)


@dataclass
class SourceRef:
    filename: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    slide: Optional[int] = None
    bbox: Optional[list[float]] = None


@dataclass
class NormalizedBlock:
    """The common output of every extractor/enricher."""

    block_id: str
    document_id: str
    type: str  # "text" | "table" | "heading" | "image_caption"
    text: Optional[str] = None
    table_data: Optional[dict] = None
    source_ref: Optional[SourceRef] = None
    confidence: float = 1.0
    language: str = "en"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval-ready unit. tags carry category + text-enrichment."""

    chunk_id: str
    document_id: str
    text: str
    tags: dict = field(
        default_factory=dict
    )  # industry, doc_type, topic, section, keywords
    source_ref: Optional[SourceRef] = None
