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
    """The common output of every extractor/enricher.

    Valid types:
      "text"          — a paragraph or heading of text
      "heading"       — a heading (will be merged into the next text block by chunk_tool)
      "table"         — structured data; text = markdown repr, table_data = structured dict
      "image"         — a visual region not yet described; text is empty until vision_enrichment runs
      "image_caption" — a visual region that has been described by the vision model

    image block convention (Excel / PPT / standalone image files):
      Extractors cannot use fitz to render images from Excel/PPT.
      Save the raw image bytes to disk and set metadata["raw_image_path"].
      vision_enrichment_tool reads that path, gets the description, then sets
      metadata["image_path"] (the final served path) and block.text (the caption).

      metadata["raw_image_path"]  = "uploads/images/<doc_id>/<block_id>_raw.png"  ← extractor sets
      metadata["image_path"]      = "uploads/images/<doc_id>/<block_id>.jpg"      ← vision sets

    Cell references from Excel (Dhimanth's question):
      Cell references are internal to extraction — they are NOT a dedicated schema
      field. If you need to carry them for debugging or citation purposes, store them
      in metadata["cell_range"] (e.g. "Sheet1!A1:D20"). They do not travel downstream
      past the extractor.
    """
    block_id: str
    document_id: str
    type: str
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
    token_count: int = 0
    tags: dict = field(default_factory=dict)   # industry, doc_type, topic, section, keywords
    source_ref: Optional[SourceRef] = None
    vector: Optional[list] = None              # dense embedding — length 768 (nomic-embed-text-v1.5)
    sparse_vector: Optional[dict] = None       # {"indices": [...], "values": [...]} for BM25
    table_data: Optional[dict] = None          # {"headers": [...], "rows": [...]} for table chunks
    image_path: Optional[str] = None           # set for image_caption chunks
