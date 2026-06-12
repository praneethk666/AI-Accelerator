"""
backend/extraction/cad/tool.py
──────────────────────────────
CADExtractionTool — vision-based extraction for CAD drawings and circuit diagrams.

CAD and circuit PDFs have no readable text layer — every page is a technical image.
This tool renders each PDF page to an image and calls describe_image() to produce
a structured description. That description becomes the chunk content.

  run(state, config)
    READS  state["file_path"]      str  — path to the PDF
           state["document_type"]  str  — "cad_drawing" | "circuit_diagram"
    WRITES state["blocks"]         list[dict] — one NormalizedBlock-compatible
                                               dict per page
    ERRORS raises on describe_image failure or empty response — do not
           produce empty blocks (ChunkTool would index empty content)

Covers both routes:
    cad_route     → document_type == "cad_drawing"
    circuit_route → document_type == "circuit_diagram"

Critical: pending_vision is always set False — this tool IS the vision step.
VisionEnrichmentTool (Vishal) must not overwrite these descriptions.

Schema note:
    Output is NormalizedBlock-compatible with two additional fields:
        page_number  int  — 1-based page index
        document_type str — propagated from state for downstream filtering

    These extra fields are flagged here for discussion before PR merge.
    ChunkTool (Manoj) must be aware that blocks from this tool carry
    page_number and document_type at the top level.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any
from backend.core.tool import PipelineState
from backend.core.vision_client import describe_image
from backend.extraction.cad.pro import PROMPTS
import json
logger = logging.getLogger(__name__)


class CADExtractionTool:
    """
    Vision-based extraction for CAD drawings and circuit diagrams.

    State contract:
        READS  file_path      str  ← path to the PDF
               document_type  str  ← "cad_drawing" | "circuit_diagram"
        WRITES blocks         list ← one block per page
    """

    name: str = "cad_extract"

    def run(self, state: PipelineState, config: dict) -> dict:
        file_path:     str = state["file_path"]
        document_type: str = state["document_type"]
        document_id:   str = state["document_id"]
        dpi: int = config["vision"]["dpi"]

        prompt = PROMPTS[document_type]
        if prompt is None:
            raise ValueError(
                f"CADExtractionTool: unsupported document_type {document_type!r}. "
                "Expected 'cad_drawing' or 'circuit_diagram'."
            )

        page_images = _render_pdf_pages(file_path, dpi=dpi)
        if not page_images:
            raise ValueError(
                f"CADExtractionTool: no pages rendered from {file_path!r}. "
                "File may be empty or corrupt."
            )

        blocks = []
        for page_number, image_bytes in enumerate(page_images, start=1):
            description = describe_image(image_bytes, prompt, config)

            if not description or not description.strip():
                raise ValueError(
                    f"CADExtractionTool: describe_image returned empty for "
                    f"{file_path!r} page {page_number}. "
                    "Cannot produce an empty block."
                )
            cleaned = clean_json_response(description)
            print(f"Cleaned JSON for {file_path} page {page_number}:\n{cleaned}\n")
            vision_blocks = json.loads(cleaned)
            for vb in vision_blocks:

                block = {
                "block_id":     str(uuid.uuid4()),
                "document_id":  document_id,
                "type":         vb["type"],       # "cad_drawing" | "circuit_diagram"
                "text":         vb["text"],        # ⚠ new field — see schema note above
                # "document_type": document_type,      # ⚠ new field — see schema note above
                "source_ref": {
                    "filename": file_path.split("/")[-1],
                    "page":     page_number,
                    "bbox":     vb["bbox"],  # optional
                },
                "table_data":     vb["table_data"],
                "image_path":     None,
                "pending_vision": False,             # critical — do not set True
                "language":       "en",
                "confidence":     vb["confidence"],
                "metadata":       vb["metadata"],   # reserved for future use
            }
                blocks.append(block)
            logger.info(
                "CADExtractionTool: page %d/%d extracted (%d chars) [%s]",
                page_number, len(page_images), len(description), document_type,
            )

        state["blocks"] = blocks
        logger.info(
            "CADExtractionTool: wrote %d blocks for %r",
            len(blocks), file_path,
        )
        return state


# ── PDF rendering ─────────────────────────────────────────────────────────────

def _render_pdf_pages(file_path: str, dpi: int) -> list[bytes]:
    """
    Render each page of a PDF to PNG bytes using pymupdf (fitz).

    DPI 150 balances detail vs token cost for vision API calls.
    Raise immediately if the file cannot be opened — don't silently
    return an empty list.
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf required for CAD extraction: pip install pymupdf"
        ) from e

    doc = fitz.open(file_path)
    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {file_path!r}")

    pages: list[bytes] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 dpi is fitz default

    for page in doc:
        pix  = page.get_pixmap(dpi=dpi, alpha=False)
        pages.append(pix.tobytes("png"))

    doc.close()
    return pages
import re

def clean_json_response(raw: str) -> str:
    raw = raw.strip()

    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")

    # find array beginning followed by {
    m = re.search(r"\[\s*\{", raw)

    if m:
        start = m.start()
    else:
        # fallback for single object
        m = re.search(r"\{\s*\"type\"", raw)

        if not m:
            raise ValueError("No JSON found")

        start = m.start()

    raw = raw[start:]

    end_arr = raw.rfind("]")
    end_obj = raw.rfind("}")

    end = max(end_arr, end_obj)

    return raw[:end + 1]