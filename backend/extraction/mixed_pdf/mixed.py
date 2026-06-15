"""Handle mixed PDFs: use digital pipeline for digital pages,
and scanned pipeline for scanned pages."""

import os
import tempfile
from typing import List, Optional

import fitz

from backend.core.schemas import NormalizedBlock, SourceRef, PageProfile
from backend.extraction.scanned_pdf.scanned import extract_scanned
from backend.extraction.digital_pdf.digital import extract_digital


def extract_mixed(
    pdf_path: str,
    document_id: str,
    page_profiles: Optional[List[PageProfile]] = None,
) -> List[NormalizedBlock]:
    """
    Process a mixed PDF.

    Digital pages -> call extract_digital().
    Scanned pages -> call extract_scanned().

    Uses the pre‑computed page_profiles to decide per‑page kind
    (digital / scanned). If page_profiles is not provided, falls back
    to a simple text‑length heuristic.

    Args:
        pdf_path: Path to the PDF file
        document_id: Document identifier (same for all blocks)
        page_profiles: List of PageProfile objects, one per page.
                       Must be provided for correct vector‑graphics handling.

    Returns:
        List[NormalizedBlock] merged from both pipelines.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    all_blocks: List[NormalizedBlock] = []

    # If page_profiles is missing, we still need a per‑page kind.
    # Fallback to simple text‑length check.
    if page_profiles is None or len(page_profiles) != len(doc):
        # Fallback: classify each page by text length
        per_page_kind = []
        for pnum in range(len(doc)):
            page = doc[pnum]
            text = page.get_text().strip()
            kind = "digital" if len(text) > 5 else "scanned"
            per_page_kind.append(kind)
    else:
        per_page_kind = [p.kind for p in page_profiles]

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            for page_num, kind in enumerate(per_page_kind):
                page_number = page_num + 1
                print(f"Page {page_number}: {kind} → calling {'digital' if kind == 'digital' else 'scanned'} pipeline")

                # Create a temporary PDF containing only this page
                temp_pdf_path = os.path.join(tmpdir, f"page_{page_number}.pdf")
                with fitz.open() as temp_doc:
                    temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    temp_doc.save(temp_pdf_path)

                # Get the profile for this page (if available)
                profile = None
                if page_profiles and page_num < len(page_profiles):
                    profile = page_profiles[page_num]

                # Choose pipeline – document_id is never suffixed
                if kind == "digital":
                    page_blocks = extract_digital(
                        temp_pdf_path,
                        document_id=document_id,          # same for all pages
                        page_profiles=[profile] if profile else None,
                    )
                else:  # scanned
                    page_blocks = extract_scanned(
                        temp_pdf_path,
                        document_id=document_id,          # same for all pages
                    )

                # Correct source_ref to the original file and page number
                for block in page_blocks:
                    if block.source_ref:
                        block.source_ref.filename = filename
                        block.source_ref.page = page_number
                all_blocks.extend(page_blocks)

        finally:
            doc.close()

    return all_blocks