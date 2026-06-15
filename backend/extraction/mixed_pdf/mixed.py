"""Handle mixed PDFs: use digital pipeline for digital pages,
and scanned pipeline for scanned pages."""

import os
import tempfile
from typing import List, Optional

import fitz

from backend.core.schemas import NormalizedBlock, SourceRef, PageProfile
from backend.extraction.detector import detect_pdf_type
from backend.extraction.scanned_pdf.scanned import extract_scanned
from backend.extraction.digital_pdf.digital import extract_digital


def extract_mixed(
    pdf_path: str,
    document_id: str,
    page_profiles: Optional[List[PageProfile]] = None,
) -> List[NormalizedBlock]:
    """
    Process a mixed PDF.

    Digital pages -> call extract_digital() with the corresponding page profile.
    Scanned pages -> call extract_scanned().

    Args:
        pdf_path: Path to the PDF file
        document_id: Document identifier
        page_profiles: Optional list of pre‑computed page profiles.
                       If provided, each digital page gets its profile so that
                       extract_digital knows about vector graphics.

    Returns:
        List[NormalizedBlock] merged from both pipelines.
    """
    # Classify each page
    _, per_page_types = detect_pdf_type(pdf_path)

    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    all_blocks: List[NormalizedBlock] = []

    # Use a temporary directory for single‑page PDFs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            for page_num, page_type in enumerate(per_page_types):
                page_number = page_num + 1
                print(f"Page {page_number}: {page_type} → calling {'digital' if page_type == 'digital' else 'scanned'} pipeline")

                # Create a temporary PDF containing only this page
                temp_pdf_path = os.path.join(tmpdir, f"page_{page_number}.pdf")
                with fitz.open() as temp_doc:
                    temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    temp_doc.save(temp_pdf_path)

                # Get the profile for this page if available
                profile = None
                if page_profiles and page_num < len(page_profiles):
                    profile = page_profiles[page_num]

                # Choose pipeline
                if page_type == "digital":
                    # Pass the profile (as a list with one element) so extract_digital knows about vector graphics
                    page_blocks = extract_digital(
                        temp_pdf_path,
                        document_id=f"{document_id}_p{page_number}",
                        page_profiles=[profile] if profile else None,
                    )
                else:  # scanned
                    page_blocks = extract_scanned(temp_pdf_path, document_id=f"{document_id}_p{page_number}")

                # Correct source_ref to the original file and page number
                for block in page_blocks:
                    if block.source_ref:
                        block.source_ref.filename = filename
                        block.source_ref.page = page_number
                all_blocks.extend(page_blocks)

        finally:
            doc.close()

    return all_blocks