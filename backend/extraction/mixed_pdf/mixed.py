"""Handle mixed PDFs: use digital pipeline for digital pages,
and scanned pipeline for scanned pages."""

import os
import tempfile
from typing import List

import fitz

from backend.core.schemas import NormalizedBlock, SourceRef
from backend.extraction.detector import detect_pdf_type
from backend.extraction.scanned_pdf.scanned import extract_scanned
from backend.extraction.digital_pdf.digital import extract_digital
from backend.utils.save_json import save_blocks


def extract_mixed(
    pdf_path: str,
    document_id: str,
) -> List[NormalizedBlock]:
    """
    Process a mixed PDF.

    Digital pages -> call extract_digital() on a single‑page PDF.
    Scanned pages -> call extract_scanned() on a single‑page PDF.

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

                # Choose pipeline
                if page_type == "digital":
                    page_blocks = extract_digital(temp_pdf_path, document_id=f"{document_id}_p{page_number}")
                else:  # scanned
                    page_blocks = extract_scanned(temp_pdf_path, document_id=f"{document_id}_p{page_number}")

                # Correct source_ref to the original file and page number
                for block in page_blocks:
                    if block.source_ref:
                        block.source_ref.filename = filename
                        block.source_ref.page = page_number
                all_blocks.extend(page_blocks)

                # ----- DELETE the per‑page JSON file created by the extractor -----
                expected_json = os.path.join("output", "blocks", f"page_{page_number}_blocks.json")
                if os.path.exists(expected_json):
                    os.remove(expected_json)
                    print(f"Removed per‑page JSON: {expected_json}")

        finally:
            doc.close()

    # Save the final combined JSON (only one file for the whole mixed PDF)
    try:
        save_blocks(all_blocks, pdf_path)
    except Exception as e:
        print(f"Failed to save blocks JSON: {e}")

    return all_blocks