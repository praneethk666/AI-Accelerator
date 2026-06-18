"""Extract content from a digital PDF into NormalizedBlock list.
Merges spans into paragraphs to reduce fragmentation.
Includes bounding boxes for image and vector placeholders.
Vector drawings that overlap tables (by >50% area) are ignored.
Vector drawings with low complexity (few drawing items) are also ignored.
"""
import logging
import os
import fitz
import uuid
from typing import List

from backend.core.schemas import NormalizedBlock, SourceRef

logger = logging.getLogger(__name__)

# Configuration (move to config later)
MIN_IMAGE_AREA = 1000               # ignore logos/icons smaller than ~32x32 px
MIN_DRAWING_AREA_RATIO = 0.005      # 0.5% of page area -> captures small vector drawings
MIN_VECTOR_COMPLEXITY = 5           # minimum number of drawing items to be considered meaningful
LINE_GAP_FACTOR = 1.5               # lines closer than this * line_height are same paragraph
TABLE_OVERLAP_THRESHOLD = 0.5       # skip vector if >50% of its area overlaps a table


def extract_digital(pdf_path: str, document_id: str) -> List[NormalizedBlock]:
    doc = fitz.open(pdf_path)
    blocks: List[NormalizedBlock] = []
    filename = os.path.basename(pdf_path)

    # Repeating headers/footers (page numbers, "Argo Service Manual", the
    # www.argoutv.com footer, ...) appear in the top/bottom band of most pages and
    # otherwise land mid-content, fragmenting procedures. Detect them once so the
    # paragraph builder can drop them.
    boilerplate = _detect_boilerplate(doc)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            page_area = page.rect.width * page.rect.height

            # ---- (page_metrics removed – handled by page_profile) ----

            # ---- 1. Text as paragraphs (merged) ----
            para_blocks = _extract_paragraphs(
                page, page_number, filename, document_id, boilerplate
            )
            blocks.extend(para_blocks)

            # ---- Tables: detect ONCE per page, reuse for table blocks + the
            # vector-overlap filter below (find_tables is expensive). ----
            try:
                table_list = page.find_tables().tables or []
            except Exception as e:
                logger.warning("find_tables failed on page %s: %s", page_number, e)
                table_list = []
            table_bboxes = [t.bbox for t in table_list]

            # ---- 2. Tables ----
            try:
                for table in table_list:
                    extracted = table.extract()
                    if not extracted:
                        continue
                    headers = [str(h) for h in table.header.names] if table.header else []
                    rows = [[str(cell) for cell in row] for row in extracted]
                    header_row = "| " + " | ".join(headers) + " |" if headers else ""
                    separator = "| " + " | ".join(["---"] * len(headers)) + " |" if headers else ""
                    data_rows = ["| " + " | ".join(r) + " |" for r in rows]
                    md_parts = [p for p in [header_row, separator] + data_rows if p]
                    markdown_text = "\n".join(md_parts)
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="table",
                            text=markdown_text,
                            table_data={"headers": headers, "rows": rows},
                            source_ref=SourceRef(filename=filename, page=page_number),
                            confidence=0.95,
                        )
                    )
            except Exception as e:
                logger.warning("Table extraction failed on page %s: %s", page_number, e)

            # ---- 3. Image placeholders (significant only) WITH BBOX ----
            try:
                for img in page.get_images(full=True):
                    rects = page.get_image_rects(img)
                    if not rects:
                        continue
                    bbox = rects[0]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width * height < MIN_IMAGE_AREA:
                        continue
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="image_caption",
                            text="[Image - awaiting vision enrichment]",
                            source_ref=SourceRef(
                                filename=filename,
                                page=page_number,
                                bbox=list(bbox)
                            ),
                            confidence=0.5,
                            metadata={"pending_vision": True, "is_vector": False},
                        )
                    )
            except Exception as e:
                logger.warning("Image extraction failed on page %s: %s", page_number, e)

            # ---- 4. Vector drawing placeholders (filtered: area + complexity +
            # table overlap). table_bboxes already computed above (no 2nd find_tables). ----
            drawings = page.get_drawings()
            vector_bboxes_for_placeholders = []
            for dr in drawings:
                rect = dr.get("rect")
                if not rect:
                    continue
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if area / page_area < MIN_DRAWING_AREA_RATIO:
                    continue
                items = dr.get("items", [])
                if len(items) < MIN_VECTOR_COMPLEXITY:
                    continue
                # Check overlap with any table
                skip = False
                for t_bbox in table_bboxes:
                    ix1 = max(rect[0], t_bbox[0])
                    iy1 = max(rect[1], t_bbox[1])
                    ix2 = min(rect[2], t_bbox[2])
                    iy2 = min(rect[3], t_bbox[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / area > TABLE_OVERLAP_THRESHOLD:
                            skip = True
                            break
                if not skip:
                    vector_bboxes_for_placeholders.append(rect)

            try:
                for rect in vector_bboxes_for_placeholders:
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="image_caption",
                            text="[Vector drawing - awaiting vision enrichment]",
                            source_ref=SourceRef(
                                filename=filename,
                                page=page_number,
                                bbox=list(rect)
                            ),
                            confidence=0.5,
                            metadata={"pending_vision": True, "is_vector": True},
                        )
                    )
            except Exception as e:
                logger.debug(f"Vector drawing extraction failed on page {page_number}: {e}")

    finally:
        doc.close()

    # REMOVED save_blocks() – no disk side effect
    return blocks


def _norm_line(text: str) -> str:
    """Normalize a line for boilerplate matching: lowercase, drop digits (page
    numbers vary), collapse whitespace. So 'FS-6' / 'Page 12' style footers match
    across pages regardless of the changing number."""
    import re as _re
    return _re.sub(r"\s+", " ", _re.sub(r"\d+", "", (text or "").lower())).strip()


def _detect_boilerplate(doc) -> set:
    """Lines that repeat in the top/bottom band of many pages = headers/footers."""
    from collections import Counter
    n = len(doc)
    if n < 3:
        return set()
    counts: Counter = Counter()
    for pno in range(n):
        page = doc[pno]
        h = page.rect.height or 1
        seen = set()
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                txt = " ".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if not txt:
                    continue
                y = line["bbox"][1]
                if y < 0.12 * h or y > 0.88 * h:   # header/footer band only
                    norm = _norm_line(txt)
                    if len(norm) >= 4 and norm not in seen:
                        counts[norm] += 1
                        seen.add(norm)
    threshold = max(3, int(0.4 * n))
    return {ln for ln, c in counts.items() if c >= threshold}


def _extract_paragraphs(page, page_num: int, filename: str, document_id: str,
                        boilerplate: set | None = None) -> List[NormalizedBlock]:
    """
    Extract text from a page, merging spans into paragraphs.
    Returns a list of NormalizedBlock (type="text" or "heading").
    """
    boilerplate = boilerplate or set()
    text_dict = page.get_text("dict")
    lines = []

    for block in text_dict.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            line_txt = " ".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if line_txt and _norm_line(line_txt) in boilerplate:
                continue  # drop repeating header/footer
            spans = []
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "size": span["size"],
                    "font": span.get("font", ""),
                    "bbox": span["bbox"],
                })
            if spans:
                # Combine bounding box of all spans in this line
                line_bbox = list(spans[0]["bbox"])
                for s in spans[1:]:
                    line_bbox[0] = min(line_bbox[0], s["bbox"][0])
                    line_bbox[1] = min(line_bbox[1], s["bbox"][1])
                    line_bbox[2] = max(line_bbox[2], s["bbox"][2])
                    line_bbox[3] = max(line_bbox[3], s["bbox"][3])
                line_text = " ".join(s["text"] for s in spans)
                avg_size = sum(s["size"] for s in spans) / len(spans)
                is_bold = any("Bold" in s["font"] or "bold" in s["font"].lower() for s in spans)
                lines.append({
                    "y0": line_bbox[1],
                    "y1": line_bbox[3],
                    "text": line_text,
                    "font_size": avg_size,
                    "is_bold": is_bold,
                })

    if not lines:
        return []

    lines.sort(key=lambda l: l["y0"])

    # Group into paragraphs by vertical gap
    paragraphs = []
    current_para = []
    prev_line = None
    for line in lines:
        if prev_line is None:
            current_para.append(line)
        else:
            prev_height = prev_line["y1"] - prev_line["y0"]
            gap = line["y0"] - prev_line["y1"]
            if gap < LINE_GAP_FACTOR * prev_height:
                current_para.append(line)
            else:
                paragraphs.append(current_para)
                current_para = [line]
        prev_line = line
    if current_para:
        paragraphs.append(current_para)

    # Convert each paragraph to a block
    result = []
    for para in paragraphs:
        para_text = " ".join(l["text"] for l in para)
        first_line = para[0]
        is_heading = (first_line["font_size"] >= 14) or (first_line["font_size"] >= 12 and first_line["is_bold"])
        block_type = "heading" if is_heading else "text"
        result.append(
            NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type=block_type,
                text=para_text,
                source_ref=SourceRef(filename=filename, page=page_num),
                confidence=1.0,
            )
        )
    return result