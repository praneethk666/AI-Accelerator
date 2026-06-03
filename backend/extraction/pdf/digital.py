"""Extract content from a digital PDF into NormalizedBlock list.
Merges spans into paragraphs to reduce fragmentation.
Includes bounding boxes for image and vector placeholders.
Vector drawings that overlap tables (by >50% area) are ignored.
Vector drawings with low complexity (few drawing items) are also ignored.
"""
import fitz
import uuid
from typing import List

from backend.core.schemas import NormalizedBlock, SourceRef
from backend.utils.save_json import save_blocks

# Configuration (move to config later)
MIN_IMAGE_AREA = 1000               # ignore logos/icons smaller than ~32x32 px
MIN_DRAWING_AREA_RATIO = 0.005      # 0.5% of page area -> captures small vector drawings
MIN_VECTOR_COMPLEXITY = 5           # minimum number of drawing items to be considered meaningful
LINE_GAP_FACTOR = 1.5               # lines closer than this * line_height are same paragraph
TABLE_OVERLAP_THRESHOLD = 0.5       # skip vector if >50% of its area overlaps a table


def extract_digital(pdf_path: str, document_id: str) -> List[NormalizedBlock]:
    doc = fitz.open(pdf_path)
    blocks: List[NormalizedBlock] = []
    filename = pdf_path.split("/")[-1]

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            page_area = page.rect.width * page.rect.height

            # ---- 1. Page metrics ----
            text = page.get_text().strip()
            text_len = len(text)
            raster_images = page.get_images(full=True)
            significant_image_count = 0
            for img in raster_images:
                rects = page.get_image_rects(img)
                if rects:
                    bbox = rects[0]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area >= MIN_IMAGE_AREA:
                        significant_image_count += 1

            drawings = page.get_drawings()
            vector_count = len(drawings)

            # Detect tables and collect their bounding boxes
            tables_found = page.find_tables()
            table_bboxes = [t.bbox for t in tables_found.tables] if tables_found.tables else []
            has_table = len(table_bboxes) > 0

            # Count significant vectors (area + complexity + table overlap filter)
            significant_vector_count = 0
            vector_bboxes_for_placeholders = []  # store rects that pass filters
            for dr in drawings:
                rect = dr.get("rect")
                if not rect:
                    continue
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                # Area threshold
                if area / page_area < MIN_DRAWING_AREA_RATIO:
                    continue   # too small – ignore

                # Complexity filter: skip simple drawings (e.g., single rectangle)
                items = dr.get("items", [])
                if len(items) < MIN_VECTOR_COMPLEXITY:
                    continue

                # Check overlap with any table
                skip = False
                for t_bbox in table_bboxes:
                    # intersection area
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
                    significant_vector_count += 1
                    vector_bboxes_for_placeholders.append(rect)

            metrics_block = NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type="page_metrics",
                text=f"Page {page_number} metrics",
                source_ref=SourceRef(filename=filename, page=page_number),
                confidence=1.0,
                metadata={
                    "text_length": text_len,
                    "raster_images_total": len(raster_images),
                    "significant_images": significant_image_count,
                    "vector_drawings_total": vector_count,
                    "significant_vectors": significant_vector_count,
                    "has_table": has_table,
                }
            )
            blocks.append(metrics_block)

            # ---- 2. Text as paragraphs (merged) ----
            para_blocks = _extract_paragraphs(page, page_number, filename, document_id)
            blocks.extend(para_blocks)

            # ---- 3. Tables ----
            try:
                tables = page.find_tables()
                for table in tables.tables:
                    extracted = table.extract()
                    if not extracted:
                        continue
                    headers = [str(h) for h in table.header.names] if table.header else []
                    rows = [[str(cell) for cell in row] for row in extracted]
                    blocks.append(
                        NormalizedBlock(
                            block_id=str(uuid.uuid4()),
                            document_id=document_id,
                            type="table",
                            table_data={"headers": headers, "rows": rows},
                            source_ref=SourceRef(filename=filename, page=page_number),
                            confidence=0.95,
                        )
                    )
            except Exception as e:
                print(f"Table extraction failed on page {page_number}: {e}")

            # ---- 4. Image placeholders (significant only) WITH BBOX ----
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
                print(f"Image extraction failed on page {page_number}: {e}")

            # ---- 5. Vector drawing placeholders (filtered: area + complexity + table overlap) ----
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
                print(f"Vector drawing extraction failed on page {page_number}: {e}")

    finally:
        doc.close()

    try:
        save_blocks(blocks, pdf_path)
    except Exception as e:
        print(f"Failed to save blocks JSON: {e}")

    return blocks


def _extract_paragraphs(page, page_num: int, filename: str, document_id: str) -> List[NormalizedBlock]:
    """
    Extract text from a page, merging spans into paragraphs.
    Returns a list of NormalizedBlock (type="text" or "heading").
    """
    text_dict = page.get_text("dict")
    lines = []

    for block in text_dict.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
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