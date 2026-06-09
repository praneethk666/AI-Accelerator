"""Per-page x-ray (required by Vishal)."""

import fitz
from typing import List

from backend.core.schemas import PageProfile, ImageRegion

# For scanned page image detection – updated import
from backend.extraction.scanned_pdf.scanned import page_to_pil, extract_ocr_text_and_boxes, detect_visual_regions

# ===== CONFIGURATION =====
MIN_IMAGE_PX = 150
MIN_VECTOR_AREA_RATIO = 0.010
MIN_VECTOR_COMPLEXITY = 5
MIN_SCANNED_IMAGE_AREA = 50000


def page_profile(pdf_path: str) -> List[PageProfile]:
    """
    Analyze every page of a PDF and generate PageProfile metadata.
    - Table detection ONLY on digital pages (fast, accurate).
    - Scanned pages: table_hint = False (to avoid false positives).
    """
    doc = fitz.open(pdf_path)
    profiles: List[PageProfile] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            # ---- Check for native text (digital) ----
            native_text = page.get_text().strip()
            has_native_text = len(native_text) > 5

            text_len = len(native_text) if has_native_text else 0
            ocr_text = None
            table_hint = False

            if has_native_text:
                # Digital page: fast native table detection
                try:
                    tables = page.find_tables()
                    table_hint = len(tables.tables) > 0 if tables.tables else False
                except Exception:
                    table_hint = False
            # Scanned pages: table detection is unreliable, so skip it.

            # ---- Vector graphics (only digital) ----
            has_vector_graphics = False
            if has_native_text:
                drawings = page.get_drawings()
                significant_vector_count = 0
                for dr in drawings:
                    rect = dr.get("rect")
                    if not rect:
                        continue
                    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                    if area / page_area < MIN_VECTOR_AREA_RATIO:
                        continue
                    items = dr.get("items", [])
                    if len(items) < MIN_VECTOR_COMPLEXITY:
                        continue
                    # Skip if drawing overlaps a table
                    skip = False
                    if table_hint:
                        try:
                            tables = page.find_tables()
                            if tables.tables:
                                for t in tables.tables:
                                    # simple overlap check
                                    ix1 = max(rect[0], t.bbox[0])
                                    iy1 = max(rect[1], t.bbox[1])
                                    ix2 = min(rect[2], t.bbox[2])
                                    iy2 = min(rect[3], t.bbox[3])
                                    if ix2 > ix1 and iy2 > iy1:
                                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                                        if inter_area / area > 0.5:
                                            skip = True
                                            break
                        except:
                            pass
                    if not skip:
                        significant_vector_count += 1
                has_vector_graphics = significant_vector_count > 0

            # ---- Images (raster) ----
            images: List[ImageRegion] = []

            if not has_native_text:   # Scanned page
                try:
                    pil_img = page_to_pil(page, dpi=200)
                    img_w, img_h = pil_img.size

                    # OCR to get text length
                    ocr_text, text_boxes = extract_ocr_text_and_boxes(pil_img)
                    text_len = len(ocr_text)
                    page_text_lines = len(ocr_text.splitlines())

                    # Detect visual regions (images, diagrams)
                    vis_regions = detect_visual_regions(
                        pil_img, text_boxes, page_text_lines,
                        min_area=MIN_SCANNED_IMAGE_AREA
                    )

                    scale_x = page_rect.width / img_w
                    scale_y = page_rect.height / img_h
                    for (x1, y1, x2, y2) in vis_regions:
                        pdf_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        width = int((x2 - x1) * scale_x)
                        height = int((y2 - y1) * scale_y)
                        images.append(ImageRegion(
                            bbox=pdf_bbox,
                            width=width,
                            height=height,
                            significant=True
                        ))
                except Exception as e:
                    print(f"Scanned page detection failed on page {page_num+1}: {e}")

            else:   # Digital page
                try:
                    for img in page.get_image_info(xrefs=True):
                        width = int(img["width"])
                        height = int(img["height"])
                        if width >= MIN_IMAGE_PX and height >= MIN_IMAGE_PX:
                            bbox = list(img["bbox"])
                            img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if img_area / page_area > 0.95:
                                continue
                            images.append(ImageRegion(
                                bbox=bbox,
                                width=width,
                                height=height,
                                significant=True,
                            ))
                except Exception:
                    pass

            # ---- Page classification ----
            kind = "digital" if has_native_text else "scanned"

            profile = PageProfile(
                page_number=page_num + 1,
                kind=kind,
                text_len=text_len,
                has_vector_graphics=has_vector_graphics,
                table_hint=table_hint,
                images=images,
            )
            profiles.append(profile)

            print(
                f"Page {page_num + 1} | "
                f"text={text_len} | "
                f"images={len(images)} | "
                f"vectors={'true' if has_vector_graphics else 'false'} | "
                f"table={table_hint} | "
                f"type={kind}"
            )

    finally:
        doc.close()
    return profiles