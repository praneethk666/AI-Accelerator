"""Per-page x-ray (required by Vishal)."""

import fitz
from typing import List

from backend.core.schemas import PageProfile, ImageRegion

# For scanned page image detection (fallback)
from backend.extraction.scanned_pdf.scanned import page_to_pil, extract_ocr_text_and_boxes, detect_visual_regions

# ===== CONFIGURATION =====
# Use the same MIN_IMAGE_AREA as in digital.py (1000 points²)
# This ensures profile and extraction agree on which raster images are significant.
MIN_IMAGE_AREA = 1000          # ignore logos/icons smaller than ~32x32 px (area)
MIN_VECTOR_AREA_RATIO = 0.010
MIN_VECTOR_COMPLEXITY = 5
MIN_SCANNED_IMAGE_AREA = 50000


def page_profile(pdf_path: str) -> List[PageProfile]:
    """
    Analyze every page of a PDF and generate PageProfile metadata.
    - Table detection ONLY on digital pages (fast, accurate).
    - Scanned pages: table_hint = False (to avoid false positives).
    - For digital pages with vector graphics, a full‑page ImageRegion is added
      to the `images` list to keep the profile and extracted blocks in sync.
    - Raster image inclusion uses the same MIN_IMAGE_AREA threshold as extraction.
    - Scanned pages use Surya for text and region detection (fallback to Paddle+YOLO).
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
            table_hint = False

            if has_native_text:
                # Digital page: fast native table detection
                try:
                    tables = page.find_tables()
                    table_hint = len(tables.tables) > 0 if tables.tables else False
                except Exception:
                    table_hint = False

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

            if not has_native_text:   # Scanned page – use Surya first
                try:
                    from backend.extraction.scanned_pdf.ocr import surya_page
                    pil_img = page_to_pil(page, dpi=200)
                    surya_res = surya_page(pil_img)

                    # Get text length from Surya
                    text_len = len(surya_res.text)

                    # Convert Surya regions to ImageRegion objects
                    scale_x = page_rect.width / pil_img.width
                    scale_y = page_rect.height / pil_img.height
                    for bbox_px in surya_res.regions:
                        pdf_bbox = [
                            bbox_px[0] * scale_x,
                            bbox_px[1] * scale_y,
                            bbox_px[2] * scale_x,
                            bbox_px[3] * scale_y
                        ]
                        width = int((bbox_px[2] - bbox_px[0]) * scale_x)
                        height = int((bbox_px[3] - bbox_px[1]) * scale_y)
                        images.append(ImageRegion(
                            bbox=pdf_bbox,
                            width=width,
                            height=height,
                            significant=True
                        ))
                except Exception:
                    # Fallback to PaddleOCR + YOLO if Surya fails
                    try:
                        pil_img = page_to_pil(page, dpi=200)
                        img_w, img_h = pil_img.size

                        ocr_text, text_boxes = extract_ocr_text_and_boxes(pil_img)
                        text_len = len(ocr_text)
                        page_text_lines = len(ocr_text.splitlines())

                        vis_regions = detect_visual_regions(
                            pil_img, text_boxes, page_text_lines,
                            min_area=MIN_SCANNED_IMAGE_AREA
                        )

                        scale_x = page_rect.width / img_w
                        scale_y = page_rect.height / img_h
                        seen_vis_bboxes = set()
                        for (x1, y1, x2, y2) in vis_regions:
                            pdf_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                            rounded_bbox = tuple(round(c, 2) for c in pdf_bbox)
                            if rounded_bbox in seen_vis_bboxes:
                                continue
                            seen_vis_bboxes.add(rounded_bbox)
                            width = int((x2 - x1) * scale_x)
                            height = int((y2 - y1) * scale_y)
                            images.append(ImageRegion(
                                bbox=pdf_bbox,
                                width=width,
                                height=height,
                                significant=True
                            ))
                    except Exception:
                        # Ultimate fallback: no images, keep text_len as 0
                        text_len = 0

            else:   # Digital page – use same area filter as extraction
                try:
                    seen_digital_bboxes = set()
                    for img in page.get_image_info(xrefs=True):
                        width = int(img["width"])
                        height = int(img["height"])
                        bbox = list(img["bbox"])
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        area = w * h
                        # Skip if area below MIN_IMAGE_AREA (matches extraction)
                        if area < MIN_IMAGE_AREA:
                            continue
                        # Skip full‑page background images (cover the whole page)
                        if area / page_area > 0.95:
                            continue
                        rounded_bbox = tuple(round(c, 2) for c in bbox)
                        if rounded_bbox in seen_digital_bboxes:
                            continue
                        seen_digital_bboxes.add(rounded_bbox)
                        images.append(ImageRegion(
                            bbox=bbox,
                            width=width,
                            height=height,
                            significant=True,
                        ))
                except Exception:
                    pass

            # ---- Add full‑page image region if the page has vector graphics ----
            if has_vector_graphics:
                full_page_bbox = [0.0, 0.0, page_rect.width, page_rect.height]
                images.append(ImageRegion(
                    bbox=full_page_bbox,
                    width=int(page_rect.width),
                    height=int(page_rect.height),
                    significant=True,
                ))

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

    finally:
        doc.close()
    return profiles