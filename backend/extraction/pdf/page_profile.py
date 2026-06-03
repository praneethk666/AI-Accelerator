"""Per-page x-ray (required by Vishal)."""

import fitz
from typing import List

from backend.core.schemas import PageProfile, ImageRegion
from backend.utils.save_json import save_page_profiles

# ===== CONFIGURATION (tune per document type) =====
MIN_IMAGE_PX = 150               # Ignore images smaller than this (both dimensions)
MIN_VECTOR_AREA_RATIO = 0.005   # 0.5% of page area – captures small vector drawings (icons, diagrams)
MIN_VECTOR_COMPLEXITY = 5        # Minimum number of drawing items to be considered meaningful


def page_profile(pdf_path: str) -> List[PageProfile]:
    """
    Analyze every page of a PDF and generate PageProfile metadata.
    Only significant vector drawings (area >= MIN_VECTOR_AREA_RATIO * page_area)
    AND with at least MIN_VECTOR_COMPLEXITY items are considered.
    Vectors that overlap tables by >50% are also skipped.
    """
    doc = fitz.open(pdf_path)
    profiles: List[PageProfile] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            # ---- Text ----
            text = page.get_text().strip()
            text_len = len(text)

            # ---- Detect tables (for vector filtering) ----
            try:
                tables = page.find_tables()
                table_bboxes = [t.bbox for t in tables.tables] if tables.tables else []
                table_hint = len(table_bboxes) > 0
            except Exception:
                table_bboxes = []
                table_hint = False

            # ---- Vector graphics (significant only, exclude table‑overlapping and simple ones) ----
            drawings = page.get_drawings()
            significant_vector_count = 0
            for dr in drawings:
                rect = dr.get("rect")
                if not rect:
                    continue
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if area / page_area < MIN_VECTOR_AREA_RATIO:
                    continue   # too small – ignore

                # Complexity filter: skip simple drawings (e.g., single rectangle)
                items = dr.get("items", [])
                if len(items) < MIN_VECTOR_COMPLEXITY:
                    continue

                # Skip if drawing heavily overlaps any table (likely a table border)
                skip = False
                for t_bbox in table_bboxes:
                    # intersection area
                    ix1 = max(rect[0], t_bbox[0])
                    iy1 = max(rect[1], t_bbox[1])
                    ix2 = min(rect[2], t_bbox[2])
                    iy2 = min(rect[3], t_bbox[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / area > 0.5:   # >50% of drawing inside table
                            skip = True
                            break
                if not skip:
                    significant_vector_count += 1
            has_vector_graphics = significant_vector_count > 0

            # ---- Images (raster) – only those meeting size threshold ----
            images: List[ImageRegion] = []
            try:
                for img in page.get_image_info(xrefs=True):
                    width = int(img["width"])
                    height = int(img["height"])
                    if width >= MIN_IMAGE_PX and height >= MIN_IMAGE_PX:
                        bbox = list(img["bbox"])
                        images.append(
                            ImageRegion(
                                bbox=bbox,
                                width=width,
                                height=height,
                                significant=True,
                            )
                        )
            except Exception:
                pass

            # ---- Page classification (old project rule) ----
            kind = "digital" if text_len > 5 else "scanned"

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
                f"sig_vectors={significant_vector_count} | "
                f"table={table_hint} | "
                f"type={kind}"
            )

    finally:
        doc.close()

    save_page_profiles(profiles, pdf_path)
    return profiles