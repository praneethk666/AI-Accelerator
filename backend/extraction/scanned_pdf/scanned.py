"""Scanned PDF Handler – YOLO + OCR, outputs NormalizedBlock list."""

import hashlib
import logging
import os
import io
import cv2
import fitz
import uuid
import numpy as np

from collections import OrderedDict
from PIL import Image
from typing import List, Optional, Tuple

# NOTE: paddleocr is imported LAZILY inside get_ocr_engine(), not here.
# Importing paddle BEFORE torch initializes (the nomic embedder) corrupts torch's
# tensor allocator -> "Tensor holds no memory" crash. Lazy import keeps paddle out
# of the process for digital PDFs entirely, and the model warm-up
# (backend.core.models.warm_up) loads torch first so scanned+embed also coexist.

from backend.core.schemas import NormalizedBlock, SourceRef

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global models (loaded once)
# ------------------------------------------------------------------
_ocr_engine = None
_yolo_model = None

# Per-page result caches keyed by rendered-image content hash. The profiler and
# the extractor both render the same page @200 DPI and run OCR + region detection;
# memoizing here means the expensive work happens ONCE per page, not twice.
_CACHE_MAX = 64
_ocr_cache: "OrderedDict[str, tuple]" = OrderedDict()
_region_cache: "OrderedDict[tuple, list]" = OrderedDict()


def _img_hash(pil_image: "Image.Image") -> str:
    return hashlib.md5(pil_image.tobytes()).hexdigest()


def _cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX:
        cache.popitem(last=False)

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        # lazy import (see module note): warm torch first so paddle can't corrupt it
        from backend.core.models import warm_up
        warm_up()
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            ocr_version='PP-OCRv4',
            show_log=False
        )
    return _ocr_engine

# Document-layout detector. We use DocLayout-YOLO (trained on document elements:
# title/text/table/figure/...), NOT a COCO model — COCO detects people/cars and
# is the wrong tool for documents. If the package or weights aren't available we
# return None and fall back to contour detection (still works, just less precise).
_LAYOUT_FIGURE_CLASSES = {"figure", "table", "isolate_formula", "chart"}
_layout_unavailable = False


def get_layout_model():
    global _yolo_model, _layout_unavailable
    if _yolo_model is None and not _layout_unavailable:
        try:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )
            _yolo_model = YOLOv10(weights)
            logger.info("DocLayout-YOLO loaded for document layout detection")
        except Exception as e:
            logger.warning(
                "DocLayout-YOLO unavailable (%s); using contour detection only", e
            )
            _layout_unavailable = True
            _yolo_model = None
    return _yolo_model


def _layout_region_boxes(pil_image: "Image.Image"):
    """Figure/table/chart boxes from the doc-layout model, or [] if unavailable."""
    model = get_layout_model()
    if model is None:
        return []
    try:
        result = model.predict(np.array(pil_image), imgsz=1024, conf=0.25, verbose=False)[0]
        names = result.names
        boxes = []
        for b in result.boxes:
            cls_name = names.get(int(b.cls), "") if isinstance(names, dict) else ""
            if cls_name in _LAYOUT_FIGURE_CLASSES:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        logger.warning("DocLayout-YOLO inference failed: %s", e)
        return []


def page_to_pil(page, dpi: int = 200) -> Image.Image:
    """Render a PDF page as a PIL image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def extract_ocr_text_and_boxes(pil_image: Image.Image) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Run OCR once and return both the full text and bounding boxes of all text lines.
    Returns (full_text, list_of_bboxes) where bbox = (x1,y1,x2,y2) in pixel coordinates.
    Memoized by image content so the profiler + extractor share one OCR pass.
    """
    key = _img_hash(pil_image)
    cached = _cache_get(_ocr_cache, key)
    if cached is not None:
        return cached

    ocr = get_ocr_engine()
    img_np = np.array(pil_image)
    result = ocr.ocr(img_np)

    text_lines = []
    text_boxes = []
    if result and result[0]:
        for line in result[0]:
            text_lines.append(line[1][0])
            box = line[0]
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            text_boxes.append((min(xs), min(ys), max(xs), max(ys)))
    out = ("\n".join(text_lines), text_boxes)
    _cache_put(_ocr_cache, key, out)
    return out


def detect_visual_regions(
    pil_image: Image.Image,
    text_boxes: List[Tuple[int, int, int, int]],
    page_text_lines: int,
    min_area: int = 50000   # pixels² at 200 DPI (~224x224)
) -> List[Tuple[int, int, int, int]]:
    """
    Use YOLO + contour detection to find meaningful non‑text regions.
    Returns list of bboxes (x1,y1,x2,y2) in pixel coordinates.
    Only regions with area >= min_area are considered significant.
    Memoized by (image, min_area) — detection is deterministic per page, so the
    profiler + extractor share one detection pass.
    """
    cache_key = (_img_hash(pil_image), int(min_area))
    cached = _cache_get(_region_cache, cache_key)
    if cached is not None:
        return cached

    img_w, img_h = pil_image.size
    page_area = img_w * img_h

    # ----- Document-layout detection (figures/tables/charts) -----
    yolo_boxes = _layout_region_boxes(pil_image)

    # ----- Contour detection (text masked) — complements/serves as fallback -----
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Mask out text regions
    for (x1, y1, x2, y2) in text_boxes:
        pad = 4
        cv2.rectangle(thresh,
                      (max(0, x1 - pad), max(0, y1 - pad)),
                      (min(img_w, x2 + pad), min(img_h, y2 + pad)),
                      0, -1)

    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue
        contour_boxes.append((x, y, x + w, y + h))

    # ----- Combine and merge overlapping boxes (IOU > 0.3) -----
    all_boxes = yolo_boxes + contour_boxes
    if not all_boxes:
        return []

    # Simple merging
    merged = list(all_boxes)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                ix1 = max(x1, bx1)
                iy1 = max(y1, by1)
                ix2 = min(x2, bx2)
                iy2 = min(y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.3:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((x1, y1, x2, y2))
            used[i] = True
        merged = new_merged

    # ----- Final filtering: area and shape heuristics -----
    filtered = []
    for (x1, y1, x2, y2) in merged:
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Minimum area threshold (ignores tiny logos/decorations)
        if area < min_area:
            continue

        # Avoid long horizontal strips (likely headers/footers)
        if w > img_w * 0.7 and h < 150:
            continue

        # Avoid huge regions on text‑heavy pages (false positive table detection)
        if area > 0.4 * page_area and page_text_lines > 80:
            continue

        filtered.append((x1, y1, x2, y2))

    _cache_put(_region_cache, cache_key, filtered)
    return filtered


def extract_scanned(
    pdf_path: str,
    document_id: str,
    use_gemma: bool = False,
    min_visual_area: int = 50000
) -> List[NormalizedBlock]:
    """
    Extract content from a scanned PDF using OCR + YOLO for visual regions.

    Args:
        pdf_path: Path to the scanned PDF.
        document_id: Unique document identifier.
        use_gemma: If True, send detected visual regions to Gemma for description
                   (placeholder – implement your own client).
        min_visual_area: Minimum area (pixels² at 200 DPI) for a region to be
                         considered a significant visual element. Default 50000.

    Returns:
        List[NormalizedBlock] containing:
        - text blocks (type="text" or "heading") from OCR
        - image_caption blocks for each detected visual region (with bbox)
        - (page_metrics block removed – handled by page_profile)
    """
    doc = fitz.open(pdf_path)
    blocks: List[NormalizedBlock] = []
    filename = os.path.basename(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            page_rect = page.rect

            # ---- Render page as image (for OCR and visual detection) ----
            pil_img = page_to_pil(page, dpi=200)
            img_w, img_h = pil_img.size

            # ---- OCR: get text and text bounding boxes ----
            ocr_text, text_boxes = extract_ocr_text_and_boxes(pil_img)
            page_text_lines = len(ocr_text.splitlines())

            # ---- Detect meaningful visual regions (YOLO + contours) ----
            visual_regions = detect_visual_regions(
                pil_img, text_boxes, page_text_lines,
                min_area=min_visual_area
            )

            # ---- Page metrics block REMOVED (handled by page_profile) ----

            # ---- Convert OCR text to NormalizedBlock (text/heading) ----
            # Simple grouping into paragraphs by empty lines
            paragraphs = []
            current_para = []
            for line in ocr_text.split('\n'):
                line = line.strip()
                if not line:
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                else:
                    current_para.append(line)
            if current_para:
                paragraphs.append(" ".join(current_para))

            for para in paragraphs:
                block_type = "heading" if (para.isupper() and len(para) < 100) else "text"
                blocks.append(
                    NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type=block_type,
                        text=para,
                        source_ref=SourceRef(filename=filename, page=page_number),
                        confidence=0.8,
                    )
                )

            # ---- Create placeholders for each visual region ----
            scale_x = page_rect.width / img_w
            scale_y = page_rect.height / img_h
            for (x1, y1, x2, y2) in visual_regions:
                pdf_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

                # (Optional) send cropped region to Gemma
                description = None
                if use_gemma:
                    # Placeholder – replace with actual gemma_client call
                    # from backend.vision.gemma_client import describe_image_with_gemma
                    # cropped = pil_img.crop((x1, y1, x2, y2))
                    # img_bytes = io.BytesIO()
                    # cropped.save(img_bytes, format="PNG")
                    # description = describe_image_with_gemma(img_bytes.getvalue())
                    pass

                blocks.append(
                    NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type="image_caption",
                        text=description or "[Image - awaiting vision enrichment]",
                        source_ref=SourceRef(
                            filename=filename,
                            page=page_number,
                            bbox=pdf_bbox
                        ),
                        confidence=0.5,
                        metadata={
                            "pending_vision": not bool(description),
                            "is_vector": False,
                            "detected_region": True,
                        }
                    )
                )

    finally:
        doc.close()
    return blocks