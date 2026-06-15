"""Scanned PDF Handler – Surya OCR (primary) with PaddleOCR+YOLO fallback.

For scanned pages, we create placeholder `image` blocks (empty text) for each
detected visual region, just like the digital extraction does. The vision
enrichment tool will later convert them to `image_caption` with descriptions.

Public functions (used by page_profile.py):
  - page_to_pil
  - extract_ocr_text_and_boxes
  - detect_visual_regions
These remain unchanged and use PaddleOCR + YOLO (the fallback path).

The main extraction function `extract_scanned` first tries Surya (torch‑based,
full‑page layout+OCR) per page. If Surya fails (missing dependencies, runtime
error, etc.), it falls back to the Paddle‑based extraction for that page.
"""

import os
import io
import cv2
import fitz
import uuid
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image

from backend.core.schemas import NormalizedBlock, SourceRef

# ------------------------------------------------------------------
# Global models (for Paddle fallback)
# ------------------------------------------------------------------
_ocr_engine = None
_yolo_model = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            ocr_version='PP-OCRv4',
            show_log=False
        )
    return _ocr_engine


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
        except Exception:
            _yolo_model = None
    return _yolo_model


# ------------------------------------------------------------------
# Public functions (used by page_profile.py) – unchanged, Paddle‑based
# ------------------------------------------------------------------
def page_to_pil(page, dpi: int = 200) -> Image.Image:
    """Render a PDF page as a PIL image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def extract_ocr_text_and_boxes(pil_image: Image.Image) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Run OCR once and return both the full text and bounding boxes of all text lines.
    Returns (full_text, list_of_bboxes) where bbox = (x1,y1,x2,y2) in pixel coordinates.
    """
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
    return "\n".join(text_lines), text_boxes


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
    """
    img_w, img_h = pil_image.size
    page_area = img_w * img_h

    # ----- YOLO detection (if model available) -----
    yolo_boxes = []
    model = get_yolo_model()
    if model is not None:
        try:
            img_np = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.25, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            yolo_boxes.append((x1, y1, x2, y2))
        except Exception:
            pass

    # ----- Contour detection (text masked) -----
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
        if area < min_area:
            continue
        if w > img_w * 0.7 and h < 150:
            continue
        if area > 0.4 * page_area and page_text_lines > 80:
            continue
        filtered.append((x1, y1, x2, y2))
    return filtered


# ------------------------------------------------------------------
# Surya‑based extraction (primary)
# ------------------------------------------------------------------
def _surya_extract_page(page, page_number: int, filename: str, document_id: str) -> List[NormalizedBlock]:
    """Extract a single scanned page using Surya. Raises exception on failure."""
    # Import from the local ocr.py (same directory)
    try:
        from .ocr import surya_page
    except ImportError:
        raise ImportError("Surya backend not found. Make sure ocr.py exists in the same directory.")

    pil_img = page_to_pil(page, dpi=200)
    surya_res = surya_page(pil_img)

    page_rect = page.rect
    img_w, img_h = pil_img.size
    scale_x = page_rect.width / img_w
    scale_y = page_rect.height / img_h

    blocks = []

    # Text from Surya (split into paragraphs by double newline)
    if surya_res.text.strip():
        paragraphs = [p.strip() for p in surya_res.text.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Simple heuristic: heading if short and uppercase
            block_type = "heading" if (para.isupper() and len(para) < 100) else "text"
            blocks.append(
                NormalizedBlock(
                    block_id=str(uuid.uuid4()),
                    document_id=document_id,
                    type=block_type,
                    text=para,
                    source_ref=SourceRef(filename=filename, page=page_number),
                    confidence=0.9,
                )
            )

    # Visual regions
    for bbox_px in surya_res.regions:
        pdf_bbox = [bbox_px[0] * scale_x, bbox_px[1] * scale_y,
                    bbox_px[2] * scale_x, bbox_px[3] * scale_y]
        blocks.append(
            NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type="image",
                text="",
                source_ref=SourceRef(filename=filename, page=page_number, bbox=pdf_bbox),
                confidence=0.9,
                metadata={
                    "pending_vision": True,
                    "is_vector": False,
                    "is_full_page": False,
                    "detected_region": True,
                }
            )
        )
    return blocks


def _paddle_extract_page(page, page_number: int, filename: str, document_id: str, min_visual_area: int) -> List[NormalizedBlock]:
    """Extract a single scanned page using PaddleOCR + YOLO (fallback)."""
    page_rect = page.rect
    pil_img = page_to_pil(page, dpi=200)
    img_w, img_h = pil_img.size

    ocr_text, text_boxes = extract_ocr_text_and_boxes(pil_img)
    page_text_lines = len(ocr_text.splitlines())
    visual_regions = detect_visual_regions(pil_img, text_boxes, page_text_lines, min_area=min_visual_area)

    blocks = []

    # OCR text -> text/heading blocks
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

    # Visual region placeholders
    scale_x = page_rect.width / img_w
    scale_y = page_rect.height / img_h
    for (x1, y1, x2, y2) in visual_regions:
        pdf_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        blocks.append(
            NormalizedBlock(
                block_id=str(uuid.uuid4()),
                document_id=document_id,
                type="image",
                text="",
                source_ref=SourceRef(filename=filename, page=page_number, bbox=pdf_bbox),
                confidence=0.9,
                metadata={
                    "pending_vision": True,
                    "is_vector": False,
                    "is_full_page": False,
                    "detected_region": True,
                }
            )
        )
    return blocks


# ------------------------------------------------------------------
# Main extraction function – Surya first, fallback to Paddle
# ------------------------------------------------------------------
def extract_scanned(
    pdf_path: str,
    document_id: str,
    config: Optional[dict] = None,
    min_visual_area: int = 50000
) -> List[NormalizedBlock]:
    """
    Extract content from a scanned PDF.

    Tries to use Surya (torch‑based) on each page. If Surya fails on any page
    (e.g., missing dependencies, runtime error), falls back to PaddleOCR + YOLO
    for that page.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    filename = os.path.basename(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1

            # Attempt Surya first
            try:
                page_blocks = _surya_extract_page(page, page_number, filename, document_id)
            except Exception:
                # Fallback to Paddle
                page_blocks = _paddle_extract_page(page, page_number, filename, document_id, min_visual_area)

            all_blocks.extend(page_blocks)
    finally:
        doc.close()

    return all_blocks