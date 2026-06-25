"""Scanned PDF Handler – YOLO + OCR, outputs NormalizedBlock list."""

import hashlib
import json
import logging
import os
import io
import cv2
import fitz
import uuid
import numpy as np
import re
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
_surya_cache: "OrderedDict[str, object]" = OrderedDict()

# OCR engine: "surya" (default) or "paddle". Settable from config via
# set_ocr_engine(); overridable by the OCR_ENGINE env var. Surya does text +
# layout in one pass and is torch-based (no paddle<->torch conflict); paddle is
# the fallback. Surya needs the llama-server binary (brew install llama.cpp).
_OCR_ENGINE = os.getenv("OCR_ENGINE", "surya").lower()

# Surya can STALL on a cold/unhealthy llama-server (observed: 20+ min at 0% CPU on
# a cold start). The existing except-based fallback only catches errors, not hangs,
# so we also bound every Surya page call with a wall-clock timeout and fall back to
# PaddleOCR / the contour detector. Configurable via ocr.surya_timeout_s
# (set_surya_timeout) or the SURYA_TIMEOUT_S env var.
_SURYA_TIMEOUT_S = float(os.getenv("SURYA_TIMEOUT_S", "90"))
# Image hashes where Surya already timed out THIS document — skip it so we don't
# pay the timeout twice (once for OCR text, once for region detection of the same
# page). Cleared at the start of each extract_scanned() so a recovered llama-server
# is retried on the next document.
_surya_failed_keys: set = set()


def set_ocr_engine(name: Optional[str]) -> None:
    global _OCR_ENGINE
    if name:
        _OCR_ENGINE = name.lower()


def set_surya_timeout(secs) -> None:
    global _SURYA_TIMEOUT_S
    try:
        if secs:
            _SURYA_TIMEOUT_S = float(secs)
    except (TypeError, ValueError):
        pass


def _surya_with_timeout(pil_image):
    """Run Surya for ONE page, bounded by _SURYA_TIMEOUT_S. Raises TimeoutError on a
    stall. We shut the worker down with wait=False: a blocked llama.cpp call can't be
    cancelled, so we abandon the thread rather than block the whole ingestion on it.
    (_get_surya_page memoizes, so a successful page is reused with no second call.)"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_get_surya_page, pil_image)
    try:
        result = fut.result(timeout=_SURYA_TIMEOUT_S)
        ex.shutdown(wait=False)
        return result
    except _FTimeout:
        ex.shutdown(wait=False)
        raise TimeoutError(f"Surya page exceeded {_SURYA_TIMEOUT_S}s")


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
        os.environ["FLAGS_use_mkldnn"] = "0"
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang='en',
            use_textline_orientation=True,
            # ocr_version='PP-OCRv4',
            # show_log=False,
            enable_hpi=False
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


def _get_surya_page(pil_image: "Image.Image"):
    """Run Surya once per page (text + layout regions), memoized by image content."""
    key = _img_hash(pil_image)
    cached = _cache_get(_surya_cache, key)
    if cached is not None:
        return cached
    from backend.extraction.scanned_pdf.ocr_backends import surya_page

    page = surya_page(pil_image)
    _cache_put(_surya_cache, key, page)
    return page


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

    # Surya: text comes from the one-pass full-page result. Fall back to Paddle
    # if Surya is unavailable (e.g. no llama-server), errors, OR stalls (timeout).
    if _OCR_ENGINE == "surya" and key not in _surya_failed_keys:
        try:
            page = _surya_with_timeout(pil_image)
            out = (page.text, page.text_boxes)
            _cache_put(_ocr_cache, key, out)
            return out
        except Exception as e:
            _surya_failed_keys.add(key)
            logger.warning("Surya OCR failed/timed out (%s); falling back to PaddleOCR", e)
    ocr = get_ocr_engine()
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = ocr.predict(img_np)

    text_lines = []
    text_boxes = []
    for page in results:
        rec_texts = page["rec_texts"]
        rec_boxes = page["rec_boxes"]

        for txt, box in zip(rec_texts, rec_boxes):
            text_lines.append(txt)

            x1, y1, x2, y2 = map(int, box)

            text_boxes.append(
                (x1, y1, x2, y2)
            )
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

    # Surya: visual regions come from the SAME one-pass result as the text
    # (no separate layout model). Fall back to DocLayout/contours on failure or
    # timeout. If OCR already timed Surya out for this page, skip straight to the
    # detector instead of paying the timeout again.
    ihash = cache_key[0]
    if _OCR_ENGINE == "surya" and ihash not in _surya_failed_keys:
        try:
            page = _surya_with_timeout(pil_image)
            regions = [
                b for b in page.regions
                if (b[2] - b[0]) * (b[3] - b[1]) >= min_area
            ]
            _cache_put(_region_cache, cache_key, regions)
            return regions
        except Exception as e:
            _surya_failed_keys.add(ihash)
            logger.warning("Surya layout failed/timed out (%s); falling back to detector", e)

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
    all_boxes = list(dict.fromkeys(all_boxes))
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

        # Minimum area threshold
        if area < min_area:
            continue

        # Avoid long horizontal strips (likely headers/footers)
        if w > img_w * 0.7 and h < 150:
            continue

        # Avoid huge regions on text-heavy pages
        if area > 0.4 * page_area and page_text_lines > 80:
            continue

        # -----------------------------
        # OCR boxes inside this region
        # -----------------------------
        inside_boxes = []

        for bx1, by1, bx2, by2 in text_boxes:

            # overlap test
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)

            if ix2 > ix1 and iy2 > iy1:
                inside_boxes.append((bx1, by1, bx2, by2))

        # Too many OCR boxes ⇒ paragraph text
        if len(inside_boxes) > 40:
            continue

        # -----------------------------
        # Text coverage
        # -----------------------------
        text_area = sum(
            (bx2 - bx1) * (by2 - by1)
            for bx1, by1, bx2, by2 in inside_boxes
        )

        coverage = text_area / area

        # Region dominated by text
        if coverage > 0.75:
            continue

        filtered.append((x1, y1, x2, y2))

    _cache_put(_region_cache, cache_key, filtered)
    return filtered

def table_to_markdown(table_data):

    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    if not headers and rows:
        headers = [f"Column{i+1}" for i in range(len(rows[0]))]

    lines = []

    lines.append(
        "| " + " | ".join(headers) + " |"
    )

    lines.append(
        "| " + " | ".join(["---"]*len(headers)) + " |"
    )

    for row in rows:

        values = [str(x).replace("\n","<br>") for x in row]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)

def extract_scanned(
    pdf_path: str,
    document_id: str,
    config: dict = None,
    min_visual_area: int = 50000,
    out_queue=None,          # if set, stream ("page", page_num, blocks_dicts) per page
) -> List[NormalizedBlock]:
    """
    Extract content from a scanned PDF using OCR + YOLO for visual regions.

    When out_queue is provided the function streams results incrementally:
      ("page", page_num, [block_dict, ...])   — after each page completes
      ("page_error", page_num, reason)         — when a page is skipped
    The return value still contains all successfully extracted blocks so
    callers that don't use the queue continue to work unchanged.

    Per-page and per-region errors are caught and skipped — a bad page or a
    hung vision call never aborts the whole document.
    """
    # Fresh Surya health per document: retry it even if it stalled on a prior doc
    # (the llama-server may have recovered). Within THIS doc, a stalled page still
    # short-circuits to Paddle so we never pay the timeout twice for one page.
    _surya_failed_keys.clear()

    doc = fitz.open(pdf_path)
    blocks: List[NormalizedBlock] = []
    filename = os.path.basename(pdf_path)

    from backend.core.vision_client import describe_image
    from backend.core.schemas import as_dicts

    total_pages = len(doc)
    try:
        for page_num in range(total_pages):
            page = doc[page_num]
            page_number = page_num + 1
            page_rect = page.rect
            page_blocks: List[NormalizedBlock] = []   # blocks for THIS page only

            # ── Per-page try: a crash/hang on one page skips it, never kills all ──
            try:
                # ---- Render page as image ----
                pil_img = page_to_pil(page, dpi=200)
                img_w, img_h = pil_img.size
                logger.info("Page %d/%d: rendered (%dx%d)", page_number, total_pages, img_w, img_h)

                # ---- OCR text + bounding boxes ----
                ocr_text, text_boxes = extract_ocr_text_and_boxes(pil_img)
                page_text_lines = len(ocr_text.splitlines())
                logger.info("Page %d: %d chars, %d text boxes", page_number, len(ocr_text), len(text_boxes))

                # ---- Detect visual regions ----
                visual_regions = detect_visual_regions(
                    pil_img, text_boxes, page_text_lines,
                    min_area=min_visual_area
                )
                logger.info("Page %d: %d visual regions", page_number, len(visual_regions))

                scale_x = page_rect.width / img_w
                scale_y = page_rect.height / img_h
                table_texts: List[str] = []
                image_descriptions: List[str] = []

                # ---- Enrich each visual region ----
                for region_idx, (x1, y1, x2, y2) in enumerate(visual_regions):
                    # ── Per-region try: a bad vision call skips this region only ──
                    try:
                        pdf_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        block_id = str(uuid.uuid4())

                        cropped_buf = io.BytesIO()
                        pil_img.crop((x1, y1, x2, y2)).save(cropped_buf, format="PNG")
                        image_bytes = cropped_buf.getvalue()

                        # Step 1 — classify region
                        CLASSIFY_PROMPT = """
                                        Classify the region into exactly one category:

                                        table
                                        image
                                        text

                                        Definitions:

                                        table:
                                        - rows and columns
                                        - key-value tables

                                        image:
                                        - screenshots
                                        - photographs
                                        - diagrams
                                        - figures
                                        - logos

                                        text:
                                        - paragraphs
                                        - bullet lists
                                        - instructions
                                        - headings
                                        - code blocks

                                        Return ONLY one word.

                                        table
                                        image
                                        text
                                    """
                        raw_kind = describe_image(image_bytes, CLASSIFY_PROMPT, config)
                        raw_kind = raw_kind.strip().lower()
                        if raw_kind.endswith("table"):
                            kind = "table"
                        elif raw_kind.endswith("image"):
                            kind = "image"
                        else:
                            kind = "text"

                        if kind == "table":
                            TABLE_PROMPT = """
                                    Determine whether the region actually contains a table.

                                    If no table exists return:

                                    {
                                    "headers": [],
                                    "rows": []
                                    }

                                    Do not invent rows.
                                    If the table is a key-value form with labels in the first column and values in the second column, use:
                                    {
                                        "headers": ["Field", "Value"]
                                    }

                                    and store each label-value pair as one row.

                                    Preserve multiline cell contents.
                                    DO NOT explain.
                                    DO NOT think step by step.
                                    DO NOT reason.
                                    DO NOT use markdown.
                                    DO NOT output ```json.
                                    Start with {
                                    End with }
                                    Output exactly one JSON object.

                                    Return ONLY JSON.
                                    Return Exactly as output format shown.
                                    OUTPUT FORMAT:
                                    {
                                    "headers": ["COLNAME1", "COLNAME2",...],
                                    "rows": [["ROW1COL1", "ROW1COL2",..], ["ROW2COL1", "ROW2COL2",..], ..]
                                    }
                                    """
                            raw_response = describe_image(image_bytes, TABLE_PROMPT, config)
                            clean = raw_response.replace("```json", "").replace("```", "")
                            table_data = None
                            for candidate in reversed(re.findall(r"\{[\s\S]*?\}", clean)):
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict) and "headers" in obj and "rows" in obj:
                                        table_data = obj
                                        break
                                except Exception:
                                    pass

                            if table_data is None or (
                                len(table_data.get("headers", [])) == 0
                                and len(table_data.get("rows", [])) == 0
                            ):
                                continue

                            flat_text = [str(cell) for row in table_data.get("rows", []) for cell in row]
                            table_texts.append(" ".join(flat_text))
                            markdown_text = table_to_markdown(table_data)
                            block = NormalizedBlock(
                                block_id=block_id,
                                document_id=document_id,
                                type="table",
                                text=markdown_text,
                                table_data=table_data,
                                source_ref=SourceRef(filename=filename, page=page_number, bbox=pdf_bbox),
                                confidence=0.85,
                                metadata={"detected_region": True, "region_kind": "table"},
                            )
                            page_blocks.append(block)

                        elif kind == "image":
                            raw_image_path = f"uploads/images/{document_id}/{block_id}_raw.png"
                            os.makedirs(os.path.dirname(raw_image_path), exist_ok=True)
                            with open(raw_image_path, "wb") as f:
                                f.write(image_bytes)
                            IMAGE_PROMPT = """
                                Describe the image.

                                Return ONLY JSON:

                                {
                                "type":"",
                                "description":"",
                                "entities":[],
                                "confidence":0.0
                                }

                                No reasoning.
                                No markdown.
                                """
                            description = describe_image(image_bytes, IMAGE_PROMPT, config)
                            if description:
                                image_descriptions.append(description)
                            block = NormalizedBlock(
                                block_id=block_id,
                                document_id=document_id,
                                type="image",
                                text=description or "[Image - awaiting vision enrichment]",
                                source_ref=SourceRef(filename=filename, page=page_number, bbox=pdf_bbox),
                                confidence=0.85 if description else 0.5,
                                metadata={
                                    "raw_image_path": raw_image_path,
                                    "image_path": None,
                                    "pending_vision": not bool(description),
                                    "is_vector": False,
                                    "detected_region": True,
                                    "region_kind": kind,
                                },
                            )
                            page_blocks.append(block)

                        # kind == "text" → fall through; OCR text handles it below

                    except Exception as region_exc:
                        # Vision call timed out, API error, JSON failure — skip region
                        logger.warning(
                            "Page %d region %d skipped (%s: %s)",
                            page_number, region_idx, type(region_exc).__name__, region_exc,
                        )
                        continue

                # ---- OCR text → paragraph blocks ----
                remaining_text = ocr_text
                for txt in table_texts:
                    remaining_text = remaining_text.replace(txt, "")
                for txt in image_descriptions:
                    remaining_text = remaining_text.replace(txt, "")

                paragraphs: List[str] = []
                current_para: List[str] = []
                for line in remaining_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    current_para.append(line)
                    if line.endswith(".") or len(current_para) >= 5:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                if current_para:
                    paragraphs.append(" ".join(current_para))

                for para in paragraphs:
                    block_type = "heading" if (para.isupper() and len(para) < 100) else "text"
                    page_blocks.append(NormalizedBlock(
                        block_id=str(uuid.uuid4()),
                        document_id=document_id,
                        type=block_type,
                        text=para,
                        source_ref=SourceRef(filename=filename, page=page_number),
                        confidence=0.8,
                    ))

                # ── Page succeeded: accumulate + stream ──
                blocks.extend(page_blocks)
                logger.info("Page %d: done, %d blocks (total so far: %d)",
                            page_number, len(page_blocks), len(blocks))
                if out_queue is not None:
                    out_queue.put(("page", page_number, as_dicts(page_blocks)))

            except Exception as page_exc:
                # Entire page failed (OCR crash, render error, etc.) — skip it
                import traceback as _tb
                reason = f"{type(page_exc).__name__}: {page_exc}"
                logger.error("Page %d skipped — %s\n%s", page_number, reason, _tb.format_exc())
                if out_queue is not None:
                    out_queue.put(("page_error", page_number, reason))
                # continue to next page

    finally:
        doc.close()
    return blocks