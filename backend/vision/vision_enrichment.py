
"""Vision enrichment tool – describes images using Gemini (or other vision API).
 
Reads:

  - state["page_profiles"] (PDF only) – bounding boxes + significance flags

  - state["blocks"] – looks for pending_vision blocks with raw_image_path (Excel/PPT)

  - state["file_path"] – to crop PDF pages
 
Writes:

  - New image_caption blocks with descriptions (PDF images)

  - Updates existing pending blocks in place (Excel/PPT)

  - Stores rendered images under uploads/images/<doc_id>/

  - Sets metadata.image_path for frontend display
 
Uses concurrency, duplicate detection, timeouts, and error handling.

"""
 
import os

import hashlib

import uuid

from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed
 
import fitz  # PyMuPDF
 
from backend.core.vision_client import describe_image

from backend.core.tool import Tool, PipelineState

from .pdf_cropper import PDFCropper

from .block_builder import build_image_caption_block, parse_caption

from .timeout import run_with_timeout, TimeoutException

from .prompts import VISION_PROMPT, build_vision_prompt   # route-aware prompt builder
 
from dotenv import load_dotenv

load_dotenv()
class VisionEnrichmentTool(Tool):

    name = "vision_enrichment"
 
    def run(self, state: PipelineState, config: dict) -> PipelineState:

        # Resolve the route-specific prompt ONCE and thread it through vision_cfg
        # so every per-image task uses the right focus (e.g. CAD/circuit) instead
        # of the generic prompt. Copy so we don't mutate the shared config dict.
        vision_cfg = dict(config.get("vision", {}))
        vision_cfg["_resolved_prompt"] = build_vision_prompt(state, config)

        timeout_s = vision_cfg.get("timeout_s", 60)

        dpi = vision_cfg.get("dpi", 150)

        debug = vision_cfg.get("debug", False)

        max_concurrency = vision_cfg.get("max_concurrency", 4)
 
        blocks = state.setdefault("blocks", [])

        errors = state.setdefault("errors", [])
 
        hash_to_first_block = {}

        hash_occurrence_count = defaultdict(int)
 
        debug_dir = os.path.join("output", "debug")

        if debug:

            os.makedirs(debug_dir, exist_ok=True)
 
        # ------------------------------------------------------------------

        # 1. Process PDF images using page_profiles

        # ------------------------------------------------------------------

        if "page_profiles" in state and state.get("file_path"):

            page_profiles = state["page_profiles"]

            doc = fitz.open(state["file_path"])

            page_dims = {p+1: (doc[p].rect.width, doc[p].rect.height) for p in range(len(doc))}

            doc.close()
 
            cropper = PDFCropper()

            image_tasks = []
 
            for profile in page_profiles:

                page_number = profile["page_number"]

                kind = profile.get("kind", "digital")

                images = profile.get("images", [])
 
                # Decision logic from the plan:

                # - If scanned page has images → send the whole page as one image.

                # - If digital page has vector graphics → send the whole page.

                # - Otherwise send each significant image region individually.

                if kind == "scanned" and images:

                    print(f"\n📄 Page {page_number}: scanned + images → FULL PAGE")

                    w, h = page_dims[page_number]

                    bbox = [0, 0, w, h]

                    image_tasks.append((cropper, state, page_number, bbox, dpi,

                                        timeout_s, vision_cfg, hash_to_first_block,

                                        hash_occurrence_count, debug_dir, blocks, errors, debug))

                elif kind == "scanned" and not images:

                    print(f"\n⏭️ Page {page_number}: scanned, no images → skip")

                else:

                    has_vector = profile.get("has_vector_graphics", False)

                    if has_vector:

                        print(f"\n📄 Page {page_number}: digital + vector → FULL PAGE")

                        w, h = page_dims[page_number]

                        bbox = [0, 0, w, h]

                        image_tasks.append((cropper, state, page_number, bbox, dpi,

                                            timeout_s, vision_cfg, hash_to_first_block,

                                            hash_occurrence_count, debug_dir, blocks, errors, debug))

                    else:

                        for img in images:

                            if not img.get("significant", False):

                                continue

                            bbox = img["bbox"]

                            image_tasks.append((cropper, state, page_number, bbox, dpi,

                                                timeout_s, vision_cfg, hash_to_first_block,

                                                hash_occurrence_count, debug_dir, blocks, errors, debug))
 
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:

                futures = []

                for task in image_tasks:

                    futures.append(executor.submit(self._process_pdf_image_task, *task))

                for future in as_completed(futures):

                    try:

                        future.result()

                    except Exception as e:

                        errors.append(f"Concurrent PDF vision task failed: {str(e)}")
 
        # ------------------------------------------------------------------

        # 2. Process pending vision blocks (Excel, PPT, etc.)

        # ------------------------------------------------------------------

        pending_blocks = []

        for block in blocks:

            metadata = block.get("metadata", {})

            if metadata.get("pending_vision") and metadata.get("raw_image_path"):

                pending_blocks.append(block)
 
        if pending_blocks:

            print(f"\n🔍 Processing {len(pending_blocks)} pending vision blocks...")

            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:

                futures = []

                for block in pending_blocks:

                    futures.append(executor.submit(self._process_pending_block_task,

                                                   block, state, timeout_s, vision_cfg,

                                                   hash_to_first_block, hash_occurrence_count,

                                                   debug_dir, errors, debug))

                for future in as_completed(futures):

                    try:

                        future.result()

                    except Exception as e:

                        errors.append(f"Concurrent pending vision task failed: {str(e)}")
 
        # ------------------------------------------------------------------

        # Duplicate summary (optional, for transparency)

        # ------------------------------------------------------------------

        total_unique = len(hash_to_first_block)

        total_images = sum(hash_occurrence_count.values())

        duplicates_saved = total_images - total_unique

        if duplicates_saved > 0:

            print("\n" + "=" * 80)

            print("DUPLICATE IMAGE SUMMARY")

            print("=" * 80)

            for h, count in hash_occurrence_count.items():

                if count > 1:

                    block_info = hash_to_first_block.get(h, {})

                    first_src = block_info.get("source_ref", {})

                    first_id = first_src.get("page") or first_src.get("slide") or first_src.get("sheet") or "unknown"

                    print(f"  Image first on {first_src.get('filename', '?')} at {first_id}: appeared {count} times total")

            print(f"\nTotal unique images processed (Gemini calls): {total_unique}")

            print(f"Total image occurrences: {total_images}")

            print(f"Duplicates reused without Gemini: {duplicates_saved}")

        else:

            print("\n✅ No duplicate images found across all sources.")
 
        return state
 
    # ------------------------------------------------------------------

    # Task: process one PDF image (crops from PDF, calls vision if new)

    # ------------------------------------------------------------------

    def _process_pdf_image_task(self, cropper, state, page_number, bbox, dpi,

                                timeout_s, vision_cfg, hash_to_first_block,

                                hash_occurrence_count, debug_dir, blocks, errors, debug):

        try:

            image_bytes = cropper.crop_region(

                pdf_path=state["file_path"],

                page_number=page_number,

                bbox=bbox,

                dpi=dpi,

            )

            print(f"🔪 Cropped page {page_number}, bbox {bbox}, size {len(image_bytes)} bytes")
 
            # Skip tiny / blank crops (likely errors or empty regions)

            if len(image_bytes) < 10_000:

                print(f"⏭️ Skipping tiny/blank crop – placeholder")

                placeholder_block = self._create_placeholder_block(

                    state, page_number, bbox, "Image too small or blank."

                )

                if placeholder_block:

                    blocks.append(placeholder_block)

                return
 
            img_hash = hashlib.md5(image_bytes).hexdigest()

            self._enrich_or_reuse_pdf(

                img_hash, image_bytes, state, page_number, bbox,

                timeout_s, vision_cfg, debug_dir,

                hash_to_first_block, hash_occurrence_count, blocks, errors, debug

            )

        except TimeoutException:

            print(f"⏰ Timeout for page {page_number}")

            errors.append(f"Vision timeout page {page_number}, bbox {bbox}")

            timeout_block = self._create_placeholder_block(

                state, page_number, bbox, "Description timed out."

            )

            if timeout_block:

                blocks.append(timeout_block)

        except Exception as e:

            errors.append(f"Vision error page {page_number}: {str(e)}")

            import traceback

            traceback.print_exc()
 
    # ------------------------------------------------------------------

    # Task: process a pending block (Excel/PPT) – updates block in place

    # ------------------------------------------------------------------

    def _process_pending_block_task(self, block, state, timeout_s, vision_cfg,

                                    hash_to_first_block, hash_occurrence_count,

                                    debug_dir, errors, debug):

        raw_path = block.get("metadata", {}).get("raw_image_path")

        if not raw_path or not os.path.exists(raw_path):

            print(f"⚠️ Missing raw_image_path: {raw_path}")

            return
 
        try:

            with open(raw_path, "rb") as f:

                image_bytes = f.read()
 
            print(f"🖼️ Processing pending image: {raw_path}")
 
            if len(image_bytes) < 10_000:

                print(f"⏭️ Skipping tiny/blank pending image – placeholder")

                block["text"] = "Image too small or blank."

                block["metadata"]["pending_vision"] = False

                block["metadata"]["enrichment_failed"] = True

                return
 
            img_hash = hashlib.md5(image_bytes).hexdigest()

            self._enrich_or_reuse_pending(

                img_hash, image_bytes, block, raw_path,

                timeout_s, vision_cfg, debug_dir,

                hash_to_first_block, hash_occurrence_count, errors, debug

            )

        except Exception as e:

            print(f"❌ Failed to enrich {raw_path}: {e}")

            block["metadata"]["enrichment_failed"] = True

            block["metadata"]["error"] = str(e)

            block["text"] = f"Failed: {str(e)}"

            block["metadata"]["pending_vision"] = False
 
    # ------------------------------------------------------------------

    # Enrich a PDF image (first occurrence) or reuse duplicate

    # ------------------------------------------------------------------

    def _enrich_or_reuse_pdf(self, img_hash, image_bytes, state, page_number, bbox,

                             timeout_s, vision_cfg, debug_dir,

                             hash_to_first_block, hash_occurrence_count, blocks, errors, debug):

        hash_occurrence_count[img_hash] += 1

        occurrence = hash_occurrence_count[img_hash]
 
        # Duplicate – copy from the first occurrence

        if img_hash in hash_to_first_block:

            first = hash_to_first_block[img_hash]

            dup = {

                "block_id": str(uuid.uuid4()),

                "document_id": state["document_id"],

                "type": first["type"],

                "text": first["text"],

                "table_data": first["table_data"],

                "source_ref": {

                    "filename": state["file_path"],

                    "page": page_number,

                    "sheet": None,

                    "slide": None,

                    "bbox": bbox,

                },

                "confidence": first["confidence"],

                "language": first["language"],

                "metadata": first["metadata"].copy(),

            }

            blocks.append(dup)

            print(f"📋 Duplicate PDF image (occ #{occurrence}) → reused from page {first['source_ref']['page']}")

            return
 
        # First time – call Gemini

        print(f"🆕 First PDF image – calling describe_image...")

        if debug:

            debug_path = os.path.join(debug_dir, f"debug_p{page_number}_x{int(bbox[0])}_{int(bbox[1])}.png")

            with open(debug_path, "wb") as f:

                f.write(image_bytes)

            print(f"💾 Debug image: {debug_path}")
 
        description = run_with_timeout(describe_image, timeout_s, image_bytes, vision_cfg.get("_resolved_prompt") or VISION_PROMPT, vision_cfg)

        print(f"📄 Description (first 200 chars): {description[:200]}")
 
        # Build a NormalizedBlock (type="image_caption") with the description

        block = build_image_caption_block(state, page_number, bbox, description)

        if not block:

            hash_to_first_block[img_hash] = {"error": "block building failed"}

            errors.append(f"Block building failed for page {page_number}")

            return
 
        # Save the rendered image to permanent storage (for frontend)

        doc_id = state["document_id"]

        img_dir = os.path.join("uploads", "images", doc_id)

        os.makedirs(img_dir, exist_ok=True)

        image_filename = f"{block['block_id']}.png"

        permanent_path = os.path.join(img_dir, image_filename)

        with open(permanent_path, "wb") as f:

            f.write(image_bytes)
 
        block["metadata"]["image_path"] = f"/images/{doc_id}/{image_filename}"

        block["metadata"]["enrichment_failed"] = False
 
        blocks.append(block)

        hash_to_first_block[img_hash] = block
 
    # ------------------------------------------------------------------

    # Enrich a pending block (Excel/PPT) – updates block in place

    # ------------------------------------------------------------------

    def _enrich_or_reuse_pending(self, img_hash, image_bytes, block, raw_path,

                                 timeout_s, vision_cfg, debug_dir,

                                 hash_to_first_block, hash_occurrence_count, errors, debug):

        hash_occurrence_count[img_hash] += 1

        occurrence = hash_occurrence_count[img_hash]
 
        if img_hash in hash_to_first_block:

            first = hash_to_first_block[img_hash]

            block["text"] = first["text"]

            block["type"] = first["type"]

            block["table_data"] = first["table_data"]

            block["confidence"] = first["confidence"]

            block["language"] = first["language"]

            block["metadata"]["pending_vision"] = False

            block["metadata"]["enrichment_failed"] = False

            block["metadata"]["image_path"] = first["metadata"].get("image_path")

            print(f"📋 Duplicate pending image (occ #{occurrence}) → reused from block {first['block_id']}")

            return
 
        # First time – call Gemini

        print(f"🆕 First pending image – calling describe_image...")

        if debug:

            debug_path = os.path.join(debug_dir, f"debug_pending_{os.path.basename(raw_path)}.png")

            with open(debug_path, "wb") as f:

                f.write(image_bytes)

            print(f"💾 Debug image: {debug_path}")
 
        description = run_with_timeout(describe_image, timeout_s, image_bytes, vision_cfg.get("_resolved_prompt") or VISION_PROMPT, vision_cfg)

        print(f"📄 Description (first 200 chars): {description[:200]}")
 
        # Parse the JSON reply into a clean caption (description + entities) —
        # same as the PDF path; never store the raw JSON blob as the chunk text.
        searchable_text, entities, vision_type, conf, failed = parse_caption(description)

        block["text"] = searchable_text
        block["confidence"] = conf or 0.95

        block["metadata"]["pending_vision"] = False
        block["metadata"]["enrichment_failed"] = failed
        block["metadata"]["entities"] = entities
        block["metadata"]["vision_type"] = vision_type
 
        # Make the image web‑accessible (assume raw_path is already under uploads/images/)

        if raw_path.startswith("uploads/images/"):

            web_path = raw_path.replace("uploads/images/", "/images/", 1)

            block["metadata"]["image_path"] = web_path

        else:

            block["metadata"]["image_path"] = raw_path
 
        hash_to_first_block[img_hash] = block
 
    # ------------------------------------------------------------------

    # Helper: create a minimal placeholder block (for tiny/timeout images)

    # ------------------------------------------------------------------

    def _create_placeholder_block(self, state, page_number, bbox, placeholder_text: str):

        return {

            "block_id": str(uuid.uuid4()),

            "document_id": state["document_id"],

            "type": "image_caption",

            "text": placeholder_text,

            "table_data": None,

            "source_ref": {

                "filename": state["file_path"],

                "page": page_number,

                "sheet": None,

                "slide": None,

                "bbox": bbox,

            },

            "confidence": 0.5,

            "language": "en",

            "metadata": {

                "pending_vision": False,

                "enrichment_failed": True,

                "is_vector": False,

            },

        }
 