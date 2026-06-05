# backend/vision/vision_enrichment.py

import os
import hashlib
import re
import json
from uuid import uuid4
from collections import defaultdict
import fitz  # PyMuPDF
from .pdf_cropper import PDFCropper
from .block_builder import build_image_caption_block
from .timeout import run_with_timeout, TimeoutException
from backend.core.vision_client import describe_image
from .prompts import VISION_PROMPT

class VisionEnrichmentTool:
    name = "vision_enrichment"

    def __init__(self, config: dict):
        self.cropper = PDFCropper()
        self.config = config

    def run(self, state, config_override=None):
        # Merge configs (allow override from call)
        final_config = self.config.copy()
        if config_override:
            final_config.update(config_override)

        page_profiles = state.get("page_profiles", [])
        blocks = state.setdefault("blocks", [])
        errors = state.setdefault("errors", [])

        vision_cfg = final_config.get("vision", {})
        timeout_s = vision_cfg.get("timeout_s", 60)
        dpi = vision_cfg.get("dpi", 150)

        hash_to_first_block = {}
        hash_occurrence_count = defaultdict(int)

        debug_dir = os.path.join(os.path.dirname(state["file_path"]), "..", "output")
        os.makedirs(debug_dir, exist_ok=True)

        # Pre‑fetch page dimensions for full‑page crops
        doc = fitz.open(state["file_path"])
        page_dims = {p+1: (doc[p].rect.width, doc[p].rect.height) for p in range(len(doc))}
        doc.close()

        for profile in page_profiles:
            page_number = profile["page_number"]
            kind = profile.get("kind", "digital")
            images = profile.get("images", [])

            # --- Scanned pages with images → full page ---
            if kind == "scanned" and images:
                print(f"\n📄 Page {page_number}: kind=scanned with image boundaries → FULL PAGE")
                w, h = page_dims[page_number]
                bbox = [0, 0, w, h]
                self._process_image(
                    state, page_number, bbox, dpi, timeout_s, vision_cfg,
                    hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors
                )
                continue
            elif kind == "scanned" and not images:
                print(f"\n⏭️ Page {page_number}: scanned but no images → skip")
                continue

            # --- Digital pages: vector detection ---
            has_vector = profile.get("has_vector_graphics", False)
            if has_vector:
                print(f"\n📄 Page {page_number}: digital + vector graphics → FULL PAGE")
                w, h = page_dims[page_number]
                bbox = [0, 0, w, h]
                self._process_image(
                    state, page_number, bbox, dpi, timeout_s, vision_cfg,
                    hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors
                )
                continue

            # --- Digital pages without vector → process raster images individually ---
            for img in images:
                if not img.get("significant", False):
                    continue
                bbox = img["bbox"]
                self._process_image(
                    state, page_number, bbox, dpi, timeout_s, vision_cfg,
                    hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors
                )

        # --- Duplicate summary (unchanged) ---
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
                    first_page = block_info.get("source_ref", {}).get("page", "unknown") if isinstance(block_info, dict) else "unknown"
                    print(f"  Image first on page {first_page}: appeared {count} times total")
            print(f"\nTotal unique images processed (Gemini calls): {total_unique}")
            print(f"Total image occurrences: {total_images}")
            print(f"Duplicates reused without Gemini: {duplicates_saved}")
        else:
            print("\n✅ No duplicate images found across pages.")

        return state

    def _process_image(self, state, page_number, bbox, dpi, timeout_s, vision_cfg,
                       hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors):
        try:
            image_bytes = self.cropper.crop_region(
                pdf_path=state["file_path"],
                page_number=page_number,
                bbox=bbox,
                dpi=dpi,
            )
            print(f"🔪 Cropped page {page_number}, bbox {bbox}, size {len(image_bytes)} bytes")

            # Skip tiny/blank images
            if len(image_bytes) < 10_000:
                print(f"⏭️ Skipping tiny/blank crop (size {len(image_bytes)} bytes) – placeholder")
                placeholder = json.dumps({"type": "blank", "description": "Image too small or blank.", "entities": [], "confidence": 0.0})
                block = build_image_caption_block(state, page_number, bbox, placeholder)
                if block:
                    blocks.append(block)
                return

            img_hash = hashlib.md5(image_bytes).hexdigest()
            hash_occurrence_count[img_hash] += 1
            occurrence = hash_occurrence_count[img_hash]

            if img_hash in hash_to_first_block:
                first = hash_to_first_block[img_hash]
                dup = {
                    "block_id": str(uuid4()),
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
                print(f"📋 Duplicate (occ #{occurrence}) → reused from page {first['source_ref']['page']}")
                return

            # First time → call shared vision client
            print(f"🆕 First occurrence – calling describe_image (shared client)...")
            debug_path = os.path.join(debug_dir, f"debug_p{page_number}_x{int(bbox[0])}_{int(bbox[1])}.png")
            with open(debug_path, "wb") as f:
                f.write(image_bytes)
            print(f"💾 Debug image: {debug_path}")

            # Call shared vision client (supports Google Gemma / Ollama)
            raw = run_with_timeout(
                describe_image,
                timeout_s,
                image_bytes,
                VISION_PROMPT,
                vision_cfg  # config dict contains provider, model, etc.
            )
            print(f"📄 Raw response (first 200 chars): {raw[:200]}")

            # Extract JSON (same as before)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    clean_json = json.dumps(parsed)
                except json.JSONDecodeError:
                    clean_json = raw
            else:
                clean_json = raw

            block = build_image_caption_block(state, page_number, bbox, clean_json)
            if block:
                blocks.append(block)
                hash_to_first_block[img_hash] = block
            else:
                hash_to_first_block[img_hash] = {"error": "block building failed"}
                errors.append(f"Block building failed for page {page_number}")

        except TimeoutException:
            print(f"⏰ Timeout for page {page_number}")
            errors.append(f"Vision timeout page {page_number}, bbox {bbox}")
            placeholder = json.dumps({"type": "timeout", "description": "API call timed out.", "entities": [], "confidence": 0.0})
            timeout_block = build_image_caption_block(state, page_number, bbox, placeholder)
            if timeout_block:
                blocks.append(timeout_block)
        except Exception as e:
            errors.append(f"Vision error page {page_number}: {str(e)}")
            import traceback
            traceback.print_exc()