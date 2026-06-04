# backend/vision/vision_enrichment.py

import os
import hashlib
from uuid import uuid4
from collections import defaultdict
import fitz  # PyMuPDF
from .pdf_cropper import PDFCropper
from .vision_client import VisionClient
from .block_builder import build_image_caption_block
from .timeout import run_with_timeout, TimeoutException

class VisionEnrichmentTool:
    name = "vision_enrichment"

    def __init__(self, model_name=None):
        self.cropper = PDFCropper()
        if model_name is None:
            model_name = "gemma-4-26b-a4b-it"  # default model
        self.vision_client = VisionClient(model_name=model_name)

    def run(self, state, config):
        page_profiles = state.get("page_profiles", [])
        blocks = state.setdefault("blocks", [])
        errors = state.setdefault("errors", [])

        vision_cfg = config.get("vision", {})
        timeout_s = vision_cfg.get("timeout_s", 45)
        dpi = vision_cfg.get("dpi", 200)

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
            has_vector = profile.get("has_vector_graphics", False)

            # --- If vector graphics are present, ignore raster images and process whole page ---
            if has_vector:
                print(f"\n📄 Page {page_number}: has_vector_graphics=True → processing FULL PAGE (ignoring any raster images)")
                page_width, page_height = page_dims[page_number]
                bbox = [0, 0, page_width, page_height]  # whole page
                self._process_image(
                    state, page_number, bbox, dpi, timeout_s, vision_cfg,
                    hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors,
                    is_vector_fallback=True
                )
                continue  # skip any possible raster images (just in case)

            # --- No vector graphics: process raster images normally ---
            images = profile.get("images", [])
            for img in images:
                if not img.get("significant", False):
                    continue
                bbox = img["bbox"]
                self._process_image(
                    state, page_number, bbox, dpi, timeout_s, vision_cfg,
                    hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors
                )

        # Duplicate summary (unchanged)
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
                       hash_to_first_block, hash_occurrence_count, debug_dir, blocks, errors,
                       is_vector_fallback=False):
        """Process a single image region (raster or full‑page vector)."""
        try:
            image_bytes = self.cropper.crop_region(
                pdf_path=state["file_path"],
                page_number=page_number,
                bbox=bbox,
                dpi=dpi,
            )
            print(f"🔪 Cropped page {page_number}, bbox {bbox}, size {len(image_bytes)} bytes")

            img_hash = hashlib.md5(image_bytes).hexdigest()
            hash_occurrence_count[img_hash] += 1
            occurrence_num = hash_occurrence_count[img_hash]

            if img_hash in hash_to_first_block:
                first_block = hash_to_first_block[img_hash]
                duplicate_block = {
                    "block_id": str(uuid4()),
                    "document_id": state["document_id"],
                    "type": first_block["type"],
                    "text": first_block["text"],
                    "table_data": first_block["table_data"],
                    "source_ref": {
                        "filename": state["file_path"],
                        "page": page_number,
                        "sheet": None,
                        "slide": None,
                        "bbox": bbox,
                    },
                    "confidence": first_block["confidence"],
                    "language": first_block["language"],
                    "metadata": first_block["metadata"].copy(),
                }
                blocks.append(duplicate_block)
                print(f"📋 Duplicate image (occurrence #{occurrence_num}) – reused from page {first_block['source_ref']['page']}")
                return

            # First time – call Gemini
            print(f"🆕 First occurrence – calling Gemini...")
            debug_path = os.path.join(debug_dir, f"debug_p{page_number}_x{int(bbox[0])}_{int(bbox[1])}.png")
            with open(debug_path, "wb") as f:
                f.write(image_bytes)
            print(f"💾 Debug image: {debug_path}")

            caption_json = run_with_timeout(
                self.vision_client.describe,
                timeout_s,
                image_bytes,
                config=vision_cfg,
            )
            print(f"📄 Caption JSON (first 200 chars): {caption_json[:200]}")

            new_block = build_image_caption_block(
                state=state,
                page_number=page_number,
                bbox=bbox,
                caption_json_str=caption_json,
            )
            if new_block is not None:
                blocks.append(new_block)
                hash_to_first_block[img_hash] = new_block
            else:
                hash_to_first_block[img_hash] = {"error": "block building failed"}
                errors.append(f"Block building failed for page {page_number}")

        except TimeoutException:
            errors.append(f"Vision timeout page {page_number}, bbox {bbox}")
        except Exception as e:
            errors.append(f"Vision error page {page_number}: {str(e)}")
            import traceback
            traceback.print_exc()