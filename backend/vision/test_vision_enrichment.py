# backend/vision/test_vision_enrichment.py

import json
import os
import sys
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.vision.vision_enrichment import VisionEnrichmentTool

# We are not using VisionEnrichmentTool anymore – we generate blocks directly.
# from backend.vision.vision_enrichment import VisionEnrichmentTool


def generate_blocks_from_json(page_profiles, document_id="doc123", filename="unknown.pdf"):
    """
    Generate image_caption blocks directly from JSON metadata.
    Includes duplicate detection for raster images (same bbox across pages).
    Vector pages are never deduplicated (assumed unique).
    """
    from uuid import uuid4

    blocks = []
    seen_raster_keys = set()   # store (bbox_tuple) for raster images

    for profile in page_profiles:
        page_number = profile["page_number"]
        has_vector = profile.get("has_vector_graphics", False)
        images = profile.get("images", [])

        # --- Process raster images ---
        for img in images:
            if not img.get("significant", True):
                continue
            bbox = img["bbox"]
            # Use tuple of bbox as key (exact coordinates)
            bbox_key = tuple(bbox)
            if bbox_key in seen_raster_keys:
                print(f"⏭️ Skipping duplicate raster image on page {page_number}, bbox {bbox}")
                continue
            seen_raster_keys.add(bbox_key)

            block = {
                "block_id": str(uuid4()),
                "document_id": document_id,
                "type": "image_caption",
                "text": f"[Raster image on page {page_number} at bbox {bbox}]",
                "table_data": None,
                "source_ref": {
                    "filename": filename,
                    "page": page_number,
                    "sheet": None,
                    "slide": None,
                    "bbox": bbox,
                },
                "confidence": 0.0,
                "language": "en",
                "metadata": {
                    "vision_type": "other",
                    "entities": [],
                    "enrichment_failed": True,
                },
            }
            blocks.append(block)

        # --- Process vector‑only pages (no deduplication) ---
        if has_vector and len(images) == 0:
            block = {
                "block_id": str(uuid4()),
                "document_id": document_id,
                "type": "image_caption",
                "text": f"[Vector graphics page {page_number} – full page]",
                "table_data": None,
                "source_ref": {
                    "filename": filename,
                    "page": page_number,
                    "sheet": None,
                    "slide": None,
                    "bbox": [0, 0, 0, 0],
                },
                "confidence": 0.0,
                "language": "en",
                "metadata": {
                    "vision_type": "diagram",
                    "entities": [],
                    "enrichment_failed": True,
                },
            }
            blocks.append(block)

    return blocks

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_folder = os.path.join(project_root, "output")
    profiles_subfolder = os.path.join(output_folder, "page_profiles")   
    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------------------------------------
    # 1. DIRECTLY LOAD YOUR EXISTING JSON – NO GENERATION
    # ------------------------------------------------------------
    json_filename = "Digital_40pages_page_profiles.json"   # your existing JSON file
    json_path = os.path.join(profiles_subfolder, json_filename)   

    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        print("Please place your existing page_profiles.json in the 'output' folder.")
        return

    print(f"📂 Loading page profiles from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        page_profiles = json.load(f)

    # ------------------------------------------------------------
    # 2. PDF is still required for cropping (must match the JSON)
    # ------------------------------------------------------------
    pdf_filename = "Digital_40pages.pdf"   # same PDF that the JSON refers to
    pdf_path = os.path.join(project_root, "test-data", pdf_filename)

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("The PDF is needed to crop images for Gemini.")
        return

    # ------------------------------------------------------------
    # 3. Run vision enrichment (calls Gemini)
    # ------------------------------------------------------------
    state = {
        "document_id": "doc123",
        "file_path": pdf_path,
        "page_profiles": page_profiles,
    }

    config = {
        "vision": {
            "timeout_s": 45,
            "dpi": 150,
        }
    }

    tool = VisionEnrichmentTool(model_name="gemma-4-26b-a4b-it")  # or "gemini-3.5-flash"
    result = tool.run(state, config)

    # ------------------------------------------------------------
    # 4. Save blocks
    # ------------------------------------------------------------
    blocks = result.get("blocks", [])
    blocks_path = os.path.join(output_folder, "Dgital_40pages_vision_blocks1.json")
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)

    print(f"\n✅ Done. {len(blocks)} blocks saved to {blocks_path}")
    if result.get("errors"):
        print("Errors:", result["errors"])


if __name__ == "__main__":
    main()
