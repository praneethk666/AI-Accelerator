# backend/vision/block_builder.py

from uuid import uuid4
import json

def build_image_caption_block(state, page_number, bbox, caption_json_str):
    enrichment_failed = False
    description = ""
    entities = []
    vision_type = "other"
    confidence = 0.0

    try:
        data = json.loads(caption_json_str)
        description = data.get("description", "").strip()
        entities = data.get("entities", [])
        vision_type = data.get("type", "other")
        confidence = float(data.get("confidence", 0.95))

        if not isinstance(entities, list):
            entities = []

    except Exception:
        enrichment_failed = True
        description = str(caption_json_str)
        entities = []
        vision_type = "other"
        confidence = 0.0

    searchable_text = description
    if entities:
        searchable_text += "\n\nEntities: " + ", ".join(entities)

    return {
        "block_id": str(uuid4()),
        "document_id": state["document_id"],
        "type": "image_caption",
        "text": searchable_text,
        "table_data": None,
        "source_ref": {
            "filename": state["file_path"],
            "page": page_number,
            "sheet": None,
            "slide": None,
            "bbox": bbox,
        },
        "confidence": confidence,
        "language": "en",
        "metadata": {
            "vision_type": vision_type,
            "entities": entities,
            "enrichment_failed": enrichment_failed,
        },
    }