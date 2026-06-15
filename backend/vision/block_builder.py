# backend/vision/block_builder.py

from uuid import uuid4
import json


def parse_caption(caption_json_str):
    """Parse the vision model's JSON reply into a clean, searchable caption.

    Returns (searchable_text, entities, vision_type, confidence, enrichment_failed).
    On a non-JSON reply, degrades to the raw text as the description so a caption
    is never lost. Used by BOTH the PDF (page-profile) and Excel/PPT (pending-block)
    paths so captions are consistent and never raw JSON blobs.
    """
    enrichment_failed = False
    entities = []
    vision_type = "other"
    confidence = 0.0
    try:
        data = json.loads(_strip_fences(caption_json_str))
        description = (data.get("description") or "").strip()
        entities = data.get("entities") or []
        vision_type = data.get("type", "other")
        confidence = float(data.get("confidence", 0.95))
        if not isinstance(entities, list):
            entities = []
    except Exception:
        enrichment_failed = True
        description = str(caption_json_str).strip()

    searchable_text = description
    if entities:
        searchable_text += "\n\nEntities: " + ", ".join(str(e) for e in entities)
    return searchable_text, entities, vision_type, confidence, enrichment_failed


def _strip_fences(text: str) -> str:
    """Tolerate ```json ... ``` fences some models wrap JSON in."""
    t = (text or "").strip()
    if t.startswith("```"):
        inner = t.split("```")
        t = inner[1] if len(inner) >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


def build_image_caption_block(state, page_number, bbox, caption_json_str):
    searchable_text, entities, vision_type, confidence, enrichment_failed = parse_caption(
        caption_json_str
    )

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