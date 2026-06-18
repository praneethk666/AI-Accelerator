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
    data = _extract_json(caption_json_str)
    if data is not None:
        description = (data.get("description") or "").strip()
        entities = data.get("entities") or []
        vision_type = data.get("type", "other")
        try:
            confidence = float(data.get("confidence", 0.95))
        except (TypeError, ValueError):
            confidence = 0.95
        if not isinstance(entities, list):
            entities = []
    else:
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


def _extract_json(text):
    """Pull the JSON object out of a model reply that may include reasoning.

    "Thinking" models (e.g. gemma-4) emit prose/analysis before the JSON, so a
    plain json.loads on the whole reply fails and we'd otherwise store the raw
    reasoning as the caption. Strip fences, try direct parse, then fall back to the
    outermost {...} span."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            return None
    return None


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