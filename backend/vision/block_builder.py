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


def _balanced_objects(t):
    """Yield every top-level {...} substring in `t`, brace-balanced and string-aware
    (braces inside JSON string values don't open/close objects). Used to recover the
    real JSON from a reasoning-heavy reply that may contain several objects."""
    objs = []
    depth = start = 0
    in_str = esc = False
    start = -1
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(t[start:i + 1])
                start = -1
    return objs


def _extract_json(text):
    """Pull the JSON object out of a model reply that may include reasoning.

    "Thinking" models (e.g. gemma-4) emit prose/analysis before the JSON — and
    often MORE than one object (a fenced ```json example in their "format as JSON"
    step, then the final answer). The old first-{ .. last-} span swallowed both
    objects plus the fence between them and failed to parse, so we stored the raw
    reasoning as the caption. Now: strip fences + direct parse for clean replies,
    else scan for balanced {...} objects and return the LAST that parses (the
    model's final answer)."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    for obj in reversed(_balanced_objects(text or "")):
        try:
            return json.loads(obj)
        except Exception:
            continue
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