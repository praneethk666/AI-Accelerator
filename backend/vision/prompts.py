# backend/vision/prompts.py
"""Vision prompts.

A vision prompt = a route-specific FOCUS instruction (what to look for) + a shared
JSON OUTPUT CONTRACT (so block_builder can parse description/entities/type). The
focus comes from config["vision"]["prompt"][<route>] when present (e.g. the
CAD/circuit instructions in global.yaml), else a generic default.
"""

# Shared output format — keeps captions structured (description + searchable
# entities) regardless of which focus instruction is used.
JSON_CONTRACT = """Return ONLY valid JSON, no markdown:
{
  "type": "diagram | flowchart | engineering_drawing | circuit | chart | photo | logo | other",
  "description": "concise, information-rich description (max 3 sentences)",
  "entities": ["key terms", "labels", "component names", "brand names", "measurements"],
  "confidence": 0.0-1.0
}
Rules: confidence in [0,1]; description captures the main visual content;
entities are highly searchable keywords; JSON only."""

DEFAULT_FOCUS = """You are an expert document-understanding assistant. Analyze the
image region and capture all visible content — diagrams, flowcharts, engineering
drawings, circuits, charts, technical illustrations, logos, photos."""

# Back-compat: the generic full prompt (focus + contract).
VISION_PROMPT = f"{DEFAULT_FOCUS}\n\n{JSON_CONTRACT}"


def build_vision_prompt(state: dict, config: dict) -> str:
    """Compose the focus instruction for this document's route + the JSON contract.

    Route prompts live in config["vision"]["prompt"] (keys match route names, e.g.
    cad_route / circuit_route, plus "default"). Falls back to DEFAULT_FOCUS.
    """
    prompts = (config.get("vision", {}) or {}).get("prompt", {}) or {}
    route = (state.get("route") or "").strip()
    focus = prompts.get(route) or prompts.get("default") or DEFAULT_FOCUS
    return f"{focus.strip()}\n\n{JSON_CONTRACT}"
