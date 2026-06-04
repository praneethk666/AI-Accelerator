"""Vision-based document_type inference.

The vision model should classify document type using document images.

Contract:
- Input: a combined image containing the cover+content (typically pages 1-3 stitched vertically)
- Output: {"document_type": <label>, "confidence": float, "reasoning": str}

This module is implemented to be resilient:
- If an actual vision API client isn't configured, it returns a conservative fallback.

IMPORTANT: Route is derived elsewhere from config.type_to_route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import base64
import json
import os
import re

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


@dataclass
class VisionResult:
    document_type: str
    confidence: float
    reasoning: str


def _b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def run_vision(
    *,
    combined_image_bytes: bytes,
    filename: str = "",
    toc_text: str = "",
    midpage_text: str = "",
) -> Dict[str, Any]:
    """Run vision inference.

    Returns dict with keys:
      - document_type
      - confidence
      - reasoning

    If no vision provider is configured, returns a fallback.
    """

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key:
            return {
                "document_type": "report",
                "confidence": 0.1,
                "reasoning": "GEMINI_API_KEY not configured; returning fallback document_type.",
            }

        if not HAS_GENAI:
            return {
                "document_type": "report",
                "confidence": 0.1,
                "reasoning": "google.generativeai not installed; returning fallback.",
            }

        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Encode image for Gemini
        image_b64 = _b64_encode_bytes(combined_image_bytes)
        image_data = {
            "mime_type": "image/png",
            "data": image_b64,
        }

        # Build the prompt with improved CAD detection
        extra_context = ""
        cad_hint = ""
        
        if filename:
            extra_context += f"\nFilename: {filename}"
            # Check if filename looks like CAD (part number pattern, mechanical keywords)
            lower_fn = filename.lower()
            cad_patterns = [
                r'^[A-Z]{1,3}\d{2}[A-Z]{3}\d{6}',  # MS03AAA981AA
                r'dwg[-_]?\d{4}',  # DWG-0001
                r'\d{2}y[-_]?[a-z]{3}\d{7}',  # 99Y_MKR2002100AB
                'motor', 'engine', 'assembly', 'drawing', 'schematic'
            ]
            if any(re.search(p, lower_fn) if isinstance(p, str) and p.startswith('^') or p.startswith(r'\\') else p in lower_fn for p in cad_patterns):
                cad_hint = "\n⚠️ FILENAME HINT: This looks like an engineering drawing or CAD document based on the filename.\nIf this is a technical/mechanical document with engineering indicators, classify as cad_drawing or circuit_diagram."
        
        if toc_text:
            extra_context += f"\nTable of Contents excerpt:\n{toc_text[:500]}"

        prompt = f"""Analyze this document image and classify its document_type.

IMPORTANT CAD DETECTION:
- CAD/Engineering drawings have: title blocks, revision blocks, part numbers, scales, dimensions (mm/inch)
- CAD drawings show: mechanical views, sections, details, BOM tables, technical notations
- If you see ANY of these CAD indicators, classify as cad_drawing or circuit_diagram{cad_hint}

DOCUMENT TYPES:
- circuit_diagram: electrical schematics, circuit boards, wiring diagrams
- cad_drawing: mechanical CAD drawings, 3D design blueprints, engineering drawings (IF NOT circuit/schematic)
- schematic: technical diagrams, flow diagrams
- invoice: bills, receipts, payment documents
- financial_statement: balance sheets, P&L, financial reports
- purchase_order: POs, shipping documents
- contract: legal agreements, terms & conditions
- policy: company policies, procedures
- research_paper: academic papers, technical papers
- report: general reports, analyses, white papers
- manual: user guides, instruction manuals
- presentation: slides, PowerPoint decks

Return ONLY a JSON object with these exact keys:
- "document_type": exactly one of the types above
- "confidence": a float between 0 and 1
- "reasoning": a short explanation (mention CAD indicators if present)

Document context:{extra_context}

Respond with ONLY the JSON object, no other text, no markdown."""

        # Call Gemini Vision
        response = model.generate_content([image_data, prompt])
        response_text = response.text.strip()

        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            doc_type = result.get("document_type", "report")
            confidence = float(result.get("confidence", 0.0) or 0.0)
            reasoning = result.get("reasoning", "Vision classification completed.")
            
            return {
                "document_type": doc_type,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        else:
            # Fallback if JSON extraction fails
            return {
                "document_type": "report",
                "confidence": 0.3,
                "reasoning": f"Vision response parsing failed: {response_text[:100]}",
            }

    except json.JSONDecodeError as e:
        return {
            "document_type": "report",
            "confidence": 0.0,
            "reasoning": f"Vision JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "document_type": "report",
            "confidence": 0.0,
            "reasoning": f"Vision inference failed: {type(e).__name__}: {str(e)[:100]}",
        }

