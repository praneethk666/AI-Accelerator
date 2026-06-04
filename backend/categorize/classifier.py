"""Core categorization engine.

Vision-first design:
- Determine `document_type` using vision (covers + content via stitched pages).
- Map `document_type` -> `route` via config.type_to_route.
- Determine `industry` via filename/text/deployment defaults.

Primary outputs to state:
- state["route"]
- state["document_type"]
- state["industry"]
- state["categorization_confidence"]
- state["reasoning"]
- state["errors"] (on failure or low confidence)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

import fitz

import yaml

from .text_extractor import extract_text, extract_toc_text, analyze_cad_document, is_cad_document_by_filename
from .vision import run_vision


def _load_config() -> Dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_filename(s: str) -> str:
    return (s or "").lower()


def _match_keywords(filename_or_text: str, keywords: list[str]) -> bool:
    hay = (filename_or_text or "").lower()
    return any(k.lower() in hay for k in keywords)


def _score_industry_from_filename(filename: str, industry_keywords: Dict[str, list[str]]) -> Optional[str]:
    fn = _normalize_filename(filename)
    for industry, kws in industry_keywords.items():
        if _match_keywords(fn, kws):
            return industry
    return None


def _score_industry_from_text(text: str, industry_keywords: Dict[str, list[str]]) -> Optional[str]:
    for industry, kws in industry_keywords.items():
        if _match_keywords(text, kws):
            return industry
    return None


def _render_pdf_pages_to_stitched_image_bytes(file_path: str, pages: list[int], zoom: float = 2.0) -> bytes:
    """Render selected PDF pages and stitch vertically into one image (PNG bytes)."""
    doc = fitz.open(file_path)
    rendered = []
    try:
        for pno in pages:
            if pno < 0 or pno >= len(doc):
                continue
            page = doc[pno]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            rendered.append(pix)

        if not rendered:
            raise ValueError("No pages rendered")

        # If only one page, return it as-is
        if len(rendered) == 1:
            return rendered[0].tobytes("png")

        # Try to stitch using PIL if available
        try:
            from PIL import Image
            import io

            pil_images = []
            for pix in rendered:
                # Convert fitz pixmap to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                pil_images.append(img)

            # Calculate stitched dimensions (vertical stacking)
            total_height = sum(img.height for img in pil_images)
            max_width = max(img.width for img in pil_images)

            # Create new image with combined dimensions
            stitched = Image.new("RGB", (max_width, total_height), color="white")

            # Paste each image vertically
            y_offset = 0
            for img in pil_images:
                stitched.paste(img, (0, y_offset))
                y_offset += img.height

            # Convert back to PNG bytes
            output = io.BytesIO()
            stitched.save(output, format="PNG")
            return output.getvalue()

        except ImportError:
            # PIL not available, return first page as fallback
            return rendered[0].tobytes("png")

    finally:
        doc.close()


def categorize(file_path: str, state: Dict[str, Any], deployment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _load_config()
    deployment_cfg = deployment or config.get("deployment", {})

    # Initialize state error list
    state.setdefault("errors", [])

    # Defaults
    best = {
        "route": "text_default",
        "document_type": "report",
        "industry": deployment_cfg.get("default_industry", "automotive"),
        "confidence": 0.0,
        "reasoning": "",
    }

    try:
        filename = os.path.basename(file_path)
        lowered = filename.lower()

        # ---- CAD Document Detection (filename-first) ----
        suspected_cad = False
        if is_cad_document_by_filename(filename):
            suspected_cad = True
        
        # If PDF, try text-based CAD detection
        cad_analysis = None
        if lowered.endswith('.pdf') and not suspected_cad:
            try:
                cad_analysis = analyze_cad_document(file_path, max_pages=3)
                if cad_analysis.get('is_cad'):
                    # Confidence for CAD detection from text analysis: 0.85
                    cad_metadata = cad_analysis.get('metadata', {})
                    if cad_metadata.get('industry'):
                        best["industry"] = cad_metadata['industry']
                        best["document_type"] = "cad_drawing"
                        best["confidence"] = 0.85
                        best["route"] = config["type_to_route"].get("cad_drawing", "diagram_heavy")
                        reasoning_parts = [
                            f"CAD document detected from engineering metadata.",
                            f"Drawing#: {cad_metadata.get('drawing_number', 'N/A')}",
                            f"Industry: {cad_metadata.get('industry', 'engineering')}",
                        ]
                        best["reasoning"] = "\n".join(reasoning_parts).strip()
                        
                        # Write to state and return early
                        state["route"] = best["route"]
                        state["document_type"] = best["document_type"]
                        state["industry"] = best["industry"]
                        state["categorization_confidence"] = best["confidence"]
                        state["reasoning"] = best["reasoning"]
                        state.setdefault("errors", [])
                        
                        return {
                            "route": state["route"],
                            "document_type": state["document_type"],
                            "industry": state["industry"],
                            "confidence": state["categorization_confidence"],
                            "reasoning": state["reasoning"],
                        }
            except Exception as e:
                # If CAD analysis fails, continue with normal flow
                pass

        # ---- Document type: filename-first ----
        type_to_route: Dict[str, str] = config["type_to_route"]
        supported_types = list(type_to_route.keys())

        # simple filename match -> document_type
        matched_type: Optional[str] = None
        for t in supported_types:
            if t in lowered:
                matched_type = t
                break

        # also handle common aliases
        aliases = {
            "circuit": "circuit_diagram",
            "schematic": "schematic",
            "cad": "cad_drawing",
            "invoice": "invoice",
            "financial_statement": "financial_statement",
            "balance": "financial_statement",
            "purchase_order": "purchase_order",
            "contract": "contract",
            "policy": "policy",
            "paper": "research_paper",
            "ppt": "presentation",
        }
        for alias, t in aliases.items():
            if alias in lowered and t in supported_types:
                matched_type = t
                break

        reasoning_parts = []

        if matched_type:
            best["document_type"] = matched_type
            best["confidence"] = 0.9
            best["route"] = type_to_route[matched_type]
            reasoning_parts.append(f"Filename matched document_type='{matched_type}'.")
        else:
            # ---- Vision-first fallback: render pages 1-3 combined ----
            # Pages are 0-indexed in fitz
            pages_1_3 = [0, 1, 2]
            toc_text = extract_toc_text(file_path, max_pages=2)

            stitched_bytes = _render_pdf_pages_to_stitched_image_bytes(file_path, pages_1_3)

            vision_res = run_vision(
                combined_image_bytes=stitched_bytes,
                filename=filename,
                toc_text=toc_text,
                midpage_text="",
            )

            doc_type = vision_res.get("document_type")
            conf = float(vision_res.get("confidence", 0.0) or 0.0)
            reasoning = vision_res.get("reasoning", "")

            if not doc_type:
                doc_type = "report"

            # ---- CAD Override: If filename suggests CAD and vision didn't catch it ----
            if suspected_cad and doc_type != "cad_drawing" and doc_type != "circuit_diagram" and doc_type != "schematic":
                # Boost CAD classification if filename is clear CAD signal
                doc_type = "cad_drawing"
                conf = max(conf, 0.75)  # At least 0.75 confidence from filename hint
                reasoning = f"Filename pattern indicates CAD document (suspected_cad=True). Original vision: {reasoning}"

            best["document_type"] = doc_type
            best["confidence"] = conf
            best["route"] = type_to_route.get(doc_type, "text_default")
            reasoning_parts.append(f"Vision predicted document_type='{doc_type}' with confidence={conf:.2f}.")
            if reasoning:
                reasoning_parts.append(f"Vision reasoning: {reasoning}")

        # ---- Industry detection (3 signals order) ----
        industry_kw = config.get("industry_keywords", {})

        industry = None
        industry = _score_industry_from_filename(filename, industry_kw)
        if industry:
            reasoning_parts.append(f"Industry inferred from filename: '{industry}'.")
        else:
            text_first_3 = extract_text(file_path, max_pages=3)
            industry = _score_industry_from_text(text_first_3, industry_kw)
            if industry:
                reasoning_parts.append(f"Industry inferred from extracted text: '{industry}'.")
            else:
                industry = deployment_cfg.get("default_industry", "automotive")
                reasoning_parts.append(f"Industry defaulted from deployment config: '{industry}'.")

        best["industry"] = industry
        best["reasoning"] = "\n".join(reasoning_parts).strip()

        # ---- Confidence cutoff + fail gracefully ----
        low_thr = float(config.get("confidence_thresholds", {}).get("categorization_low_confidence", 0.5))
        if best["confidence"] < low_thr:
            best["route"] = "text_default"
            state["errors"].append(
                f"categorize: low confidence ({best['confidence']:.2f}), flagged for review"
            )

    except Exception as e:
        state["errors"].append(f"categorize: exception {type(e).__name__}: {e}")
        # Route fallback already in defaults.

    # ---- Write required state fields (always) ----
    state["route"] = best["route"]
    state["document_type"] = best["document_type"]
    state["industry"] = best["industry"]
    state["categorization_confidence"] = best["confidence"]
    state["reasoning"] = best["reasoning"]

    return {
        "route": state["route"],
        "document_type": state["document_type"],
        "industry": state["industry"],
        "confidence": state["categorization_confidence"],
        "reasoning": state["reasoning"],
    }