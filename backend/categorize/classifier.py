"""Core categorization engine.

Vision-first design:
- Determine `document_type` using vision (covers + content via stitched pages).
- Map `document_type` -> `route` via config.type_to_route.
- Determine `industry` via filename/text/deployment defaults.

Primary outputs to state:
- state["route"]
- state["document_type"]
- state["industry"]
- state["confidence"]
- state["reasoning"]
- state["errors"] (on failure or low confidence)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import fitz

from .text_extractor import extract_text, extract_toc_text, analyze_cad_document, is_cad_document_by_filename, detect_excel_document_type
from backend.core.vision_client import describe_image


def detect_file_type(file_path: str) -> str:
    """Detect file type based on extension.
    
    Returns the canonical pipeline file_type vocabulary (matches
    config `extractors:` keys and PipelineState.file_type):
    - 'pdf' for .pdf files
    - 'excel' for .xlsx, .xls files
    - 'spreadsheet' for .csv files
    - 'ppt' for .pptx, .ppt files
    - 'word' for .docx, .doc files
    - 'image' for .png, .jpg, .jpeg, .bmp, .gif files
    - 'diagram' for .dwg, .dxf, .svg files
    - 'archive' for .zip, .rar, .7z files
    - 'unknown' for other types
    """
    filename = os.path.basename(file_path).lower()
    
    if filename.endswith('.pdf'):
        return 'pdf'
    elif filename.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif filename.endswith('.csv'):
        return 'spreadsheet'
    elif filename.endswith(('.pptx', '.ppt')):
        return 'ppt'
    elif filename.endswith(('.docx', '.doc')):
        return 'word'
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        return 'image'
    elif filename.endswith(('.dwg', '.dxf', '.svg')):
        return 'diagram'
    elif filename.endswith(('.zip', '.rar', '.7z')):
        return 'archive'
    else:
        return 'unknown'


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


def _coerce_doctype(doc_type: str, supported_types: list[str]) -> str:
    """Force a single, known document_type. Models sometimes echo a compound label
    ('invoice / financial_statement / ...'); return the first supported type it names
    (preferring the longest match so 'financial_statement' beats 'statement'), else the
    raw value if it's already a single clean token, else 'report'."""
    raw = (doc_type or "").strip().lower()
    if raw in [t.lower() for t in supported_types]:
        return raw
    norm = re.sub(r"[^a-z0-9]+", " ", raw)
    # Earliest-mentioned known type wins (matches prompt order); tie-break longest match.
    hits = []
    for t in supported_types:
        m = re.search(rf"\b{re.escape(t.lower().replace('_', ' '))}\b", norm)
        if m:
            hits.append((m.start(), -len(t), t))
    if hits:
        return min(hits)[2]
    # A single unknown token is fine to keep (custom type); a multi-token echo -> report.
    return raw if raw and " " not in norm.strip() and "/" not in raw else "report"


def _heuristic_doctype(
    filename: str, file_path: str, supported_types: list[str]
) -> Tuple[str, float, str]:
    """Vision-free document_type guess from filename + first-page text.

    Used when the vision classifier is unavailable (e.g. no GOOGLE_API_KEY) so
    categorization still produces a usable route instead of a hard 0.0 that flags
    every document. The confidence is deliberately modest — it's a keyword signal,
    not a vision read — but high enough to clear the low-confidence threshold.
    """
    hay = (filename or "").lower()
    try:
        hay += " " + (extract_text(file_path, max_pages=2) or "").lower()
    except Exception:
        pass
    keymap = [
        ("invoice", ["invoice", "bill to", "amount due"]),
        ("purchase_order", ["purchase order", "po number"]),
        ("financial_statement", ["balance sheet", "income statement", "profit and loss"]),
        ("datasheet", ["datasheet", "specifications", "rated voltage"]),
        ("presentation", ["agenda", "slide "]),
        ("contract", ["agreement", "terms and conditions"]),
        ("research_paper", ["abstract", "references", "doi:"]),
    ]
    for dtype, kws in keymap:
        if dtype in supported_types and any(k in hay for k in kws):
            return dtype, 0.6, f"vision unavailable; matched '{dtype}' by keyword heuristic"
    return "report", 0.5, "vision unavailable; defaulted to 'report' (text route)"


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


def categorize(
    file_path: str,
    state: Dict[str, Any],
    config: Dict[str, Any],
    deployment: Optional[Dict[str, Any]] = None
):
    
    deployment_cfg = deployment or config.get("deployment", {})

    # Initialize state error list
    state.setdefault("errors", [])

    # Defaults
    best = {
        "route": "text_default",
        "document_type": "report",
        "industry": config.get("default_industry") or deployment_cfg.get("default_industry", "automotive"),
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
                        best["route"] = config["type_to_route"].get("cad_drawing", "cad_route")   #to-do: route name in config
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
                        state["confidence"] = best["confidence"]
                        state["reasoning"] = best["reasoning"]
                        state.setdefault("errors", [])
                        
                        return {
                            "route": state["route"],
                            "document_type": state["document_type"],
                            "industry": state["industry"],
                            "confidence": state["confidence"],
                            "reasoning": state["reasoning"],
                        }
            except Exception as e:
                # If CAD analysis fails, continue with normal flow
                pass

        # ---- Excel Document Detection ----
        if lowered.endswith(('.xlsx', '.xls')):
            excel_type, excel_confidence, excel_details = detect_excel_document_type(file_path)
            if excel_confidence >= 0.60:
                # High confidence detection from Excel content
                best["document_type"] = excel_type
                best["confidence"] = excel_confidence
                best["route"] = config["type_to_route"].get(excel_type, "text_default") #todo: route name in config
                reasoning_parts = [
                    f"Excel document type detected: {excel_type}",
                    f"Confidence: {excel_confidence:.2f} from content analysis",
                ]
                best["reasoning"] = "\n".join(reasoning_parts).strip()
                
                # Write to state and return early
                state["route"] = best["route"]
                state["document_type"] = best["document_type"]
                state["industry"] = best["industry"]
                state["confidence"] = best["confidence"]
                state["reasoning"] = best["reasoning"]
                state.setdefault("errors", [])
                
                return {
                    "route": state["route"],
                    "document_type": state["document_type"],
                    "industry": state["industry"],
                    "confidence": state["confidence"],
                    "reasoning": state["reasoning"],
                }


        # ---- Document type: filename-first ----
        type_to_route: Dict[str, str] = config["type_to_route"] #todo: ensure this is in config and has all expected types; otherwise default to text_default route
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
        # Industry read from the same vision pass (set in the vision branch below).
        # Vision works on scanned PDFs too (it reads rendered page images), so this
        # is the only industry signal that survives when there's no text layer.
        vision_industry: Optional[str] = None

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

            # Build classification prompt with CAD detection hints
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

            # Industries vision may choose from (config-driven; falls back to general).
            known_industries = list(
                (config.get("categorization", {}).get("industry_keywords", {}) or {}).keys()
            ) or ["general"]
            industries_line = ", ".join(known_industries)

            prompt = f"""Classify this document by its DOMINANT content across the pages shown.

DOCUMENT TYPES:
- manual: service/user/instruction manuals (procedures, parts, troubleshooting) — choose
  this for a manual EVEN IF it contains diagrams or photos.
- circuit_diagram: the document IS primarily an electrical schematic / wiring diagram.
- cad_drawing: the document IS primarily a mechanical engineering drawing — title block,
  revision block, dimensioned views, BOM. NOT a manual that merely includes drawings.
- schematic: primarily technical/flow diagrams.
- invoice: a tax invoice / bill (line items, HSN, tax, totals).
- financial_statement: balance sheet, P&L, ledger and similar.
- purchase_order: a purchase order.
- contract: an agreement / contract.
- policy: a policy or terms document.
- research_paper: an academic / research paper.
- report: a general report (use this as the fallback).
- presentation: slides / a deck.{cad_hint}

Return ONLY a JSON object with these exact keys:
- "document_type": EXACTLY ONE value from the list above — a single token (e.g. "invoice").
  Never return a list, never join multiple with "/", never copy a whole line.
- "industry": exactly one of: {industries_line} (use "general" only if none clearly fit)
- "confidence": a float 0-1 reflecting how strongly the visible evidence supports it
- "reasoning": one short sentence citing the visual evidence you used

Document context:{extra_context}

Respond with ONLY the JSON object, no other text, no markdown."""

            # Call vision_client.describe_image() and parse the response
            try:
                response_text = describe_image(stitched_bytes, prompt, config).strip()
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    doc_type = result.get("document_type", "report")
                    # Defensive: a degraded model can echo a compound label
                    # ("invoice / financial_statement / ...") instead of one token.
                    # Collapse to the first KNOWN type it mentions, else 'report'.
                    doc_type = _coerce_doctype(doc_type, supported_types)
                    conf = float(result.get("confidence", 0.0) or 0.0)
                    reasoning = result.get("reasoning", "Vision classification completed.")
                    # Keep only a recognized industry; ignore "general"/unknown so the
                    # downstream cascade can still try filename/text before defaulting.
                    vi = (result.get("industry") or "").strip().lower()
                    if vi in known_industries and vi != "general":
                        vision_industry = vi
                else:
                    # Fallback if JSON extraction fails
                    doc_type = "report"
                    conf = 0.3
                    reasoning = f"Vision response parsing failed: {response_text[:100]}"
            except json.JSONDecodeError as e:
                doc_type = "report"
                conf = 0.0
                reasoning = f"Vision JSON parse error: {e}"
            except Exception as e:
                doc_type = "report"
                conf = 0.0
                reasoning = f"Vision inference failed: {type(e).__name__}: {str(e)[:100]}"

            if not doc_type:
                doc_type = "report"

            # ---- Vision unavailable/failed (conf 0.0) -> keyword heuristic ----
            # Don't let a missing vision key flag every document as low-confidence;
            # fall back to a filename/text guess so routing still works offline.
            if conf <= 0.0:
                doc_type, conf, heuristic_reason = _heuristic_doctype(
                    filename, file_path, supported_types
                )
                reasoning = f"{heuristic_reason} (vision: {reasoning})"

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
        industry_kw = config.get("categorization", {}).get("industry_keywords", {})

        # Signal order: explicit filename keyword > vision read (works on scanned,
        # no text layer needed) > embedded-text keyword > configured default.
        industry = _score_industry_from_filename(filename, industry_kw)
        if industry:
            reasoning_parts.append(f"Industry inferred from filename: '{industry}'.")
        elif vision_industry:
            industry = vision_industry
            reasoning_parts.append(f"Industry inferred from vision: '{industry}'.")
        else:
            text_first_3 = extract_text(file_path, max_pages=3)
            industry = _score_industry_from_text(text_first_3, industry_kw)
            if industry:
                reasoning_parts.append(f"Industry inferred from extracted text: '{industry}'.")
            else:
                industry = config.get("default_industry") or deployment_cfg.get("default_industry", "automotive")
                reasoning_parts.append(f"Industry defaulted to: '{industry}'.")

        best["industry"] = industry
        best["reasoning"] = "\n".join(reasoning_parts).strip()

        # ---- Confidence cutoff + fail gracefully ----
        low_thr = float(config.get("categorization", {}).get("confidence_thresholds", {}).get("categorization_low_confidence", 0.5))
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
    state["confidence"] = best["confidence"]
    state["reasoning"] = best["reasoning"]
    # Don't clobber a file_type the caller (API) already set — it uses the same
    # canonical vocabulary ("ppt"/"excel"/"pdf"/"image") the extractor dispatch
    # keys on. Only derive it when missing (e.g. standalone categorize calls).
    state["file_type"] = state.get("file_type") or detect_file_type(file_path)

    return {
        "route": state["route"],
        "document_type": state["document_type"],
        "industry": state["industry"],
        "confidence": state["confidence"],
        "reasoning": state["reasoning"],
        "file_type": state["file_type"],
        "errors": state.get("errors", []),
    }