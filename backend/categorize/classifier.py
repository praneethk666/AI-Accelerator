"""Core categorization engine.

Vision-first design:
- Determine `document_type` from content: vision for PDFs/images (rendered pages),
  a text LLM for word/ppt/excel/csv (extracted text). The filename + parent folder
  are supplied as a WEAK HINT only (helpful in structured corpora where folders
  name the type, e.g. '03_Instruction manual') — content is the primary signal and
  wins on conflict. The prompt's type menu is built from config['document_types'].
- Map `document_type` -> `route` via config.type_to_route.
- Determine `industry` from the same vision/text-LLM read, else a text
  keyword scan, else the configured default. A vision/text-LLM industry
  pick is only trusted if the model's claimed evidence phrase is actually
  found in the extracted/OCR'd text (see _evidence_supported) — this stops
  a model from asserting a specific industry with no real grounding.

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

from .text_extractor import extract_text, extract_toc_text
from backend.core.vision_client import describe_image
from backend.core.llm_client import get_llm_for, resolve_model_provider


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
        return 'docx'
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        return 'image'
    elif filename.endswith(('.dwg', '.dxf', '.svg')):
        return 'diagram'
    elif filename.endswith(('.zip', '.rar', '.7z')):
        return 'archive'
    else:
        return 'unknown'


# Weighted, tiered scoring — real finding (27-Jul): a flat keyword list let 3 WEAK
# automotive terms (torque, chassis, transmission — generic mechanical vocabulary
# shared with any industrial manual) cluster together and false-positive even past
# a naive "2 distinct hits" threshold. Each industry's keywords are now tiered in
# config: strong (brand names, acronyms — genuinely exclusive to that domain; ONE
# hit is real evidence) vs weak (plausible but shared with adjacent domains; needs
# more corroboration). Score = 2x per strong hit + 1x per weak hit; a single strong
# hit (score=2) qualifies, so does 2+ weak hits, but one weak hit alone (score=1)
# correctly does not.
_STRONG_WEIGHT = 2
_WEAK_WEIGHT = 1
_MIN_INDUSTRY_SCORE = 2


def _industry_keyword_evidence(text: str, industry_keywords: Dict[str, dict]) -> Dict[str, dict]:
    """Every industry's matched keywords, tier-separated, with a weighted score —
    used both to pick a fallback industry and, separately, as grounding evidence fed
    into the vision prompt so the LLM can weigh real document-wide keyword signal
    alongside what it actually sees on the rendered pages (see categorize() below).
    Returns {industry: {"strong": [...], "weak": [...], "score": int}}."""
    hay = (text or "").lower()
    evidence: Dict[str, dict] = {}
    for industry, tiers in industry_keywords.items():
        if not isinstance(tiers, dict):
            continue
        strong_hits = [k for k in (tiers.get("strong") or [])
                       if re.search(rf"\b{re.escape(k.lower())}\b", hay)]
        weak_hits = [k for k in (tiers.get("weak") or [])
                     if re.search(rf"\b{re.escape(k.lower())}\b", hay)]
        if strong_hits or weak_hits:
            score = _STRONG_WEIGHT * len(strong_hits) + _WEAK_WEIGHT * len(weak_hits)
            evidence[industry] = {"strong": strong_hits, "weak": weak_hits, "score": score}
    return evidence


def _score_industry_from_text(text: str, industry_keywords: Dict[str, dict]) -> Optional[str]:
    evidence = _industry_keyword_evidence(text, industry_keywords)
    best_industry, best_score = None, 0
    for industry, ev in evidence.items():
        if ev["score"] > best_score:
            best_industry, best_score = industry, ev["score"]
    if best_score >= _MIN_INDUSTRY_SCORE:
        return best_industry
    return None


def _evidence_supported(evidence: str, *source_texts: str) -> bool:
    """Require the model's claimed justification phrase to actually appear in
    the text it was given, so an industry pick can't be accepted purely on
    the model's say-so. Normalizes case/punctuation for a forgiving substring
    match — not exact-string, since models paraphrase slightly even when
    quoting."""
    if not evidence or not evidence.strip():
        return False
    ev = re.sub(r"[^a-z0-9 ]+", " ", evidence.lower()).strip()
    ev = re.sub(r"\s+", " ", ev)
    if not ev:
        return False
    for src in source_texts:
        if not src:
            continue
        hay = re.sub(r"[^a-z0-9 ]+", " ", src.lower())
        hay = re.sub(r"\s+", " ", hay)
        if ev in hay:
            return True
    return False


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
    file_path: str, supported_types: list[str], file_type: str = ""
) -> Tuple[str, float, str]:
    """Content-only document_type guess, used as a last resort when the
    text-LLM classification call itself fails (bad key, network error, etc.) —
    a keyword scan over extracted page/sheet/slide text. No filename involved.
    """
    hay = ""
    try:
        hay = (extract_text(file_path, max_pages=3) or "").lower()
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
            return dtype, 0.6, f"text-LLM unavailable; matched '{dtype}' by keyword heuristic on extracted text"
    # A raw image file with no matching document keywords is far more likely to
    # be a plain photo/screenshot than a "report" — default it to 'image' (routes
    # via image_route) instead of the generic text-document fallback.
    if file_type == "image" and "image" in supported_types:
        return "image", 0.6, "text-LLM/vision unavailable; defaulted to 'image' (raw image file, no document keywords found)"
    return "report", 0.5, "text-LLM unavailable; defaulted to 'report' (text route)"


# Human-readable hints for the prompt. The ENUM is config["document_types"] (single
# source of truth); this only describes each type. Add a type in BOTH config and here.
_DOCTYPE_HINTS = {
    "invoice": "a tax invoice / bill (line items, HSN, tax, totals)",
    "purchase_order": "a purchase order (ordered items, PO number, vendor)",
    "financial_statement": "balance sheet, P&L, ledger and similar",
    "report": "a general report — only if the content actually reads as a report",
    "research_paper": "an academic / research / scientific publication",
    "manual": "service/user/instruction manual (procedures, parts, troubleshooting) — even if it contains diagrams/photos",
    "contract": "an agreement / contract",
    "policy": "a policy or terms document",
    "datasheet": "a component/product datasheet (specs, ratings, pinouts)",
    "presentation": "slides / a deck",
    "spreadsheet": "structured tabular data in rows and columns",
    "cad_drawing": "a mechanical engineering drawing — title block, revision block, dimensioned views, BOM; NOT a manual that merely includes drawings",
    "circuit_diagram": "an electrical schematic / wiring diagram",
    "schematic": "primarily technical/flow diagrams (not a full CAD/circuit drawing)",
    "image": "a plain photo/screenshot/graphic with NO structured document layout",
    "unknown": "not any of the above, or too ambiguous to classify",
}


def _doctype_menu(config: Dict[str, Any]) -> str:
    """Build the prompt's DOCUMENT TYPES menu from config['document_types'] so the model
    can only pick a type the router actually knows (no silent collapse to 'report')."""
    types = config.get("document_types") or list(_DOCTYPE_HINTS)
    return "\n".join(
        f"- {t}: {_DOCTYPE_HINTS[t]}" if t in _DOCTYPE_HINTS else f"- {t}"
        for t in types
    )


def _path_hint(file_path: str) -> str:
    """filename + parent folder as a WEAK hint. In structured corpora the folder
    (e.g. '03_Instruction manual', '07_Electronic data') is highly informative; the
    prompt tells the model to weigh it as a hint only and let page content win on conflict."""
    fn = os.path.basename(file_path)
    folder = os.path.basename(os.path.dirname(file_path.rstrip("/")))
    return f"filename: {fn}" + (f"\nparent folder: {folder}" if folder else "")


def _text_llm_classify(
    text: str, known_industries: list[str], config: Dict[str, Any], file_path: str = ""
) -> Tuple[str, float, str, Optional[str], str]:
    """Classify a document from extracted TEXT CONTENT (plus filename/folder as a weak
    hint) using the configured text LLM (backend.core.llm_client.get_llm_for). Used for
    file types vision can't render here (word/ppt/excel/csv) — same document-type
    vocabulary as the vision prompt, same JSON contract.

    Returns (document_type, confidence, reasoning, industry_or_None, industry_evidence).
    industry_evidence is the model's claimed quoted justification for its industry
    pick (empty string if industry is None/general) — the caller is responsible for
    verifying this evidence actually appears in the source text before trusting the
    industry (see _evidence_supported).
    """
    industries_line = ", ".join(known_industries)
    path_hint = _path_hint(file_path) if file_path else ""
    hint_block = (
        f"\n\nFILENAME / FOLDER (a HINT only — page content is the primary signal; "
        f"ignore it if it conflicts with the content):\n{path_hint}" if path_hint else ""
    )
    prompt = f"""Classify this document by its DOMINANT content, based mainly on the text below.

DOCUMENT TYPES:
{_doctype_menu(config)}

Return ONLY a JSON object with these exact keys:
- "document_type": EXACTLY ONE value from the list above — a single token (e.g. "invoice").
- "industry": "general" OR exactly one of: {industries_line}. Only choose a
  specific industry if the text content contains clear, specific
  terminology tied to that industry — not just because the document looks
  technical or professional. Default to "general" for generic diagrams,
  architecture/flow charts, templates, or cross-industry content.
- "industry_evidence": if industry is not "general", quote the EXACT short
  phrase (2-6 words) of text that justifies the industry choice.
  Use "" if industry is "general".
- "confidence": a float 0-1 reflecting how strongly the text supports it
- "reasoning": one short sentence citing the textual evidence you used

Document text (truncated):
{text[:4000]}{hint_block}

Respond with ONLY the JSON object, no other text, no markdown."""

    # per-step override: categorization.model falls back to global llm.
    llm = get_llm_for(config, config.get("categorization"))
    response = llm.invoke(prompt)
    response_text = (getattr(response, "content", None) or str(response)).strip()
    try:
        from backend.core import usage
        model_name, provider_name = resolve_model_provider(config, config.get("categorization"))
        usage.record_from_message("categorize", response, prompt=prompt,
                                  provider=provider_name, model=model_name)
    except Exception:
        pass

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return "report", 0.3, f"Text-LLM response parsing failed: {response_text[:100]}", None, ""

    result = json.loads(json_match.group())
    doc_type = result.get("document_type", "unknown")
    conf = float(result.get("confidence", 0.0) or 0.0)
    reasoning = result.get("reasoning", "Text-LLM classification completed.")
    industry = (result.get("industry") or "").strip().lower()
    evidence = (result.get("industry_evidence") or "").strip()
    industry = industry if industry in known_industries and industry != "general" else None
    return doc_type, conf, reasoning, industry, evidence


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
        "industry": config.get("default_industry") or deployment_cfg.get("default_industry", "general"),
        "confidence": 0.0,
        "reasoning": "",
    }

    try:
        filename = os.path.basename(file_path)
        lowered = filename.lower()

        type_to_route: Dict[str, str] = config["type_to_route"]
        supported_types = list(type_to_route.keys())

        reasoning_parts = []
        # Industry read from the same vision/text-LLM pass (set below).
        # Vision works on scanned PDFs too (it reads rendered page images), so this
        # is the only industry signal that survives when there's no text layer.
        vision_industry: Optional[str] = None

        # Industries the classifier may choose from (config-driven; falls back to general).
        industry_kw = config.get("categorization", {}).get("industry_keywords", {}) or {}
        known_industries = list(industry_kw.keys()) or ["general"]

        # ---- Get something vision can look at ----
        # PDFs: render+stitch pages 1-3. Images: read the raw bytes directly.
        # Other types (word/ppt/excel/csv/archive/unknown) have no page-renderer
        # in this module, so vision genuinely cannot run on them here — that's
        # a capability gap, not a decision to skip vision, and it's handled
        # below by running the same classification through a TEXT LLM on the
        # extracted content instead (never by guessing via filename matching).
        file_type = detect_file_type(file_path)
        image_bytes: Optional[bytes] = None
        toc_text = ""
        verify_text = ""
        keyword_evidence: Dict[str, dict] = {}
        if file_type == "pdf":
            toc_text = extract_toc_text(file_path, max_pages=2)
            image_bytes = _render_pdf_pages_to_stitched_image_bytes(file_path, [0, 1, 2])
            # Text used to verify a claimed industry actually has grounding in the
            # doc (TOC + first pages) — separate from the image sent to vision.
            verify_text = f"{toc_text}\n{extract_text(file_path, max_pages=3)}"
            # WIDER scan (free — native PDF text extraction, no LLM call) fed into the
            # prompt below as grounding: vision only SEES pages 1-3, but a real industry
            # signal appearing later in a long manual (spec/procedure pages) is still
            # meaningful even if invisible on the pages shown. Real finding, 27-Jul: a
            # 105-page manual's genuine industry vocabulary (circuit/voltage/signal)
            # only showed up past page 3 — the title/TOC/safety front matter had none.
            keyword_evidence = _industry_keyword_evidence(
                extract_text(file_path, max_pages=30), industry_kw)
        elif file_type == "image":
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            # OCR text (if pytesseract is available) is the only verification
            # source for a raw image — may legitimately be empty.
            verify_text = extract_text(file_path)

        # Resolve industry from the keyword scan FIRST when it's strong enough, before
        # even looking at what vision says — empirically more reliable for this
        # specific sub-field (live-tested 27-Jul: 3 real runs, keyword scan correct
        # all 3 times; vision's own industry pick right ~1/3, otherwise defaulting to
        # "general"). Vision is still asked below (still needed for document_type,
        # which genuinely benefits from visual judgment), but its industry answer is
        # only consulted if the keyword scan didn't already resolve this — free,
        # deterministic, and doesn't depend on the vision call succeeding at all.
        _top_kw_industry, _top_kw_score = None, 0
        for _industry, _ev in keyword_evidence.items():
            if _ev["score"] > _top_kw_score:
                _top_kw_industry, _top_kw_score = _industry, _ev["score"]
        if _top_kw_score >= _MIN_INDUSTRY_SCORE:
            vision_industry = _top_kw_industry
            _kw = keyword_evidence[_top_kw_industry]
            reasoning_parts.append(
                f"Industry '{_top_kw_industry}' resolved from document-wide keyword "
                f"evidence (score={_top_kw_score}: strong={_kw['strong']}, "
                f"weak={_kw['weak']}) before vision was asked — more reliable for "
                f"this field per live testing."
            )

        if image_bytes is not None:
            # ---- Vision-first (and vision-final) ----
            stitched_bytes = image_bytes

            # Classification prompt = page content (primary) + a weak filename/folder
            # hint (added to extra_context below). Content wins on conflict.
            extra_context = ""
            if toc_text:
                extra_context += f"\nTable of Contents excerpt:\n{toc_text[:500]}"
            path_hint = _path_hint(file_path)
            if path_hint:
                extra_context += (
                    f"\nFILENAME / FOLDER (a HINT only — the page content shown is the "
                    f"primary signal; ignore it if it conflicts):\n{path_hint}"
                )
            # Only include keyword evidence in the prompt (and ask about industry at
            # all) if the keyword scan DIDN'T already resolve it above — no point
            # burdening the prompt/output with a question we're not going to use the
            # answer to.
            ask_industry = vision_industry is None
            if ask_industry and keyword_evidence:
                ev_lines = "\n".join(
                    f"  - {industry}: strong={ev['strong'] or '-'}, weak={ev['weak'] or '-'}"
                    for industry, ev in sorted(keyword_evidence.items(),
                                                key=lambda kv: -kv[1]["score"])
                )
                extra_context += (
                    f"\nKEYWORD EVIDENCE (from a scan of the WHOLE document's text, not "
                    f"just the pages shown — a real signal even if not visible here; "
                    f"weigh it alongside what you actually see, don't just copy the top "
                    f"one blindly — a single generic ('weak') term isn't strong evidence, "
                    f"look for multiple weak terms or one strong term for one "
                    f"industry):\n{ev_lines}"
                )

            industries_line = ", ".join(known_industries)

            # Only ask about industry when the keyword scan didn't already resolve it
            # (see ask_industry above) — saves output tokens and avoids vision
            # potentially disagreeing with an already-reliable free signal.
            if ask_industry:
                industry_fields = f"""- "industry": "general", one of the KNOWN industries: {industries_line},
  OR — only if genuinely justified and none of the known ones fit — a SHORT new
  industry name not in that list (e.g. "aerospace", "construction"). Propose a new
  one only when the visible content clearly, specifically indicates a domain none
  of the known industries cover; this is rare, not a way to avoid picking a known
  one. Choose a KNOWN specific industry if the visible pages contain clear,
  specific terminology tied to it, or the KEYWORD EVIDENCE section below (if
  present) shows real support for one. Default to "general" when nothing supports
  a specific industry, or the content is genuinely generic/cross-industry
  (templates, flow charts, etc.) — not just because it looks technical/professional.
- "industry_evidence": if industry is not "general", quote the EXACT short phrase
  (2-6 words) of VISIBLE text that justifies it. REQUIRED for a NEW industry name
  not in the known list — a proposal with no quotable evidence will be discarded.
  Use "" if industry is "general".
"""
            else:
                industry_fields = ""

            prompt = f"""Classify this document by its DOMINANT content across the pages shown.

DOCUMENT TYPES:
{_doctype_menu(config)}

Return ONLY a JSON object with these exact keys:
- "document_type": EXACTLY ONE value from the list above — a single token (e.g. "invoice").
  Never return a list, never join multiple with "/", never copy a whole line.
{industry_fields}- "confidence": a float 0-1 reflecting how strongly the visible evidence supports it
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
                    doc_type = result.get("document_type", "unknown")
                    # Defensive: a degraded model can echo a compound label
                    # ("invoice / financial_statement / ...") instead of one token.
                    # Collapse to the first KNOWN type it mentions, else 'report'.
                    doc_type = _coerce_doctype(doc_type, supported_types)
                    conf = float(result.get("confidence", 0.0) or 0.0)
                    reasoning = result.get("reasoning", "Vision classification completed.")
                    MIN_VERIFY_TEXT_CHARS = 40  # if less than this, we can't verify the model's claimed evidence
                    if not ask_industry:
                        pass  # already resolved from the keyword scan before this call
                    else:
                        # Keep only a recognized/proposed industry; ignore "general" so the
                        # downstream cascade can still try filename/text before defaulting.
                        vi = (result.get("industry") or "").strip().lower()
                        evidence = (result.get("industry_evidence") or "").strip()
                        if vi and vi in known_industries and vi != "general":
                            # NOTE: no keyword-corroboration branch here — if the keyword
                            # scan's TOP score had cleared _MIN_INDUSTRY_SCORE, pre-resolution
                            # above would have already fired and we'd never reach this code
                            # (ask_industry would be False). So by construction, every
                            # industry's keyword score is BELOW threshold whenever we get
                            # here — verification must come from what vision actually saw.
                            if len(verify_text.strip()) < MIN_VERIFY_TEXT_CHARS:
                                # Nothing to verify against (pure-visual content, e.g. CAD/vector
                                # drawings) — trust the model's own confidence, with a HIGHER bar.
                                if conf >= 0.75:
                                    vision_industry = vi
                                    reasoning_parts.append(
                                        f"No extractable text to verify industry evidence "
                                        f"(len={len(verify_text.strip())}); accepted '{vi}' on "
                                        f"confidence={conf:.2f} alone."
                                    )
                                else:
                                    reasoning_parts.append(
                                        f"No extractable text to verify industry evidence and "
                                        f"confidence={conf:.2f} too low to accept '{vi}' unverified."
                                    )
                            elif _evidence_supported(evidence, verify_text):
                                vision_industry = vi
                            else:
                                reasoning_parts.append(
                                    f"Vision claimed industry='{vi}' but evidence '{evidence}' "
                                    f"wasn't found in extractable text; discarding (falling through "
                                    f"to keyword scan/default)."
                                )
                        elif vi and vi != "general":
                            # NOVEL industry — not in our configured taxonomy, so no keyword
                            # list exists to corroborate it. No shortcut available: require
                            # BOTH a verified quote from what vision actually saw AND high
                            # confidence. Accepted, but flagged distinctly in the reasoning so
                            # a human can decide whether to promote it into the permanent
                            # config (with its own keyword list) rather than it silently
                            # becoming a one-off unofficial category.
                            if (len(verify_text.strip()) >= MIN_VERIFY_TEXT_CHARS
                                    and _evidence_supported(evidence, verify_text)
                                    and conf >= 0.75):
                                vision_industry = vi
                                reasoning_parts.append(
                                    f"NEW industry candidate '{vi}' (not in the configured "
                                    f"list) — verified evidence '{evidence}', "
                                    f"confidence={conf:.2f}. Flagged for human review before "
                                    f"promoting to a permanent category."
                                )
                            else:
                                reasoning_parts.append(
                                    f"Vision proposed new industry '{vi}' (not in the "
                                    f"configured list) but evidence/confidence wasn't strong "
                                    f"enough to accept unverified; discarding."
                                )
                else:
                    # Fallback if JSON extraction fails
                    doc_type = "unknown"
                    conf = 0.3
                    reasoning = f"Vision response parsing failed: {response_text[:100]}"
            except json.JSONDecodeError as e:
                doc_type = "unknown"
                conf = 0.0
                reasoning = f"Vision JSON parse error: {e}"
            except Exception as e:
                doc_type = "unknown"
                conf = 0.0
                reasoning = f"Vision inference failed: {type(e).__name__}: {str(e)[:100]}"

            if not doc_type:
                doc_type = "unknown"

            # ---- Vision unavailable/failed (conf 0.0) -> content-only keyword heuristic ----
            # Don't let a missing vision key flag every document as low-confidence;
            # fall back to a text-content guess so routing still works offline.
            if conf <= 0.0:
                doc_type, conf, heuristic_reason = _heuristic_doctype(
                    file_path, supported_types, file_type
                )
                reasoning = f"{heuristic_reason} (vision: {reasoning})"

            # No override here on purpose, and no filename signal fed the prompt
            # either: vision's document_type comes from the page content alone
            # and is authoritative.
            best["document_type"] = doc_type
            best["confidence"] = conf
            best["route"] = type_to_route.get(doc_type, "text_default")
            reasoning_parts.append(f"Vision predicted document_type='{doc_type}' with confidence={conf:.2f}.")
            if reasoning:
                reasoning_parts.append(f"Vision reasoning: {reasoning}")
        else:
            # No page-renderer for this file_type (word/ppt/excel/csv/archive/unknown) —
            # vision cannot see it here. Classify from extracted TEXT CONTENT via the
            # configured text LLM instead of guessing from the filename.
            extracted = extract_text(file_path, max_pages=5)
            if extracted.strip():
                try:
                    doc_type, conf, reasoning, text_industry, text_evidence = _text_llm_classify(
                        extracted, known_industries, config, file_path=file_path
                    )
                    doc_type = _coerce_doctype(doc_type, supported_types)
                    if text_industry and _evidence_supported(text_evidence, extracted):
                        vision_industry = text_industry  # reuse the same downstream slot
                    elif text_industry:
                        reasoning_parts.append(
                            f"Text-LLM claimed industry='{text_industry}' but evidence "
                            f"'{text_evidence}' wasn't found in extracted text; discarding."
                        )
                    reasoning_parts.append(
                        f"Text-LLM predicted document_type='{doc_type}' with confidence={conf:.2f} "
                        f"(file_type='{file_type}', no vision renderer available)."
                    )
                    if reasoning:
                        reasoning_parts.append(f"Text-LLM reasoning: {reasoning}")
                except Exception as e:
                    doc_type, conf, heuristic_reason = _heuristic_doctype(file_path, supported_types)
                    reasoning_parts.append(
                        f"Text-LLM call failed ({type(e).__name__}: {str(e)[:100]}); {heuristic_reason}"
                    )
            else:
                # No extractable text either (e.g. empty/corrupt file, archive) —
                # last resort is the keyword heuristic, which degrades to 'report'.
                doc_type, conf, heuristic_reason = _heuristic_doctype(file_path, supported_types)
                reasoning_parts.append(
                    f"No extractable text for file_type='{file_type}'; {heuristic_reason}"
                )

            best["document_type"] = doc_type
            best["confidence"] = conf
            best["route"] = type_to_route.get(doc_type, "text_default")

        # ---- Industry detection (content-only, 2 signals order) ----
        # Signal order: vision/text-LLM read (already comes from actual content,
        # whichever branch above ran, and passed the evidence check) > embedded-text
        # keyword scan > configured default. No filename signal at all — same
        # reasoning as document_type: a filename tells you nothing reliable about
        # the file's actual industry.
        if vision_industry:
            industry = vision_industry
            reasoning_parts.append(f"Industry inferred from content (vision/text-LLM): '{industry}'.")
        else:
            # 3 pages found ZERO signal on a real 105-page manual (title/TOC/safety
            # pages don't carry industry vocabulary) — widened to 30. Free: native
            # PDF text extraction, no LLM call, so more pages costs time, not money.
            text_sample = extract_text(file_path, max_pages=30)
            industry = _score_industry_from_text(text_sample, industry_kw)
            if industry:
                reasoning_parts.append(f"Industry inferred from extracted text: '{industry}'.")
            else:
                industry = config.get("default_industry") or deployment_cfg.get("default_industry", "general")
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