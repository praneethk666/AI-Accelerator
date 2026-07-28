"""Vision-LLM page transcription for COMPLEX pages (routed, not every page).

Why: native text extraction can be garbled (broken font encodings) and pymupdf /
OCR table detection is unreliable, but a document VLM reads the rendered page like
a human — clean text + correct tables — regardless of the underlying font/scan. So
for the pages that need it we render the page and ask a VLM to transcribe it.

Routing (config `vision_ocr`):
  mode: off      -> never (pure native/OCR)
  mode: complex  -> ONLY pages flagged complex: a table, a figure region, or garbled
                    text. Plain text pages stay on the cheap path. (default)
  mode: always   -> every page through the VLM (max quality, more calls)
  route_gates    -> only apply on these document routes (e.g. text_default), so CAD/
                    circuit routes that have their own handling are untouched.

This runs in the MAIN process (just a fitz render + an API call) at the end of the
digital/scanned extract tools, operating on the block list (engine-agnostic). It
REPLACES a complex page's text/table blocks with the VLM's, and KEEPS any
image_caption blocks so figure captioning (vision_enrichment) still runs.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from collections import defaultdict

import fitz

from backend.core import prompts
from backend.core.paths import display_filename
from backend.core.vision_client import describe_image

logger = logging.getLogger(__name__)


def _page_of(block: dict):
    ref = block.get("source_ref") or {}
    return ref.get("page") if isinstance(ref, dict) else None


def _is_garbled(text: str, threshold: float = 0.25) -> bool:
    """True if a text block looks like broken-font garble.

    Primary signal: stray glyphs from a bad ToUnicode map (Ɣ ¿ ¶ and the À-ÿ Latin-1
    band) — these almost never appear in normal technical English, so even a few %
    means corruption. (A pure vowel-less test misses it: the substitution cipher
    still emits stray vowels like 'U'.) Fallback: lots of vowel-less long tokens."""
    s = text or ""
    if len(re.findall(r"\S+", s)) < 4:
        return False
    letters = len(re.findall(r"[A-Za-z]", s)) or 1
    weird = len(re.findall(r"[À-ÿƐ-ʒ¿¶]", s))
    if weird / letters > 0.05:                # >5% stray glyphs -> garbled
        return True
    words = re.findall(r"[A-Za-z]{4,}", s)
    if len(words) >= 6:
        novowel = sum(1 for w in words if not re.search(r"[aeiouAEIOU]", w))
        if novowel / len(words) >= threshold:
            return True
    return False


def _max_consonant_run(w: str) -> int:
    run = mx = 0
    for ch in w.lower():
        if ch in "aeiouy":
            run = 0
        else:
            run += 1
            mx = max(mx, run)
    return mx


def _gibberish_token(w: str) -> bool:
    """A long alphabetic token that isn't English-like. Broken-ToUnicode garble maps
    to plain ASCII with incidental vowels (so vowel tests miss it), but it produces
    improbable letter runs — a consonant run of 4+ almost never occurs in real English
    words. That single signal catches the cipher-style garble."""
    return len(w) >= 5 and _max_consonant_run(w) >= 4


def _line_garbled(s: str) -> bool:
    """Garble test for a SINGLE line — same signals as _is_garbled but tuned for the
    short length of one line (so partial-page garble isn't diluted away)."""
    letters = len(re.findall(r"[A-Za-z]", s)) or 1
    if len(re.findall(r"[À-ÿƐ-ʒ¿¶]", s)) / letters > 0.08:
        return True
    words = re.findall(r"[A-Za-z]{4,}", s)
    if len(words) >= 2:
        novowel = sum(1 for w in words if not re.search(r"[aeiouAEIOU]", w))
        if novowel / len(words) >= 0.5:
            return True
        gib = sum(1 for w in words if _gibberish_token(w))
        if gib / len(words) >= 0.4:
            return True
    return False


def text_is_unreliable(text: str, line_frac: float = 0.12) -> bool:
    """Whether a page's extracted text should be DISTRUSTED and re-read by the VLM.

    Page-level _is_garbled misses *partial* garble: when only some lines are corrupt
    (e.g. broken-font bullet lines on an otherwise-clean spec page — validated on the
    Argo electrical manual), the whole-page glyph ratio stays under threshold. So we
    also check per line and flag the page if a meaningful FRACTION of substantial
    lines are garbled."""
    if _is_garbled(text):
        return True
    lines = [ln for ln in (text or "").splitlines() if len(ln.strip()) >= 8]
    if len(lines) < 3:
        return False
    garbled = sum(1 for ln in lines if _line_garbled(ln))
    return garbled / len(lines) >= line_frac


def _block(document_id, page, filename, btype, text, table_data=None) -> dict:
    return {
        "block_id": str(uuid.uuid4()),
        "document_id": document_id,
        "type": btype,
        "text": text,
        "table_data": table_data,
        "source_ref": {"filename": filename, "page": page,
                       "sheet": None, "slide": None, "bbox": None},
        "confidence": 0.9,
        "language": "en",
        "metadata": {"source": "vision_ocr"},
    }

import re as _re
_BR_RE = _re.compile(r"<br\s*/?>", _re.IGNORECASE)
_TAG_RE = _re.compile(r"<[^>]+>")

def _clean_cell(s: str) -> str:
    """Normalize any HTML the model leaked into a cell: <br> -> '; ', strip any
    other stray tags. Net for PAGE_TRANSCRIBE/TABLE_TRANSCRIBE's '; not <br>' rule."""
    s = _BR_RE.sub("; ", s)
    s = _TAG_RE.sub("", s)
    return _re.sub(r"\s*;\s*", "; ", s).strip("; ").strip()

def _parse_md_table(rows_md: list[str]) -> dict | None:
    """Markdown table lines -> {headers, rows}, dropping the |---| separator row."""
    rows = []
    for ln in rows_md:
        cells = [_clean_cell(c) for c in ln.strip().strip("|").split("|")]
        if cells and all(c and set(c) <= set("-: ") for c in cells):
            continue  # separator row
        rows.append(cells)
    if not rows:
        return None
    return {"headers": rows[0], "rows": rows[1:]}


def _strip_fences(md: str) -> str:
    """Remove a leading/trailing ``` code-fence wrapper the VLM sometimes wraps the
    whole page in — otherwise a blank page comes back as '``` ```' and becomes a junk
    chunk."""
    s = (md or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


import re as _re3

_ATX_HEADING_RE = _re3.compile(r"^(#{1,6})\s+(.+)$")
_SETEXT_RE = _re3.compile(r"^[=-]{3,}$")
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"^\[(image|figure|diagram|photo|illustration|chart|graphic)\s*:",
    re.IGNORECASE,
)


def markdown_to_blocks(md: str, document_id: str, page: int, filename: str) -> list[dict]:
    """
    Split a VLM markdown transcription into structured blocks.

    Supports:
      - Markdown headings (#, ##, ### ...)
      - Setext headings (Heading + ===== / -----)
      - Markdown tables
      - Paragraphs
    """

    lines = _strip_fences(md).splitlines()

    blocks: list[dict] = []
    para: list[str] = []

    def flush():
        if not para:
            return

        text = " ".join(s.strip() for s in para if s.strip()).strip()

        if text and re.search(r"[A-Za-z0-9]", text):
            is_heading = (text.isupper() and len(text) < 100)

            blocks.append(
                _block(document_id,page,filename,"heading" if is_heading else "text",text,))

        para.clear()

    i = 0

    while i < len(lines):

        s = lines[i].strip()

        # ------------------------------------------------------
        # Skip VLM image placeholders
        # ------------------------------------------------------

        if _IMAGE_PLACEHOLDER_RE.match(s):
            flush()
            i += 1
            continue

        # ------------------------------------------------------
        # Blank line
        # ------------------------------------------------------
        if not s:
            flush()
            i += 1
            continue

        # ------------------------------------------------------
        # ATX Markdown heading
        # Example:
        # ## Installation
        # ------------------------------------------------------
        m = _ATX_HEADING_RE.match(s)

        if m:
            flush()

            heading = m.group(2).strip()

            if heading:
                blocks.append(_block( document_id, page, filename, "heading", heading,))

            i += 1
            continue

        # ------------------------------------------------------
        # Setext heading
        #
        # Installation
        # ------------
        # ------------------------------------------------------
        if (i + 1 < len(lines) and _SETEXT_RE.match(lines[i + 1].strip())):
        
            flush()

            blocks.append(_block(document_id, page, filename, "heading",s,))

            i += 2
            continue

        # ------------------------------------------------------
        # Markdown table
        # ------------------------------------------------------
        if s.startswith("|") and "|" in s[1:]:

            flush()
            tbl = []

            while (i < len(lines) and lines[i].strip().startswith("|")):
                tbl.append(lines[i])
                i += 1

            td = _parse_md_table(tbl)
            if td:
                blocks.append( _block(document_id,page,filename,"table","\n".join(t.strip() for t in tbl),td,))

            continue

        # ------------------------------------------------------
        # Normal paragraph
        # ------------------------------------------------------
        para.append(s)
        i += 1

    flush()

    return blocks

def transcribe_page(page, config: dict) -> str:
    cfg = config.get("vision_ocr") or {}
    dpi = int(cfg.get("dpi", 200))
    # Auto-correct a sideways SCAN (rotated invoice / 90-270° page with no /Rotate) to
    # upright before the VLM — better than relying on the model to read rotated text.
    from backend.extraction.orientation import upright_png
    png = upright_png(page, dpi, config)
    # describe_image reads config["vision"]; hand it the vision_ocr provider block.
    # return describe_image(png, prompts.PAGE_TRANSCRIBE, {"vision": cfg})
    raw = describe_image(png, prompts.PAGE_TRANSCRIBE, {"vision": cfg})
    return _unwrap_markdown(raw)

def _unwrap_markdown(raw: str) -> str:
    """The VLM sometimes wraps its page transcription as JSON instead of bare
    markdown — either [{"content": "..."}], {"content": "..."}, or just ["..."]
    (a single string in a list). Unwrap any of these so markdown_to_blocks() sees
    real pipe-table lines and can build table_data — otherwise the whole JSON
    string becomes one text block and structure is lost."""
    from backend.vision.block_builder import _extract_json
    data = _extract_json(raw)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("content") or raw
        if isinstance(first, str) and first.strip():
            return first
    if isinstance(data, dict):
        return data.get("content") or raw
    return raw

def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _yolo_boxes_pts(page, dpi: int = 150, classes=None):
    """DocLayout-YOLO region boxes for this page, converted to PDF points. `classes`
    restricts which layout classes to keep (e.g. {'figure','chart'} to exclude tables).
    [] if the layout model isn't available. Free + CPU — unlike Surya layout (GPU-only)."""
    try:
        from backend.extraction.scanned_pdf.scanned import page_to_pil, _layout_region_boxes
        pil = page_to_pil(page, dpi=dpi)
        boxes_px = _layout_region_boxes(pil, classes=classes)
        if not boxes_px:
            return []
        sx, sy = page.rect.width / pil.width, page.rect.height / pil.height
        return [[x0 * sx, y0 * sy, x1 * sx, y1 * sy] for (x0, y0, x1, y1) in boxes_px]
    except Exception as e:
        logger.warning("vision_ocr: DocLayout-YOLO unavailable (%s)", e)
        return []


# kinds the model flags as page furniture (not real content) — dropped by the gate.
_DROP_KINDS = {"logo", "banner", "header_footer", "decoration", "rule_line",
               "text", "blank"}


def classify_caption_crop(png_bytes: bytes, page_context: str, config: dict) -> dict:
    """SEMANTIC FIGURE GATE: one VLM call both classifies a crop (photo / diagram /
    schematic / circuit / cad_drawing / chart / … vs logo / banner / text / blank) and
    captions it. Returns {keep, kind, caption}. This replaces geometry thresholds for
    deciding what's a real figure — the model can tell a logo from a wide schematic, so
    we no longer wrongly drop large-format diagrams or wrongly keep logos/banners.

    Robust to parse failures: if the model doesn't return clean JSON, we keep the crop
    and use the raw reply as the caption (never silently lose content)."""
    from backend.vision.block_builder import _extract_json
    cfg = config.get("vision_ocr") or {}
    raw = describe_image(png_bytes, prompts.figure_prompt(page_context), {"vision": cfg})
    data = _extract_json(raw)
    if not isinstance(data, dict):
        # Couldn't parse structure — don't lose the figure; keep with the raw caption.
        return {"keep": True, "kind": "unknown", "caption": (raw or "").strip()[:600]}
    kind = str(data.get("kind") or "").strip().lower()
    keep = data.get("keep")
    if keep is None:                       # infer from kind if the model omitted the flag
        keep = kind not in _DROP_KINDS
    keep = bool(keep) and kind not in _DROP_KINDS
    caption = (data.get("caption") or "").strip()
    reasoning = (data.get("reasoning") or "").strip()
    if reasoning:
        logger.info("Figure classification reason: %s", reasoning)
    return {"keep": keep, "kind": kind or "unknown", "caption": caption}


def _paddle_page_text(page) -> str:
    """Free/local OCR fallback for a scanned page (no API). Used past the VLM budget."""
    try:
        from backend.extraction.scanned_pdf import scanned as S
        S.set_ocr_engine("paddle")
        text, _boxes = S.extract_ocr_text_and_boxes(S.page_to_pil(page, dpi=200))
        return text or ""
    except Exception as e:
        logger.warning("rescue: paddle fallback failed (%s)", e)
        return ""


def _page_is_blank(page, thr: float = 0.999) -> bool:
    """True if the rendered page is essentially all-white — a blank inserted page (e.g.
    a manual's inside-cover). Cheap low-dpi grayscale render; avoids spending a VLM
    'scanned rescue' call (and emitting a junk chunk) on a page with nothing on it."""
    try:
        pix = page.get_pixmap(dpi=36, colorspace=fitz.csGRAY)
        data = pix.samples
        if not data:
            return True
        white = sum(1 for v in data if v >= 245)
        return white / len(data) >= thr
    except Exception:
        return False


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def page_text_duplicated(page_blocks: list[dict], min_len: int = 20,
                         min_dupes: int = 2) -> bool:
    """True when a page's extracted text contains repeated blocks — the signature of a
    multi-column layout (e.g. a side-by-side spec sheet) whose reading order Docling
    scrambled into duplicate/fragmented blocks. Such pages are re-transcribed by the
    VLM (one clean pass, correct reading order). Validated on Argo p12 spec sheet."""
    texts = [_norm_text(b.get("text")) for b in page_blocks
             if b.get("type") in ("text", "heading")
             and len((b.get("text") or "").strip()) >= min_len]
    if len(texts) < 3:
        return False
    return (len(texts) - len(set(texts))) >= min_dupes


def _set_route(report: dict | None, pg: int, route: str, reason=None) -> None:
    """Record this page's routing action in the report's per-page ledger (creating the
    record if the extractor didn't already, e.g. a text-only page with no tables/figs)."""
    if report is None:
        return
    per = report.setdefault("per_page", {})
    rec = per.get(str(pg))
    if rec is None:
        rec = {"page": pg, "route": route, "reason": reason,
               "tables": {"tableformer": 0, "vlm": 0},
               "figures": {"proposed": 0, "kept": 0, "dropped": 0, "kinds": []}}
        per[str(pg)] = rec
    rec["route"] = route
    rec["reason"] = reason


def route_and_rescue(blocks: list[dict], pdf_path: str, document_id: str,
                     config: dict, report: dict | None = None,
                     document_type: str = "") -> list[dict]:
    """Per-page router for the hybrid extractor. Docling (run with OCR OFF) gives clean
    text+tables for digital pages and figure boxes for all pages, but NO text for
    scanned pages and possibly-garbled text for bad-font pages. This pass walks EVERY
    pdf page and routes it by MEASURED signals + the document type:
      - diagram / large-format sheet (CAD, circuit, E-size) -> high-DPI tiled transcription
      - scanned (no text layer) or garbled (broken fonts) or duplicated (scrambled
        multi-column) -> VLM page transcription
      - clean digital -> untouched (0 tokens)
    Figure blocks are preserved in every case.

    Budget: extraction.docling.max_vlm_pages caps VLM page-transcriptions per doc
    (0 = unlimited). Past the cap, ANY page needing rescue (scanned, garbled, or
    duplicated) falls back to free local PaddleOCR re-reading the rendered page —
    so a 1000-page scan still finishes without unbounded API cost, and a page with
    broken font encoding or scrambled reading order still gets SOME rescue rather
    than being kept as-is once the budget runs out."""
    from backend.extraction.page_router import profile_page, classify_page, should_tile
    from backend.core.tool import check_cancelled
    cfg = (config.get("extraction") or {}).get("docling") or {}
    lf_enabled = ((config.get("extraction") or {}).get("large_format") or {}).get("enabled", True)
    cap = int(cfg.get("max_vlm_pages", 0) or 0)
    filename = display_filename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("rescue: cannot open %s (%s)", pdf_path, e)
        return blocks

    from backend.storage.postgres_store import PostgresStore
    pg_store = PostgresStore()
    try:
        existing_blocks = pg_store.get_blocks(document_id)
    except Exception:
        existing_blocks = []
    finally:
        pg_store.close()

    existing_by_page = defaultdict(list)
    for b in existing_blocks:
        p = _page_of(b)
        if p is not None:
            existing_by_page[p].append(b)

    def _save_page_blocks_immediate(page_num: int, page_blocks: list[dict]):
        pg_s = PostgresStore()
        try:
            pg_s.write_page_blocks(document_id, page_num, page_blocks)
            logger.info("Saved page %d blocks immediately to database for %s", page_num, document_id)
        except Exception:
            logger.exception("Failed to save page %d blocks to database", page_num)
        finally:
            pg_s.close()

    by_page = defaultdict(list)
    for b in blocks:
        by_page[_page_of(b)].append(b)

    if report is not None:
        report["pages"]["total"] = len(doc)

    out: list[dict] = []
    rescued = paddle_used = digital_kept = 0
    try:
        for pg in range(1, len(doc) + 1):
            check_cancelled(document_id)
            
            # If this page was already extracted successfully in a previous run, use it!
            if pg in existing_by_page:
                logger.info("Using cached blocks for page %d of document %s (resuming ingest)", pg, document_id)
                p_blocks = existing_by_page[pg]
                out.extend(p_blocks)
                
                # Re-tally metrics for skipped page
                tbl_count = sum(1 for nb in p_blocks if nb.get("type") == "table")
                if report is not None:
                    if tbl_count:
                        report["tables"]["total"] += tbl_count
                continue

            pblocks = by_page.get(pg, [])
            native = doc[pg - 1].get_text()
            prof = profile_page(doc[pg - 1])
            is_scanned = len(native.strip()) <= 5 or prof.get("image_dominant", False)
            # A blank inserted page (no text layer + all-white) needs nothing — don't
            # spend a VLM "scanned rescue" call (it would only emit an empty/junk chunk).
            if is_scanned and _page_is_blank(doc[pg - 1]):
                digital_kept += 1
                _set_route(report, pg, "blank")
                out.extend(pblocks)
                _save_page_blocks_immediate(pg, pblocks)
                continue
            # Diagram / large-format sheets (CAD, circuit schematics, oversized engineering
            # drawings) lose all their small text when sent to the VLM whole. Detect them
            # by MEASURED signals (vector density / physical size) + the document type, and
            # transcribe with high-DPI TILING so component/net/title-block detail survives.
            # Only fires when the sheet is actually too big to read whole (should_tile).
            if lf_enabled:
                page_class = classify_page(profile_page(doc[pg - 1]), document_type)
                if page_class in ("diagram", "large_format") and should_tile(doc[pg - 1], page_class, config):
                    try:
                        from backend.extraction.large_format import transcribe_large_page
                        prompt = prompts.SCHEMATIC_TILE if page_class == "diagram" else prompts.PAGE_TRANSCRIBE
                        md = transcribe_large_page(doc[pg - 1], config, prompt)
                        figs = [b for b in pblocks if b.get("type") == "image_caption"]
                        new_blocks = markdown_to_blocks(md, document_id, pg, filename)
                        page_out = new_blocks + figs
                        out.extend(page_out)
                        rescued += 1
                        _set_route(report, pg, "tiled_diagram", page_class)
                        if report is not None:
                            tbl_count = sum(1 for nb in new_blocks if nb.get("type") == "table")
                            if tbl_count:
                                report["tables"]["total"] += tbl_count
                                report["tables"]["vlm_escalated"] += tbl_count
                            report["pages"]["rescued"].append(
                                {"page": pg, "via": "tiled", "reason": page_class})
                        _save_page_blocks_immediate(pg, page_out)
                        continue
                    except Exception as e:
                        logger.warning("rescue: diagram tiling page %s failed (%s); "
                                       "falling through to standard routing", pg, e)
            # The native (pymupdf) text is the canonical signal that a page's fonts are
            # broken: when it's garbled, the page is "hard" and the VLM gives the cleanest
            # result (validated on the Argo electrical spec pages). Docling sometimes
            # decodes such pages into space-joined table text, so checking ITS output
            # would miss them — we trigger off the native layer instead. Duplicated text
            # blocks flag a multi-column page whose reading order Docling scrambled.
            garbled = text_is_unreliable(native)
            duplicated = (not is_scanned and not garbled and page_text_duplicated(pblocks))
            needs = is_scanned or garbled or duplicated
            if not needs:
                digital_kept += 1
                _set_route(report, pg, "digital")
                out.extend(pblocks)
                _save_page_blocks_immediate(pg, pblocks)
                continue
            reason = "scanned" if is_scanned else ("garbled" if garbled else "duplicated")
            figs = [b for b in pblocks if b.get("type") == "image_caption"]
            if cap and rescued >= cap:
                # Past budget: free local re-OCR of the RENDERED page. This isn't
                # limited to "scanned" (no text layer) pages — Paddle reads pixels,
                # so it's just as blind to a broken font/ToUnicode map (garbled) and
                # to Docling's scrambled reading order (duplicated) as it is to a
                # missing text layer. Previously only `is_scanned` got this fallback
                # and garbled/duplicated pages past the cap just kept their broken
                # text with no rescue at all.
                txt = _paddle_page_text(doc[pg - 1])
                if txt:
                    page_out = markdown_to_blocks(txt, document_id, pg, filename) + figs
                    out.extend(page_out)
                    paddle_used += 1
                    _set_route(report, pg, "paddle", reason)
                    if report is not None:
                        report["pages"]["rescued"].append({"page": pg, "via": "paddle", "reason": reason})
                    _save_page_blocks_immediate(pg, page_out)
                else:
                    _set_route(report, pg, "kept_original", reason)
                    out.extend(pblocks)
                    _save_page_blocks_immediate(pg, pblocks)
                continue
            try:
                md = transcribe_page(doc[pg - 1], config)
                new_blocks = markdown_to_blocks(md, document_id, pg, filename)
                page_out = new_blocks + figs
                out.extend(page_out)
                rescued += 1
                _set_route(report, pg, "vlm_rescue", reason)
                if report is not None:
                    tbl_count = sum(1 for nb in new_blocks if nb.get("type") == "table")
                    if tbl_count:
                        report["tables"]["total"] += tbl_count
                        report["tables"]["vlm_escalated"] += tbl_count
                    report["pages"]["rescued"].append({"page": pg, "via": "vlm", "reason": reason})
                _save_page_blocks_immediate(pg, page_out)
            except Exception as e:
                logger.warning("rescue: page %s VLM failed (%s); keeping originals", pg, e)
                _set_route(report, pg, "vlm_failed", reason)
                out.extend(pblocks)
                _save_page_blocks_immediate(pg, pblocks)
    finally:
        doc.close()
    if report is not None:
        report["pages"]["digital_kept"] = digital_kept
        report["pages"]["vlm_rescued"] = rescued
        report["pages"]["paddle_fallback"] = paddle_used
    if rescued or paddle_used:
        logger.info("rescue: %d page(s) via VLM, %d via Paddle fallback (cap=%s)",
                    rescued, paddle_used, cap or "∞")
    return out