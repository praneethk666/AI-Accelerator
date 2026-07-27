"""Self-hosted Unlimited-OCR (baidu/Unlimited-OCR) as an alternative rescue engine
for route_and_rescue's per-page transcription — a swap-in for the hosted VLM call in
`transcribe_page()`, not a replacement for Docling. Docling still does the free,
fast structural pass on every page; this only fires for pages route_and_rescue has
already flagged as needing rescue (scanned/garbled/duplicated).

Why: live bake-off on this project's real 105-page servo manual (24-Jul) found the
hosted z.ai VLM rescue path bottlenecked on account concurrency (1-2 concurrent
calls, real production 429s). Unlimited-OCR's single-page `infer()` mode
(crop_mode=True), run locally on a GPU box, was validated on 7 diverse real pages:
zero repetition, ~26-30s/page steady-state, and correctly-structured HTML tables
with rowspan for exactly the kind of alarm-code tables this project's table-repair
machinery (chunk_tool.py) exists to fix. `infer_multi()` (the "one-shot many pages"
mode) was tested too and REJECTED — it degenerates into repeating the same line on
dense/repetitive real content (validated on servo manual pages 50-90); this module
only ever calls the model once per page.

This module does NOT run the model in-process — that needs a CUDA GPU with the
model loaded (~7.3GB BF16), which the main app server doesn't have. It's an HTTP
client for a small inference server meant to run on a GPU box (see
scripts/ocr_server.py) and is only used when
config["vision_ocr"]["engine"] == "local".

Known open gap: the model's own <|det|> bounding boxes are in an unverified
model-internal coordinate space (looks like a resize-longest-side-to-~1024
normalization, but this hasn't been calibrated against real PDF-point space the way
Docling/YOLO boxes are). Trusting an unverified scale for the icon-size-threshold
exclusion logic in chunk_tool.py, or for view_page_image crops, would silently
misclassify content. So the raw det-space box is kept in metadata for future
calibration, but source_ref.bbox is left None here — identical to what the existing
VLM markdown rescue path already does (vision_ocr._block()), so nothing regresses.
"""
from __future__ import annotations

import logging
import re
import uuid
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

# `<|det|>TYPE [x0, y0, x1, y1]<|/det|>CONTENT` — CONTENT runs to the next tag (or EOS).
_DET_RE = re.compile(
    r"<\|det\|>\s*(?P<type>\w+)\s*\[(?P<box>[^\]]*)\]\s*<\|/det\|>",
)

# Page furniture the model tags explicitly — never worth indexing (matches how the
# existing VLM markdown rescue path implicitly drops repeated headers/footers, and
# how image_caption logo/banner crops are dropped in chunk_tool.py).
_SKIP_TYPES = {"header", "page_number", "image"}
_HEADING_TYPES = {"title"}
_TABLE_TYPES = {"table"}


class _RowspanTableParser(HTMLParser):
    """Parses a <table> with rowspan/colspan into a DENSE per-row grid — every
    spanning cell's value is repeated into every row/column it visually covers,
    so each output row is self-contained (no blank cells relying on an earlier
    row for context). This is the same self-containment property
    _row_traceability_error() in chunk_tool.py already assumes for repaired
    tables, so tables that arrive this way should need little/no LLM repair."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._pending: dict[int, list[tuple[int, str]]] = {}  # col -> [(rows_left, text)]
        self._row: list[str] | None = None
        self._col = 0
        self._cell_text: list[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
            self._col = 0
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = []
            self._cell_rowspan = int(a.get("rowspan", 1) or 1)
            self._cell_colspan = int(a.get("colspan", 1) or 1)
        elif tag == "br" and self._in_cell:
            self._cell_text.append("; ")

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._skip_filled_cols()
            text = re.sub(r"\s+", " ", "".join(self._cell_text)).strip()
            for i in range(self._cell_colspan):
                self._set_cell(self._col + i, text)
                if self._cell_rowspan > 1:
                    self._pending.setdefault(self._col + i, []).append(
                        (self._cell_rowspan - 1, text))
            self._col += self._cell_colspan
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self._flush_trailing_pending()
            if any(c.strip() for c in self._row):
                self.rows.append(self._row)
            self._row = None

    def _skip_filled_cols(self):
        while self._col in self._pending and self._pending[self._col]:
            self._set_cell(self._col, None)  # placeholder consumed below
            self._col += 1

    def _flush_trailing_pending(self):
        """A continuation row with FEWER real <td>s than there are actively-
        spanning columns (e.g. only a 'details' cell, while 'Indication' and
        'Remarks' are ALSO under rowspan from the header row above) leaves those
        trailing columns' carry-down un-consumed by _skip_filled_cols, which only
        ever advances far enough to place the row's own new <td>s. Left alone, that
        stale (rows_left, value) entry survives into the NEXT row untouched by this
        row's pass at all -- and once a later, unrelated group's own column lands
        on that same index, it gets fed the wrong group's leftover value instead of
        its own. Real bug found live, 27-Jul, on the servo manual's F7H-FFH alarm
        table: a later alarm group's Remarks cell also carried a genuine rowspan,
        and inherited an earlier group's stale Indication/Remarks carry-down,
        corrupting both from that point on. Fix: after all of THIS row's real
        <td>s are placed, walk every remaining column index that has ANY pending
        state at all (active or not) and consume/decrement the active ones, so
        pending state always reflects exactly how many physical rows have passed
        -- regardless of whether this particular row had a new <td> reach there."""
        if not self._pending:
            return
        last_col = max(self._pending.keys())
        while self._col <= last_col:
            if self._pending.get(self._col):
                self._set_cell(self._col, None)
            self._col += 1

    def _set_cell(self, col, text):
        if self._row is None:
            return
        while len(self._row) <= col:
            self._row.append("")
        if text is None:
            # consume one pending carry-down entry for this column
            entries = self._pending.get(col)
            if entries:
                left, val = entries[0]
                self._row[col] = val
                if left > 1:
                    entries[0] = (left - 1, val)
                else:
                    entries.pop(0)
        else:
            self._row[col] = text


def _parse_html_table(html: str) -> dict | None:
    p = _RowspanTableParser()
    try:
        p.feed(html)
    except Exception as e:
        logger.warning("unlimited_ocr: table parse failed (%s)", e)
        return None
    if not p.rows:
        return None
    width = max(len(r) for r in p.rows)
    rows = [r + [""] * (width - len(r)) for r in p.rows]
    return {"headers": rows[0], "rows": rows[1:]}


def _block(document_id, page, filename, btype, text, table_data=None, det_bbox=None) -> dict:
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
        "metadata": {"source": "unlimited_ocr", "det_bbox_raw": det_bbox},
    }


def det_to_blocks(raw: str, document_id: str, page: int, filename: str) -> list[dict]:
    """Unlimited-OCR's `<|det|>type [x0,y0,x1,y1]<|/det|>content` transcription ->
    our block schema. Mirrors vision_ocr.markdown_to_blocks() for the VLM path."""
    matches = list(_DET_RE.finditer(raw or ""))
    blocks: list[dict] = []
    for i, m in enumerate(matches):
        btype = m.group("type").strip().lower()
        box_str = m.group("box")
        try:
            det_bbox = [float(x) for x in box_str.split(",")] if box_str.strip() else None
        except ValueError:
            det_bbox = None
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()

        if btype in _SKIP_TYPES:
            continue
        if not content:
            continue

        if btype in _TABLE_TYPES:
            table_data = _parse_html_table(content)
            if table_data is None:
                continue  # unparseable table markup -> drop rather than emit raw HTML
            blocks.append(_block(document_id, page, filename, "table",
                                  content, table_data, det_bbox))
            continue

        out_type = "heading" if btype in _HEADING_TYPES else "text"
        blocks.append(_block(document_id, page, filename, out_type, content, None, det_bbox))

    return blocks


def _call_ocr_server(png_bytes: bytes, cfg: dict, *, engine: str = "unlimited_ocr",
                      extra_form: dict | None = None) -> str:
    """POST an already-rendered PNG to scripts/ocr_server.py and return its raw text.
    `cfg` must carry local_endpoint/local_api_key/local_timeout_s — both callers below
    (whole-page rescue, table-crop escalation) point at the SAME physical server, so
    they share this one connection block (vision_ocr.local_*) rather than duplicating
    endpoint/key config per call site."""
    endpoint = cfg.get("local_endpoint")
    if not endpoint:
        raise ValueError(
            "engine is 'local' but local_endpoint is not set "
            "(point it at scripts/ocr_server.py running on the GPU box)")
    timeout = float(cfg.get("local_timeout_s", 90))
    headers = {}
    api_key = cfg.get("local_api_key")
    if api_key:
        headers["X-API-Key"] = api_key
    data = {"engine": engine, **(extra_form or {})}
    resp = httpx.post(endpoint, files={"image": ("page.png", png_bytes, "image/png")},
                       data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["text"]


def transcribe_page_local(page, config: dict) -> str:
    """Render `page` and send it to the self-hosted Unlimited-OCR inference server.
    Returns the raw <|det|>-tagged transcription (feed to det_to_blocks)."""
    from backend.extraction.orientation import upright_png
    cfg = config.get("vision_ocr") or {}
    dpi = int(cfg.get("dpi", 200))
    png = upright_png(page, dpi, config)
    return _call_ocr_server(png, cfg, engine="unlimited_ocr")


def transcribe_table_local(png_bytes: bytes, config: dict) -> dict | None:
    """Send an ALREADY-CROPPED table-region PNG (from PDFCropper) to the self-hosted
    Unlimited-OCR server and return {headers, rows} directly — no markdown round trip,
    so the rowspan-denormalized structure (backend/extraction/unlimited_ocr.py's whole
    point) survives intact. Swap-in for docling_extract.py's _vlm_table() when
    extraction.docling.table_engine == 'local'. Uses the SAME connection config as the
    whole-page rescue path (vision_ocr.local_endpoint/local_api_key) since both hit the
    same physical server.

    On a CUDA-OOM-shaped failure (real finding, 27-Jul: some large/dense table crops
    need more peak GPU memory than crop_mode=True's patch-tiling path has free),
    retries ONCE with crop_mode=False -- a different, less memory-hungry codepath --
    before letting the caller fall back to the paid VLM call. ocr_server.py returns a
    distinguishable 503 for this specific case (a plain 500 could be anything)."""
    cfg = config.get("vision_ocr") or {}
    try:
        raw = _call_ocr_server(png_bytes, cfg, engine="unlimited_ocr")
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 503:
            raise
        logger.warning("unlimited_ocr: CUDA OOM on table crop, retrying with crop_mode=False")
        raw = _call_ocr_server(png_bytes, cfg, engine="unlimited_ocr",
                               extra_form={"crop_mode": False})
    matches = list(_DET_RE.finditer(raw))
    for idx, m in enumerate(matches):
        if m.group("type").strip().lower() != "table":
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        return _parse_html_table(raw[start:end].strip())
    return None
