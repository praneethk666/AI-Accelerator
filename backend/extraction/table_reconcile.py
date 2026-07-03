"""Document-level structure reconciliation — cross-page table stitching.

Per-page extraction (Docling, VLM, any of them) sees each page in isolation, so a
table that spans pages becomes several blocks: the lead has the header, each
continuation is a headerless orphan on the next page. That wrecks chunking and
retrieval. This pass runs AFTER extraction and BEFORE chunking: it detects
continuation tables and merges them back into one, inheriting the lead's header.

Continuation test (all must hold) — strong + free, no LLM:
  - the two tables are on consecutive pages,
  - same column count,
  - the lead table ends near the bottom of its page,
  - the continuation starts near the top of its page.
Chains across 3+ pages (a long table) by letting the lead keep absorbing.

Operates on the shared block list (engine-agnostic), so any extractor benefits.
Gate with config extraction.stitch_tables (default on).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import fitz

logger = logging.getLogger(__name__)


def _cell_type(s: str) -> str:
    """Coarse data type of a cell — the per-column fingerprint primitive."""
    s = (s or "").strip()
    if not s:
        return "empty"
    if re.search(r"\d{1,4}[/-]\d{1,4}", s):
        return "date"
    if re.fullmatch(r"[\d.,$%/+\-\s]+", s):
        return "num"
    if len(s) <= 4 and re.search(r"\d", s):
        return "code"
    return "text"


def _col_types(rows: list[list], ncols: int) -> list[str]:
    cols: list[list] = [[] for _ in range(ncols)]
    for r in rows:
        for i in range(min(ncols, len(r))):
            cols[i].append(_cell_type(r[i]))
    return [Counter(c).most_common(1)[0][0] if c else "empty" for c in cols]


def _fingerprint_decision(lead_td: dict, cur_td: dict) -> str:
    """Compare per-column data types of the lead vs candidate continuation.
    Returns 'merge' (informative columns agree), 'separate' (they clash — a DIFFERENT
    table that merely shares the column count), or 'ambiguous' (all-text columns, can't
    tell). This is the free discriminator for the false-merge case."""
    lr = (lead_td.get("rows") or []) if isinstance(lead_td, dict) else []
    cr = (cur_td.get("rows") or []) if isinstance(cur_td, dict) else []
    n = max(_cols(lead_td), _cols(cur_td))
    if not lr or not cr or n == 0:
        return "ambiguous"
    a, b = _col_types(lr, n), _col_types(cr, n)
    informative = {"num", "date"}
    conflicts = comparable = 0
    for x, y in zip(a, b):
        if x in informative or y in informative:
            comparable += 1
            if x != y:
                conflicts += 1
    if comparable == 0:
        return "ambiguous"
    if conflicts == 0:
        return "merge"
    return "separate" if conflicts / comparable > 0.3 else "ambiguous"


def _llm_is_continuation(lead: dict, cur: dict, config: dict) -> bool:
    """Ask the LLM whether `cur` is a continuation of `lead` — only for genuinely
    ambiguous boundaries (all-text columns). Cheap: one tiny call, header + a few rows."""
    try:
        from backend.core.llm_client import get_llm

        def head(td):
            if not isinstance(td, dict):
                return ""
            hs = " | ".join(map(str, td.get("headers") or []))
            rows = td.get("rows") or []
            sample = " || ".join(" | ".join(map(str, r)) for r in rows[:2])
            return f"headers: {hs}\nrows: {sample}"

        prompt = (
            "Two tables appear on consecutive PDF pages with the same number of columns. "
            "Is table B a CONTINUATION of table A (same table split across pages), or a "
            "SEPARATE table? Answer with one word: CONTINUATION or SEPARATE.\n\n"
            f"TABLE A:\n{head(lead.get('table_data'))}\n\nTABLE B:\n{head(cur.get('table_data'))}"
        )
        llm = get_llm(config)
        reply = llm.invoke(prompt)
        ans = (getattr(reply, "content", str(reply)) or "").strip().upper()
        return "CONTINUATION" in ans and "SEPARATE" not in ans
    except Exception as e:
        logger.warning("stitch: LLM arbitration failed (%s); not merging", e)
        return False


def _page_heights(pdf_path: str) -> dict:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("stitch: cannot open %s (%s)", pdf_path, e)
        return {}
    try:
        return {i + 1: float(doc[i].rect.height) for i in range(len(doc))}
    finally:
        doc.close()


def _cols(td: dict | None) -> int:
    if isinstance(td, dict) and isinstance(td.get("headers"), list):
        return len(td["headers"])
    return 0


def _bbox(block: dict):
    ref = block.get("source_ref") or {}
    bb = ref.get("bbox") if isinstance(ref, dict) else None
    return bb if (isinstance(bb, (list, tuple)) and len(bb) == 4) else None


def _page(block: dict):
    ref = block.get("source_ref") or {}
    return ref.get("page") if isinstance(ref, dict) else None


def _rebuild_markdown(headers: list, rows: list[list]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _is_continuation(lead, lead_pg, lead_bb, cur, cur_pg, cur_bb, heights,
                     bottom_frac=0.70, top_frac=0.30) -> bool:
    if not (isinstance(lead_pg, int) and isinstance(cur_pg, int) and cur_pg == lead_pg + 1):
        return False
    lc, cc = _cols(lead.get("table_data")), _cols(cur.get("table_data"))
    if lc == 0 or cc == 0 or lc != cc:
        return False
    if not (lead_bb and cur_bb):
        return False
    hl = heights.get(lead_pg) or 792.0
    hc = heights.get(cur_pg) or 792.0
    return lead_bb[3] >= bottom_frac * hl and cur_bb[1] <= top_frac * hc


def _is_placeholder_headers(headers: list) -> bool:
    """True if headers are just positional placeholders (0,1,2,…) — TableFormer's
    default when the continuation page had no header row. Those are NOT data."""
    return list(map(str, headers)) == [str(i) for i in range(len(headers))]


def _merge(lead: dict, cur: dict) -> None:
    """Absorb `cur` into `lead`. If the continuation page had no real header,
    TableFormer labels its columns 0,1,2,… (placeholders) and puts the actual cells
    in rows — so we drop the placeholder header. If the continuation DID carry a
    repeated header row, that header is really data and we keep it as a row."""
    ltd, ctd = lead.get("table_data"), cur.get("table_data")
    if isinstance(ltd, dict) and isinstance(ctd, dict):
        lead_hdr_norm = [str(c).strip().lower() for c in (ltd.get("headers") or [])]

        def _is_repeat_header(row) -> bool:
            # The continuation page often REPEATS the column header (manuals reprint it
            # on each page). That repeated header is furniture, NOT data — drop it so it
            # doesn't land mid-table as a bogus row (validated: Argo p17 troubleshooting).
            return bool(lead_hdr_norm) and [str(c).strip().lower() for c in row] == lead_hdr_norm

        rows = list(ltd.get("rows") or [])
        ch = ctd.get("headers")
        if (isinstance(ch, list) and any(str(c).strip() for c in ch)
                and not _is_placeholder_headers(ch)
                and not _is_repeat_header(ch)):         # a NON-repeated header row = data
            rows.append([str(c) for c in ch])
        rows.extend(r for r in (ctd.get("rows") or []) if not _is_repeat_header(r))
        ltd["rows"] = rows
        lead["table_data"] = ltd
        lead["text"] = _rebuild_markdown(ltd.get("headers") or [], rows)
    else:
        # markdown-only tables (e.g. VLM source): append cur's body lines
        cur_lines = [ln for ln in (cur.get("text") or "").splitlines()
                     if ln.strip().startswith("|")
                     and not set(ln.replace("|", "").strip()) <= set("-: ")]
        lead["text"] = (lead.get("text") or "") + "\n" + "\n".join(cur_lines)
    meta = lead.setdefault("metadata", {})
    meta["stitched_pages"] = sorted(set(meta.get("stitched_pages", [_page(lead)]) + [_page(cur)]))


def reconcile_tables(blocks: list[dict], pdf_path: str, config: dict | None = None,
                     report: dict | None = None) -> list[dict]:
    """Merge cross-page continuation tables in `blocks`. Returns the new block list.

    Decision per candidate boundary: geometry+column-count gate (cheap) -> per-column
    data fingerprint (cheap) -> LLM arbitration for the genuinely ambiguous case
    (all-text columns), which avoids false-merging a DIFFERENT table that merely shares
    the column count. Arbitration is gated by extraction.stitch_arbitration (default on);
    when off, ambiguous boundaries are left UNMERGED (favor precision)."""
    cfg = config or {}
    arbitrate = ((cfg.get("extraction") or {}).get("stitch_arbitration", True))
    heights = _page_heights(pdf_path)
    table_idx = [i for i, b in enumerate(blocks) if b.get("type") == "table"]
    if len(table_idx) < 2:
        return blocks

    drop: set[int] = set()
    arb_calls = 0
    lead_i = table_idx[0]
    lead_pg, lead_bb = _page(blocks[lead_i]), _bbox(blocks[lead_i])
    for i in table_idx[1:]:
        cur, cur_pg, cur_bb = blocks[i], _page(blocks[i]), _bbox(blocks[i])
        merge = False
        if _is_continuation(blocks[lead_i], lead_pg, lead_bb, cur, cur_pg, cur_bb, heights):
            decision = _fingerprint_decision(blocks[lead_i].get("table_data"),
                                             cur.get("table_data"))
            if decision == "merge":
                merge = True
            elif decision == "ambiguous" and arbitrate:
                arb_calls += 1
                merge = _llm_is_continuation(blocks[lead_i], cur, cfg)
            # 'separate', or ambiguous-without-arbitration -> don't merge
        if merge:
            _merge(blocks[lead_i], cur)
            drop.add(i)
            lead_pg, lead_bb = cur_pg, cur_bb     # chain to the next page if it continues
        else:
            lead_i, lead_pg, lead_bb = i, cur_pg, cur_bb

    if report is not None:
        report["stitch"]["merged"] = len(drop)
        report["stitch"]["arbitrations"] = arb_calls
    if drop or arb_calls:
        logger.info("stitch: merged %d continuation table(s); %d LLM arbitration(s)",
                    len(drop), arb_calls)
    return [b for j, b in enumerate(blocks) if j not in drop]
