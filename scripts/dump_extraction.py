"""Dump what an extractor produces for a PDF to a readable text file, so we can eyeball
the actual content (text + STRUCTURED tables + image placeholders) per page.

    python scripts/dump_extraction.py digital  <pdf> <out.txt>
    python scripts/dump_extraction.py scanned  <pdf> <out.txt>   # uses ocr.engine via env
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _table_md(td: dict) -> str:
    headers = td.get("headers") or []
    rows = td.get("rows") or []
    out = []
    if headers:
        out.append("| " + " | ".join(map(str, headers)) + " |")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def run(kind: str, pdf_path: str, out_path: str) -> None:
    doc_id = "dump-doc"
    if kind == "digital":
        from backend.extraction.digital_pdf.digital import extract_digital
        blocks = extract_digital(pdf_path, doc_id)
    elif kind == "scanned":
        from backend.extraction.scanned_pdf.scanned import extract_scanned
        blocks = extract_scanned(pdf_path, doc_id)
    else:
        raise SystemExit("kind must be 'digital' or 'scanned'")

    # blocks may be dataclasses or dicts — normalize access
    def g(b, k, default=None):
        return getattr(b, k, None) if hasattr(b, k) else b.get(k, default)

    counts: dict[str, int] = {}
    lines = [f"# Extraction dump ({kind})", f"# file: {pdf_path}",
             f"# blocks: {len(blocks)}", ""]
    for b in blocks:
        counts[g(b, "type")] = counts.get(g(b, "type"), 0) + 1

    lines.insert(3, "# by type: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append("")

    for i, b in enumerate(blocks):
        btype = g(b, "type")
        ref = g(b, "source_ref") or {}
        page = getattr(ref, "page", None) if hasattr(ref, "page") else (ref.get("page") if isinstance(ref, dict) else None)
        lines.append(f"───── [{i}] type={btype}  page={page} " + "─" * 30)
        td = g(b, "table_data")
        if btype == "table" and td:
            lines.append(_table_md(td))
        else:
            lines.append((g(b, "text") or "").rstrip())
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path} — {len(blocks)} blocks; by type: {counts}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
