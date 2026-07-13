"""Head-to-head: Docling TableFormer vs VLM (GLM) on the SAME table region.

Usage:
    python scripts/table_compare.py <pdf> [page]

For each table Docling finds (optionally only on `page`), prints:
  (A) Docling TableFormer -> markdown
  (B) GLM transcription of the cropped table region -> markdown
so you can eyeball which is correct on YOUR complex tables before we wire
table_source: docling | vlm. The VLM call uses config['vision_ocr'] (GLM, free).
"""
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()                      # load ZAI_API_KEY etc. like the backend does
except Exception:
    pass

from backend.core.config import load_config
from backend.vision.pdf_cropper import PDFCropper
from backend.core.vision_client import describe_image
from backend.extraction.docling_pdf.docling_extract import (
    _converter, _prov, _page_height, _bbox_topleft_pts, _table_markdown,
)

_VLM_TABLE_PROMPT = (
    "Transcribe ONLY the table in this image as a GitHub markdown table. Preserve every "
    "row and column, keep merged/spanning headers as repeated cells, and copy all "
    "numbers, units and codes VERBATIM. Output only the markdown table, nothing else."
)


def main():
    pdf = sys.argv[1]
    only_page = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = load_config("config/global.yaml")
    dcfg = (cfg.get("extraction") or {}).get("docling") or {}

    conv = _converter({**dcfg, "do_ocr": False, "do_table_structure": True})
    doc = conv.convert(pdf).document
    cropper = PDFCropper()
    vcfg = {"vision": cfg.get("vision_ocr")}

    n = 0
    for tb in doc.tables:
        page_no, bbox = _prov(tb)
        if only_page and page_no != only_page:
            continue
        n += 1
        bb = _bbox_topleft_pts(bbox, _page_height(doc, page_no))
        print("\n" + "=" * 78)
        print(f"TABLE #{n}  page {page_no}  bbox {[round(v, 1) for v in bb] if bb else None}")
        print("-" * 78 + "\n[A] DOCLING TableFormer:\n")
        print(_table_markdown(tb, doc) or "(empty)")
        if bb:
            png = cropper.crop_region(pdf, page_no, bb)
            print("\n[B] GLM region transcription:\n")
            try:
                print(describe_image(png, _VLM_TABLE_PROMPT, vcfg).strip())
            except Exception as e:
                print(f"(VLM failed: {e})")
    if not n:
        print("No tables found", "on page %s" % only_page if only_page else "")


if __name__ == "__main__":
    main()
