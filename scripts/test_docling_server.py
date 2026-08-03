#!/usr/bin/env python3
"""
scripts/test_docling_server.py
A test script to verify connection to and functionality of the remote Docling Extraction Server.
"""

import os
import sys
import uuid
import argparse
import requests
import json
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def run_health_check(server_url: str) -> bool:
    print(f"Checking health status at {server_url}/health...", flush=True)
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print("\nHealth Status:")
        print(json.dumps(data, indent=2))
        return True
    except Exception as e:
        print(f"\n[ERROR] Health check failed: {e}", flush=True)
        return False

def extract_pdf(pdf_path: str, server_url: str, api_key: str, 
                table_source: str, do_ocr: bool, do_table_structure: bool,
                min_picture_pts: float, document_id: str) -> None:
    filename = os.path.basename(pdf_path)
    print(f"\nSending extraction request for {filename}...", flush=True)
    print(f"Server URL:        {server_url}")
    print(f"Document ID:       {document_id}")
    print(f"Table Source:      {table_source}")
    print(f"Do OCR:            {do_ocr}")
    print(f"Do Table Structure: {do_table_structure}")
    print(f"Min Picture Pts:   {min_picture_pts}")
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        print(f"API Key:           {'*' * 8}{api_key[-4:] if len(api_key) > 4 else ''}")
    else:
        print("API Key:           NOT SET (no auth header will be sent)")

    data = {
        "document_id": document_id,
        "filename": filename,
        "table_source": table_source,
        "do_ocr": str(do_ocr).lower(),
        "do_table_structure": str(do_table_structure).lower(),
        "min_picture_pts": str(min_picture_pts),
    }

    try:
        with open(pdf_path, "rb") as f:
            files = {"pdf": (filename, f, "application/pdf")}
            resp = requests.post(
                f"{server_url}/extract",
                headers=headers,
                files=files,
                data=data,
                timeout=600,
            )
        resp.raise_for_status()
        payload = resp.json()
        
        # Save raw JSON to output folder
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(output_dir, exist_ok=True)
        pdf_basename = os.path.splitext(filename)[0]
        json_path = os.path.join(output_dir, f"{pdf_basename}_blocks.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2)
        print(f"\n[INFO] Saved raw JSON response to: {json_path}")
    except requests.exceptions.HTTPError as he:
        print(f"\n[HTTP ERROR] Server returned status code {he.response.status_code}", flush=True)
        print(f"Response: {he.response.text}", flush=True)
        return
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}", flush=True)
        return

    blocks = payload.get("blocks", [])
    n_pages = payload.get("n_pages", 0)
    elapsed_s = payload.get("elapsed_s", 0.0)

    print(f"\n[SUCCESS] Extraction completed successfully!", flush=True)
    print(f"Pages:             {n_pages}")
    print(f"Blocks extracted:  {len(blocks)}")
    print(f"Server elapsed time: {elapsed_s:.1f}s")

    # Group by type
    counts = Counter(b.get("type", "unknown") for b in blocks)
    print("\nBlock count by type:")
    for btype, count in counts.items():
        print(f"  - {btype}: {count}")

    # Display some blocks
    print("\n--- SAMPLE BLOCKS (First 10) ---")
    for i, b in enumerate(blocks[:10], 1):
        btype = b.get("type")
        ref = b.get("source_ref") or {}
        page = ref.get("page", "?")
        text_snippet = (b.get("text") or "").replace("\n", " ")
        if len(text_snippet) > 80:
            text_snippet = text_snippet[:80] + "..."
        print(f"[{i:02d}] Type: {btype:<15} Page: {page:<3} Text: {text_snippet}", flush=True)
        
        if btype == "table":
            td = b.get("table_data")
            if td and isinstance(td, dict):
                headers = td.get("headers") or []
                rows = td.get("rows") or []
                print(f"     -> Table dimensions: {len(headers)} columns x {len(rows)} rows", flush=True)
                if headers:
                    print(f"     -> Headers: {headers}", flush=True)
                if rows:
                    print(f"     -> First row: {rows[0]}", flush=True)
            esc_hint = b.get("metadata", {}).get("escalation_hint")
            if esc_hint:
                print(f"     -> Escalation Hint: {esc_hint}", flush=True)

        elif btype == "image_caption":
            bbox = ref.get("bbox")
            print(f"     -> Bounding box: {bbox}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script to call remote Docling server.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to test.")
    parser.add_argument("--server-url", default=os.environ.get("DOCLING_SERVER_URL", "http://localhost:8083"),
                        help="Docling server base URL.")
    parser.add_argument("--api-key", default=os.environ.get("DOCLING_API_KEY", ""),
                        help="Docling API Key.")
    parser.add_argument("--table-source", choices=["auto", "docling", "pymupdf"], default="auto",
                        help="Table source option for extraction.")
    parser.add_argument("--do-ocr", action="store_true", help="Enable OCR for scanned pages.")
    parser.add_argument("--do-table-structure", action="store_true", default=True, help="Recover table structure (default: True).")
    parser.add_argument("--no-table-structure", dest="do_table_structure", action="store_false", help="Disable table structure recovery.")
    parser.add_argument("--min-picture-pts", type=float, default=24.0, help="Minimum picture size in points.")
    parser.add_argument("--document-id", default="", help="Custom document ID (UUID generated if omitted).")
    parser.add_argument("--health", action="store_true", help="Only run health check and exit.")

    args = parser.parse_args()

    # If --health is set, run only the health check
    if args.health:
        run_health_check(args.server_url.rstrip("/"))
        sys.exit(0)

    # Health check before running extraction to verify connection
    server_url_clean = args.server_url.rstrip("/")
    if not run_health_check(server_url_clean):
        print("\n[WARNING] Health check failed or timed out. Attempting extraction anyway...", flush=True)

    if not args.pdf_path:
        # Try to find a default PDF in uploads
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            pdfs = [f for f in os.listdir(uploads_dir) if f.endswith(".pdf")]
            if pdfs:
                args.pdf_path = os.path.join(uploads_dir, pdfs[0])
                print(f"\nNo PDF specified. Automatically selected first PDF found in '{uploads_dir}': {args.pdf_path}", flush=True)
            else:
                print("\n[ERROR] No PDF file specified and no PDF files found in 'uploads/' directory.", flush=True)
                print("Usage: python scripts/test_docling_server.py <path_to_pdf>", flush=True)
                sys.exit(1)
        else:
            print("\n[ERROR] No PDF file specified.", flush=True)
            print("Usage: python scripts/test_docling_server.py <path_to_pdf>", flush=True)
            sys.exit(1)

    if not os.path.exists(args.pdf_path):
        print(f"\n[ERROR] File not found: {args.pdf_path}", flush=True)
        sys.exit(1)

    doc_id = args.document_id or f"test-doc-{uuid.uuid4().hex[:8]}"
    
    extract_pdf(
        pdf_path=args.pdf_path,
        server_url=server_url_clean,
        api_key=args.api_key,
        table_source=args.table_source,
        do_ocr=args.do_ocr,
        do_table_structure=args.do_table_structure,
        min_picture_pts=args.min_picture_pts,
        document_id=doc_id
    )
