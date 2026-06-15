"""Full pipeline test on Scanned_22pages.pdf with semantic chunking (chonkie)
using configuration from config/global.yaml.
Prints per‑page embedding time (approximated) and total extraction time.
"""

import sys
import os
import json
import time
import yaml
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.extraction.mixed_pdf.tool import MixedPDFTool
from backend.chunking.chunk_tool import ChunkTool
from backend.embeddings.embed_tool import EmbedTool

def load_config(config_path: str = "config/global.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    pdf_path = "test-data/Scanned_22pages.pdf"
    pdf_name = "12new_Scanned_22pages"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ---------- 1. Extraction ----------
    print("1. Extracting...")
    start_extract = time.time()
    tool = MixedPDFTool()
    state = {"file_path": pdf_path, "document_id": pdf_name}
    state = tool.run(state, config)
    extract_time = time.time() - start_extract
    print(f"   Total extraction time: {extract_time:.2f}s (all pages)")

    blocks_dict = [asdict(b) for b in state["blocks"]]
    profiles_dict = [asdict(p) for p in state["page_profiles"]]
    num_pages = len(profiles_dict)
    print(f"   Number of pages: {num_pages}")
    print(f"   Average extraction time per page (approx): {extract_time/num_pages:.2f}s")

    with open(output_dir / f"{pdf_name}_blocks.json", "w", encoding="utf-8") as f:
        json.dump(blocks_dict, f, indent=2, default=str)
    with open(output_dir / f"{pdf_name}_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles_dict, f, indent=2, default=str)
    print("  Saved blocks and profiles.")

    # Convert blocks to dicts for chunk_tool
    state["blocks"] = blocks_dict

    # ---------- 2. Chunking ----------
    print("2. Chunking...")
    start_chunk = time.time()
    chunker = ChunkTool()
    state = chunker.run(state, config)
    chunk_time = time.time() - start_chunk
    print(f"   Chunking completed in {chunk_time:.2f}s")

    num_chunks = len(state["chunks"])
    print(f"   Created {num_chunks} chunks")

    # Count chunks per page (from source_ref.page)
    chunks_per_page = {}
    for ch in state["chunks"]:
        page = ch["source_ref"].get("page", 0)
        chunks_per_page[page] = chunks_per_page.get(page, 0) + 1
    print("   Chunks per page:", chunks_per_page)

    with open(output_dir / f"{pdf_name}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f, indent=2, default=str)
    print("  Saved chunks (before embedding).")

    # ---------- 3. Embedding ----------
    dense_model = config["embeddings"]["dense_model"]
    print(f"3. Embedding (using {dense_model})...")
    start_embed = time.time()
    embedder = EmbedTool()
    state = embedder.run(state, config)
    embed_time = time.time() - start_embed
    print(f"   Total embedding time: {embed_time:.2f}s")

    avg_per_chunk = embed_time / num_chunks if num_chunks else 0
    print(f"   Average embedding time per chunk: {avg_per_chunk:.3f}s")

    # Per‑page embedding time (approximated by chunk count)
    print("   Approximate embedding time per page (based on chunk count):")
    for page, cnt in chunks_per_page.items():
        approx = cnt * avg_per_chunk
        print(f"     Page {page:2d}: {approx:6.3f}s ({cnt:2d} chunks)")

    with open(output_dir / f"{pdf_name}_chunks_embedded.json", "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f, indent=2, default=str)
    print("  Saved chunks with vectors.")

    print(f"\nAll outputs saved to {output_dir}/")
    print("Summary:")
    print(f"  Extraction (total): {extract_time:.2f}s ({num_pages} pages)")
    print(f"  Chunking:          {chunk_time:.2f}s")
    print(f"  Embedding:         {embed_time:.2f}s ({num_chunks} chunks)")
    print(f"  Dense model:       {dense_model}")

if __name__ == "__main__":
    main()