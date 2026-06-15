"""Full pipeline test on Scanned_8pages.pdf: extraction -> chunking -> embedding.

Uses configuration from config/global.yaml.
Caches extraction output (blocks/profiles) to JSON so repeated runs can
skip the slow OCR step while iterating on chunking/embedding.
"""

import sys
import os
import json
import time
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.extraction.mixed_pdf.tool import MixedPDFTool
from backend.chunking.chunk_tool import ChunkTool
from backend.embeddings.embed_tool import EmbedTool


def load_config(config_path: str = "config/global.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dict(obj):
    """Recursively convert pydantic models (v1/v2), dataclasses, or plain
    objects into plain dicts/lists so downstream tools can use dict.get()
    on nested fields like source_ref."""
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if hasattr(obj, "model_dump"):       # pydantic v2
        return _to_dict(obj.model_dump())
    if hasattr(obj, "dict"):             # pydantic v1
        return _to_dict(obj.dict())
    if hasattr(obj, "__dict__"):         # plain objects / dataclasses
        return _to_dict(dict(obj.__dict__))
    return obj


def _check_errors(state: dict, stage: str):
    errors = state.get("errors")
    if errors:
        print(f"   !!! ERRORS during {stage}:")
        for err in errors:
            print(f"       [{err.get('tool')}] {err.get('message')}")


def main():
    config = load_config()

    pdf_path = "test-data/Mixed.pdf"
    pdf_name = "A_Mixed"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    blocks_file = output_dir / f"{pdf_name}_blocks.json"
    profiles_file = output_dir / f"{pdf_name}_profiles.json"

    USE_CACHE = True  # set False to force re-extraction

    if USE_CACHE and blocks_file.exists() and profiles_file.exists():
        print("1. Loading cached extraction results...")
        with open(blocks_file, "r", encoding="utf-8") as f:
            blocks_dict = json.load(f)
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles_dict = json.load(f)
        num_pages = len(profiles_dict)
        extract_time = 0.0
        print(f"   Loaded {len(blocks_dict)} blocks, {num_pages} page profiles")
        state = {
            "blocks": blocks_dict,
            "page_profiles": profiles_dict,
            "document_id": pdf_name,
            "file_path": pdf_path,
        }
    else:
        # ---------- 1. Extraction ----------
        print("1. Extracting...")
        start_extract = time.time()
        tool = MixedPDFTool()
        state = {"file_path": pdf_path, "document_id": pdf_name}
        state = tool.run(state, config)
        extract_time = time.time() - start_extract
        print(f"   Total extraction time: {extract_time:.2f}s (all pages)")

        _check_errors(state, "extraction")

        blocks_dict = [_to_dict(b) for b in state["blocks"]]
        profiles_dict = [_to_dict(p) for p in state["page_profiles"]]
        num_pages = len(profiles_dict)
        print(f"   Number of pages: {num_pages}")
        print(f"   Average extraction time per page (approx): {extract_time/num_pages:.2f}s")

        with open(blocks_file, "w", encoding="utf-8") as f:
            json.dump(blocks_dict, f, indent=2, default=str)
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(profiles_dict, f, indent=2, default=str)
        print("   Saved blocks and profiles.")

        state["blocks"] = blocks_dict
        state["page_profiles"] = profiles_dict

    # ---------- 2. Chunking ----------
    print("2. Chunking...")
    start_chunk = time.time()
    chunker = ChunkTool()
    state = chunker.run(state, config)
    chunk_time = time.time() - start_chunk
    print(f"   Chunking completed in {chunk_time:.2f}s")

    _check_errors(state, "chunking")

    if "chunks" not in state:
        print("   !!! 'chunks' key missing from state — aborting.")
        return

    num_chunks = len(state["chunks"])
    print(f"   Created {num_chunks} chunks")

    if num_chunks == 0:
        print("   !!! 0 chunks created — check chunking errors above.")
        return

    # Count chunks per page (from source_ref.page)
    chunks_per_page = {}
    for ch in state["chunks"]:
        source_ref = ch.get("source_ref", {})
        page = source_ref.get("page", 0) if isinstance(source_ref, dict) else 0
        chunks_per_page[page] = chunks_per_page.get(page, 0) + 1
    print("   Chunks per page:", chunks_per_page)

    with open(output_dir / f"{pdf_name}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f, indent=2, default=str)
    print("   Saved chunks (before embedding).")

    # ---------- 3. Embedding ----------
    dense_model = config["embeddings"]["dense_model"]
    print(f"3. Embedding (using {dense_model})...")
    start_embed = time.time()
    embedder = EmbedTool()
    state = embedder.run(state, config)
    embed_time = time.time() - start_embed
    print(f"   Total embedding time: {embed_time:.2f}s")

    _check_errors(state, "embedding")

    avg_per_chunk = embed_time / num_chunks if num_chunks else 0
    print(f"   Average embedding time per chunk: {avg_per_chunk:.3f}s")

    # Per-page embedding time (approximated by chunk count)
    print("   Approximate embedding time per page (based on chunk count):")
    for page, cnt in sorted(chunks_per_page.items(), key=lambda x: (x[0] is None, x[0])):
        approx = cnt * avg_per_chunk
        page_label = page if page is not None else "?"
        print(f"     Page {page_label}: {approx:6.3f}s ({cnt} chunks)")

    with open(output_dir / f"{pdf_name}_chunks_embedded.json", "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f, indent=2, default=str)
    print("   Saved chunks with vectors.")

    print(f"\nAll outputs saved to {output_dir}/")
    print("Summary:")
    print(f"  Extraction (total): {extract_time:.2f}s ({num_pages if 'num_pages' in dir() else len(profiles_dict)} pages)")
    print(f"  Chunking:          {chunk_time:.2f}s")
    print(f"  Embedding:         {embed_time:.2f}s ({num_chunks} chunks)")
    print(f"  Dense model:       {dense_model}")


if __name__ == "__main__":
    main()