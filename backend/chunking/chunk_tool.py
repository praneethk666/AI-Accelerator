"""Convert extracted blocks into retrieval-sized chunks.

Handles text, heading, table, and image_caption blocks.
Uses chonkie SemanticChunker with the shared dense model (no second model in memory).
Writes chunks into state["chunks"].
"""

import uuid
import re
from typing import List, Dict, Any, Optional

from backend.core.tool import Tool, PipelineState
from backend.core.models import get_dense_model, get_tokenizer

try:
    from chonkie import SemanticChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    # No print – silence is golden

from tabulate import tabulate

# ── singleton chunker (uses shared dense model) ──────────────────────────────
_SEMANTIC_CHUNKER = None
_CHUNKER_SIZE = None

# Chunks with fewer tokens than this are merged into their neighbour.
MIN_TOKENS = 20


def _get_chunker(chunk_size: int, config: dict):
    """Return a cached SemanticChunker that uses the shared dense model."""
    global _SEMANTIC_CHUNKER, _CHUNKER_SIZE
    if _SEMANTIC_CHUNKER is None or _CHUNKER_SIZE != chunk_size:
        dense_model = get_dense_model(config)
        # Create an embedding function that returns a list of floats
        embed_fn = lambda text: dense_model.encode(text, normalize_embeddings=False).tolist()
        _SEMANTIC_CHUNKER = SemanticChunker(
            embedding_function=embed_fn,
            chunk_size=chunk_size,
            threshold=0.5,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=10,
        )
        _CHUNKER_SIZE = chunk_size
    return _SEMANTIC_CHUNKER


def _count_tokens(text: str, config: dict) -> int:
    """Exact token count using bge-large tokenizer."""
    tokenizer = get_tokenizer(config)
    return len(tokenizer.encode(text))


class ChunkTool(Tool):
    name = "chunk"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        try:
            blocks: List[Dict[str, Any]] = state.get("blocks", [])

            # ── chunking config ──────────────────────────────────────────────
            chunking_cfg = config.get("chunking", {})
            max_tokens: int = chunking_cfg.get("size")
            if max_tokens is None:
                raise ValueError("config['chunking']['size'] is required")

            # Ensure dense_model is present for tokenizer (and chunker)
            embeddings_cfg = config.get("embeddings", {})
            dense_model_name = embeddings_cfg.get("dense_model")
            if dense_model_name is None:
                raise ValueError("config['embeddings']['dense_model'] is required")

            # ── merge headings with following content ────────────────────────
            blocks = self._merge_heading_blocks(blocks)

            # ── chunk each block ─────────────────────────────────────────────
            chunks: List[Dict[str, Any]] = []

            for block in blocks:
                block_type = block.get("type")

                if block_type in ("text", "heading"):
                    chunks.extend(
                        self._split_text_block(block, max_tokens, config)
                    )
                elif block_type == "table":
                    chunk = self._make_table_chunk(block, config)
                    if chunk:
                        chunks.append(chunk)
                elif block_type == "image_caption":
                    chunk = self._make_image_chunk(block, config)
                    if chunk:
                        chunks.append(chunk)
                # other block types (image, vector_drawing) are skipped

            # ── merge small chunks (same page only) ──────────────────────────
            final_chunks = self._merge_small_chunks(chunks, MIN_TOKENS)

            state["chunks"] = final_chunks
            state["dense_model"] = dense_model_name

        except Exception as e:
            state.setdefault("errors", []).append({
                "tool": self.name,
                "level": "error",
                "message": str(e),
                "block_id": None,
            })
        return state

    # ── heading merge (unchanged) ─────────────────────────────────────────────
    def _merge_heading_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.get("type") != "heading":
                merged.append(block)
                i += 1
                continue

            heading_text = block.get("text", "").strip()
            combined_parts = [heading_text]
            source_ref = block.get("source_ref", {})
            j = i + 1
            while j < len(blocks):
                next_block = blocks[j]
                next_type = next_block.get("type")
                if next_type == "text":
                    part = next_block.get("text", "").strip()
                    if part:
                        combined_parts.append(part)
                    j += 1
                else:
                    break

            merged.append({
                **block,
                "type": "text",
                "text": "\n\n".join(combined_parts),
                "source_ref": source_ref,
                "metadata": block.get("metadata", {}),
            })
            i = j
        return merged

    # ── small chunk merge (page‑aware) ───────────────────────────────────────
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], min_tokens: int) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks

        # First pass: merge small into previous (same page)
        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            block_type = chunk.get("tags", {}).get("block_type")
            is_structural = block_type in ("table", "image_caption")
            is_small = (
                not is_structural
                and chunk.get("token_count", min_tokens) < min_tokens
            )

            if not is_small:
                result.append(chunk)
                continue

            merged = False
            for i in range(len(result) - 1, -1, -1):
                prev = result[i]
                prev_type = prev.get("tags", {}).get("block_type")
                if prev_type not in ("table", "image_caption") and \
                   prev.get("source_ref", {}).get("page") == chunk.get("source_ref", {}).get("page"):
                    prev["text"] = prev["text"].rstrip() + " " + chunk["text"].strip()
                    prev["token_count"] = prev.get("token_count", 0) + chunk.get("token_count", 0)
                    merged = True
                    break

            if not merged:
                result.append(chunk)

        # Second pass: prepend leading small chunks to next (same page)
        final: List[Dict[str, Any]] = []
        i = 0
        while i < len(result):
            chunk = result[i]
            block_type = chunk.get("tags", {}).get("block_type")
            is_structural = block_type in ("table", "image_caption")
            is_small = (
                not is_structural
                and chunk.get("token_count", min_tokens) < min_tokens
            )

            if is_small and i + 1 < len(result):
                next_chunk = result[i + 1]
                next_type = next_chunk.get("tags", {}).get("block_type")
                if next_type not in ("table", "image_caption") and \
                   next_chunk.get("source_ref", {}).get("page") == chunk.get("source_ref", {}).get("page"):
                    next_chunk["text"] = chunk["text"].strip() + " " + next_chunk["text"].lstrip()
                    next_chunk["token_count"] = chunk.get("token_count", 0) + next_chunk.get("token_count", 0)
                    i += 1
                    continue
            final.append(chunk)
            i += 1

        return final

    # ── text splitting ────────────────────────────────────────────────────────
    def _split_text_block(
        self,
        block: Dict[str, Any],
        max_tokens: int,
        config: dict,
    ) -> List[Dict[str, Any]]:
        text = block.get("text", "")
        if not text.strip():
            return []

        if CHONKIE_AVAILABLE:
            try:
                chunker = _get_chunker(max_tokens, config)
                splits = chunker.chunk(text)
                if splits:
                    chunks = []
                    for split in splits:
                        token_count = _count_tokens(split.text, config)
                        chunks.append(
                            self._create_chunk_from_block(block, split.text, token_count)
                        )
                    return chunks
            except Exception:
                # Fallback to sentence splitter
                pass

        return self._fallback_split(block, text, max_tokens, config)

    def _fallback_split(
        self,
        block: Dict[str, Any],
        text: str,
        max_tokens: int,
        config: dict,
    ) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.replace('\n', ' '))
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = _count_tokens(sent, config)
            if current_tokens + sent_tokens > max_tokens and current:
                chunk_text = " ".join(current)
                chunks.append(self._create_chunk_from_block(block, chunk_text, current_tokens))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current)
            chunks.append(self._create_chunk_from_block(block, chunk_text, current_tokens))
        return chunks

    # ── table / image helpers ─────────────────────────────────────────────────
    def _make_table_chunk(self, block: Dict[str, Any], config: dict) -> Optional[Dict[str, Any]]:
        table_data = block.get("table_data")
        if not table_data:
            text = block.get("text", "")
            if not text.strip():
                return None
            token_count = _count_tokens(text, config)
            return self._create_chunk_from_block(block, text, token_count)

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        markdown = tabulate(rows, headers=headers, tablefmt="pipe")
        token_count = _count_tokens(markdown, config)
        chunk = self._create_chunk_from_block(block, markdown, token_count)
        chunk["table_data"] = table_data
        return chunk

    def _make_image_chunk(self, block: Dict[str, Any], config: dict) -> Optional[Dict[str, Any]]:
        text = block.get("text", "")
        if not text.strip():
            return None
        token_count = _count_tokens(text, config)
        chunk = self._create_chunk_from_block(block, text, token_count)
        metadata = block.get("metadata", {})
        if "raw_image_path" in metadata:
            chunk["image_path"] = metadata["raw_image_path"]
        return chunk

    # ── chunk factory ─────────────────────────────────────────────────────────
    def _create_chunk_from_block(
        self,
        block: Dict[str, Any],
        text: str,
        token_count: int,
    ) -> Dict[str, Any]:
        source_ref = block.get("source_ref", {})
        metadata = block.get("metadata", {})

        return {
            "chunk_id": str(uuid.uuid4()),
            "document_id": block.get("document_id"),
            "text": text,
            "token_count": token_count,
            "tags": {
                "block_type": block.get("type"),
                "language": block.get("language"),
                "page": source_ref.get("page"),
                "section": metadata.get("section"),
            },
            "source_ref": source_ref,
            "table_data": None,
            "image_path": None,
            "vector": None,
            "sparse_vector": None,
        }