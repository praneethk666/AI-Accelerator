# Chunking Module

The Chunking module breaks down parsed page elements (`NormalizedBlock[]`) into structured, token-bounded sequences (`Chunk[]`) optimized for embedding and retrieval.

## Core Dependencies

* **tiktoken**: Used to count exact tokens (using the `cl100k_base` encoding).
* **chonkie**: Used to perform semantic-aware boundary splits if the semantic strategy is enabled.
* **re**: Implements clean-up heuristics for raw OCR text.

## Ingest Processing Logic

The entrypoint is `ChunkTool::run()`. It processes blocks as follows:

### 1. Block Extraction
The tool loops through `state["blocks"]`. Blocks of type `heading` are merged with the subsequent `text` block to keep structural context inside the passage.

### 2. Character Collapsing (`_collapse_repeated_chars`)
To protect local embedding models from memory exhaustion (OOM), the chunker cleans up repeating character patterns:
* **The Problem**: Table-of-contents dot-leaders (e.g. `Section 1.1 ...................... Page 5`) can generate thousands of repeating dot/space characters during OCR. These large, unsplittable sequences cause quadratic memory scaling during self-attention calculations in the embedding model.
* **The Solution**: Regular expressions collapse any pattern of 1-3 characters that repeats more than 9 times down to just 3 repetitions.

### 3. Splitting Strategies
* **Token Sliding Window**: Splits prose text into segments of `chunking.size` (default: 400) tokens with a `chunking.overlap` (default: 50) overlap.
* **Semantic Split (Chonkie)**: When `strategy: semantic` is configured, it instantiates `chonkie.SemanticChunker` to split text at semantic topic transitions.
* **Atomic Elements**: Blocks of type `table` and `image_caption` are never split. If a table exceeds the maximum token length, `_split_table_rows` partitions it by row and prepends the column headers to each sub-table chunk.
