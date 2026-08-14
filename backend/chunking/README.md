# Chunking Subsystem

The **Chunking Module** (`backend/chunking/`) partitions parsed document blocks (`NormalizedBlock[]`) into structured, token-bounded retrieval units (`Chunk[]`) optimized for vector embedding and hybrid retrieval.

---

## 1. Key Capabilities & Features

- **Token Sliding Window & Boundary Protection** ([`chunk_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/chunking/chunk_tool.py)):
  - Splits running text passages into chunks of `chunking.size` (default: 400 tokens) with `chunking.overlap` (default: 50 tokens).
  - Merges heading blocks into subsequent text blocks to retain hierarchical context.
  - Merges callout headers (`NOTE`, `WARNING`, `IMPORTANT`, `CAUTION`, `DANGER`) with their descriptive bodies.
- **Semantic Boundary Chunking (Chonkie)**:
  - Integrates `chonkie.SemanticChunker` (using local embeddings like `minishlab/potion-base-32M`) to split text at natural semantic topic transitions.
- **Atomic Element Protection & Large Table Partitioning**:
  - Tables and image captions are treated as atomic units.
  - When tables exceed maximum chunk size, `_split_table_rows()` partitions table rows while prepending original column headers to every resulting sub-table chunk.
- **Memory Blowup Defense (Dot-Leader Collapsing)**:
  - Collapses repeating character sequences (e.g. TOC dot-leaders `..........`) that would otherwise cause quadratic memory explosions ($O(N^2)$ self-attention) during transformer embedding.
- **Unified CAD Chunking** ([`cad_chunk_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/chunking/cad_chunk_tool.py)):
  - Executes single-pass LLM chunking and tag enrichment tailored for CAD drawings and electrical circuit schematics.

---

## 2. Dependencies & Integrations

- **tiktoken**: Exact token count evaluation via `cl100k_base` encoding.
- **chonkie**: Semantic boundary chunking engine.
- **backend.core.schemas**: `NormalizedBlock` and `Chunk` data contracts.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    Blocks[NormalizedBlock List] --> TypeCheck{Block Type?}
    
    TypeCheck -->|Heading / Callout| MergeHead[Merge into Subsequent Text Block]
    TypeCheck -->|Text Block| TextClean[Dot-Leader Collapsing & Token Counting]
    TypeCheck -->|Table Block| TableCheck{Table > Size?}
    TypeCheck -->|Image Caption| Atomic[Preserve as Single Atomic Chunk]
    
    MergeHead --> TextClean
    TextClean --> Splitter{Strategy: Sliding Window vs Chonkie Semantic}
    Splitter --> ChunksOut[Token-Bounded Text Chunks]

    TableCheck -->|No| Atomic
    TableCheck -->|Yes| RowSplit[Split Rows + Prepend Header to Each Chunk]
    RowSplit --> ChunksOut
    Atomic --> ChunksOut

    ChunksOut --> Enrich[state['chunks'] -> enrich_chunks step]
```

---

## 4. Configuration & Testing

### Configuration Blueprint (`config/global.yaml`)
```yaml
chunking:
  strategy: semantic                  # semantic | sliding_window
  size: 400
  overlap: 50
  section_aware: true
  split_large_tables: true
  chunking_model: minishlab/potion-base-32M
```

### Verification & Unit Tests
```powershell
# Run chunking unit tests
pytest tests/test_chunking.py
```
