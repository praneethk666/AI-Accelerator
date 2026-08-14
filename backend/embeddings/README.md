# Vector Embeddings Subsystem

The **Embeddings Module** (`backend/embeddings/`) generates dense semantic vectors and sparse lexical BM25 vectors for text passages to enable high-recall hybrid vector retrieval.

---

## 1. Key Capabilities & Features

- **Contextual Vector Augmentation** ([`embed_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/embeddings/embed_tool.py)):
  - Augments embedding inputs with enrichment metadata (`tags["summary"]` + `tags["keywords"]` + `chunk["text"]`) prior to encoding:
    $$\text{Embedding Input} = \text{tags["summary"]} + \text{"\nKeywords: "} + \text{tags["keywords"]} + \text{chunk["text"]}$$
  - Enriches vector representation without altering the original raw text.
- **Multimodal Dense Vector Support**:
  - Supports local inference models (`BAAI/bge-m3`, 1024-dim; `nomic-ai/nomic-embed-text-v1.5`, 768-dim) and remote cloud APIs (`OpenAIEmbeddingsAPIClient`, `JinaEmbeddingsAPIClient`).
  - Enforces unit normalization ($L_2$ norm) for cosine similarity calculation.
- **Sparse BM25 Indexing**:
  - Generates sparse passage vectors using Qdrant Fastembed BM25 (`{"indices": [...], "values": [...]}`).
- **Hardware Acceleration & Memory Protection**:
  - Batches text inputs in slices of `dense_batch_size` (default: 16) and invokes `torch.mps.empty_cache()` on Apple Silicon / CUDA to prevent OOM crashes.

---

## 2. Dependencies & Integrations

- **sentence-transformers**: Local dense embedding inference.
- **fastembed**: Qdrant BM25 sparse vector computation.
- **backend.core.models**: Singleton model caching and Jina/OpenAI client wrappers.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    ChunksIn[Enriched Chunks from Pipeline] --> Augment[Context Augmentation: summary + keywords + text]
    
    Augment --> DenseGen[Dense Embedder: BGE-M3 / OpenAI / Jina]
    Augment --> SparseGen[Sparse Embedder: FastEmbed BM25]

    DenseGen --> Normalize[L2 Unit Normalization]
    SparseGen --> SparseMap[Index/Value Token Weights]

    Normalize --> Stamp[chunk['vector'] = 1024-dim list]
    SparseMap --> Stamp2[chunk['sparse_vector'] = dict]

    Stamp & Stamp2 --> StateOut[state['chunks'] -> index step]
```

---

## 4. Configuration & Testing

### Configuration Blueprint (`config/global.yaml`)
```yaml
embeddings:
  dense_provider: openai             # openai | local | jina
  dense_model: text-embedding-3-small
  dense_dim: 1024
  dense_batch_size: 16
  sparse_model: Qdrant/bm25
```

### Verification & Unit Tests
```powershell
# Test embedding generators
pytest tests/test_jina_embeddings.py tests/test_openai_embeddings.py
```
