# Embeddings Module

The Embeddings module converts text passages into vector representations to support high-recall hybrid retrieval.

## Core Dependencies

* **fastembed** / **sentence-transformers**: Local inference models loaded as singletons.
  * **Dense**: Default `nomic-embed-text-v1.5` (768-dim) or `bge-m3` (1024-dim).
  * **Sparse**: Default Qdrant Fastembed BM25 model.
* **torch**: Used to manage hardware acceleration caches.

## Execution Flow & Code Logic

The embedding workflow is executed in `EmbedTool::run()`:

1. **Context Augmentation (`_embed_input`)**:
   * Before embedding, the tool prepends summary metadata and keyword tags (generated in the enrichment step) to the raw text:
     $$\text{Embedding Input} = \text{tags["summary"]} + \text{"\nKeywords: "} + \text{tags["keywords"]} + \text{chunk["text"]}$$
   * This embeds document-level context directly into the vector representation without modifying the raw text display.
2. **Dense Vector Generation**:
   * Applies the model-specific prefix (e.g. `"search_document: "` for Nomic) to the inputs.
   * Batches the text inputs into slices of `dense_batch_size` (default: 16) to limit peak memory usage.
   * Encodes and normalizes the vectors to unit length ($L_2$ normalization) for cosine similarity queries.
3. **Sparse Vector Generation**:
   * Runs `sparse.passage_embed()` on the inputs to generate index/value mappings representing BM25 term weights.
4. **Memory Allocation Rescue (`_free_mps_cache`)**:
   * On Apple Silicon (MPS), loading YOLO, Docling, and OCR models concurrently can exhaust GPU memory. The tool runs `torch.mps.empty_cache()` prior to embedding to release cached memory allocations.
