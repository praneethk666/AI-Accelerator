# Chunk Enrichment Subsystem

The **Enrichment Module** (`backend/enrichment/`) generates domain summaries and keyword tags for text chunks using structured LLM prompts, significantly boosting metadata filtering and dense/sparse search recall.

---

## 1. Key Capabilities & Features

- **Batched LLM Prompt Execution** ([`enrich_chunks_tool.py`](file:///d:/AI-Acc-updated/AI-Accelerator/backend/enrichment/enrich_chunks_tool.py)):
  - Groups chunks into batches (size 10–15) to process multiple passages per single LLM call, reducing API roundtrips and staying within rate limits.
- **Dynamic Completion Token Budget**:
  - Dynamically scales completion token limits:
    $$\text{max\_tokens} = 160 \times \text{batch\_size} + 256$$
  - Prevents truncated JSON responses across large batches.
- **Pacing Delay**:
  - Integrates minimum-interval pacing (`min_interval_s: 0.3s`–`2.0s`) to prevent HTTP 429 rate limit exceptions.
- **Offline TF-IDF Keyword Fallback**:
  - Automatically falls back to local TF-IDF frequency term extraction if external LLM APIs fail or time out.

---

## 2. Dependencies & Integrations

- **backend.core.llm_client**: LLM completions for OpenAI GPT-4o-mini, Google Gemini, or Groq.
- **re & collections.Counter**: Offline TF-IDF keyword extraction.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    ChunksIn[Raw Chunks from chunk_tool] --> Batcher[Group Chunks into Batches Size=15]
    Batcher --> DynamicBudget[Compute Dynamic Token Budget: 160 * Batch + 256]
    DynamicBudget --> LLMCall[Send Batch Prompt to LLM Provider]

    LLMCall --> ResponseCheck{LLM Call Succeeded?}
    ResponseCheck -->|Yes| ParseJSON[Parse JSON Array of summary and keywords]
    ResponseCheck -->|No / 429 Error| Fallback[Local TF-IDF Keyword Extractor]

    ParseJSON --> StampTags[Stamp tags['summary'] and tags['keywords'] on Chunks]
    Fallback --> StampTags

    StampTags --> StateOut[state['chunks'] -> embed step]
```

---

## 4. Configuration & Testing

### Configuration Blueprint (`config/global.yaml`)
```yaml
enrichment:
  summarize: true
  keyword_count: 6
  min_interval_s: 0.3
  batch_size: 15
  provider: openai
  model: gpt-4o-mini
```

### Verification & Unit Tests
```powershell
# Run enrichment unit tests
pytest tests/test_enrichment.py
```
