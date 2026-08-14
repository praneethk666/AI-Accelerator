# Central Configuration Engine Module

The **Config Module** (`config/`) manages runtime profiles, model declarations, pipeline route maps, database connections, AI safety guardrails, and cost tracking rates for the entire platform.

---

## 1. Key Capabilities & Features

- **Master Configuration Profile**: [`global.yaml`](file:///d:/AI-Acc-updated/AI-Accelerator/config/global.yaml) is the single source of truth for runtime orchestration, model routing, database credentials, and guardrail policies.
- **Dynamic Variable Interpolation**: Resolves `${ENVIRONMENT_VARIABLE}` declarations at runtime using regex substitution in `backend.core.config::load_config()`.
- **Pipeline Route Maps**: Declarative definitions of route paths (`text_default`, `diagram_heavy`, `cad_route`, `circuit_route`, `image_route`) and route gate step filters.
- **Multimodal AI Declarations**: Configurable providers (OpenAI, NVIDIA NIM, Google Gemini, Ollama, Jina AI) across primary LLM, vision captioning, OCR rescue, dense/sparse embeddings, and cross-encoder rerankers.
- **3-Stage AI Safety Policies**: Configuration for input prompt injection checks, Indian-market PII redaction, chunk injection scanners, output hallucination thresholds, and token quotas.
- **Token Cost Matrix**: Native per-model input/output pricing per 1M tokens with configurable USD-to-INR currency conversion rates.

---

## 2. Configuration Profiles Reference

| File | Purpose |
|---|---|
| [`global.yaml`](file:///d:/AI-Acc-updated/AI-Accelerator/config/global.yaml) | Active production configuration loaded by backend services and scripts. |
| `pipeline.example.yaml` | Blueprint and documentation reference demonstrating all configurable fields and comments. |

---

## 3. Configuration Section Breakdown

### 1. Model & Provider Declarations (`llm`, `vision`, `vision_ocr`, `embeddings`)
```yaml
llm:
  provider: openai                    # openai | google | groq | ollama
  model: gpt-4o-mini
  answer_model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

vision:
  provider: openai                    # NVIDIA NIM / OpenAI
  model: meta/llama-3.2-11b-vision-instruct
  base_url: https://integrate.api.nvidia.com/v1
  api_key: ${NVIDIA_API_KEY}

embeddings:
  dense_provider: openai             # openai | local | jina
  dense_model: text-embedding-3-small
  dense_dim: 1024
  sparse_model: Qdrant/bm25
  reranker_provider: jina             # jina | local
  reranker_model: jina-reranker-v2-base-multilingual
  reranker_api_key: ${JINA_API_KEY}  # Supports comma-separated keys for auto-failover
```

### 2. Ingestion Routes & Route Gates (`routes`, `ingestion`)
```yaml
type_to_route:
  cad_drawing: cad_route
  circuit_diagram: circuit_route
  schematic: diagram_heavy
  report: text_default
  manual: text_default
  image: image_route

ingestion:
  steps:
    - categorize
    - extract
    - vision_enrichment
    - chunk
    - cad_chunk_llm
    - enrich_chunks
    - embed
    - index
  route_gates:
    vision_enrichment: [text_default, diagram_heavy, image_route]
    chunk: [text_default, diagram_heavy, image_route]
    cad_chunk_llm: [cad_route, circuit_route]
```

### 3. Query & Agent Settings (`query`)
```yaml
query:
  steps:
    - query_planner
    - retrieval
    - answerer
  planner:
    max_sub_questions: 4
  retrieval:
    method: hybrid_rerank
    candidate_k: 80
    rerank_top_k: 5
    dense_weight: 0.6
    sparse_weight: 0.4
  agent:
    max_iterations: 8
    max_history_messages: 20
    write_tools:
      - ingest_document
```

### 4. Safety Guardrails & Token Accounting (`guardrails`, `models_cost`)
```yaml
guardrails:
  enabled: true
  policy:
    block_threshold: 80
    warn_threshold: 40
  input:
    pii_redact: true
    injection_check: true
  output:
    pii_mask: true
    groundedness:
      enabled: false

models_cost:
  gpt-4o-mini: { input: 0.15, output: 0.60 }
  gpt-4o: { input: 2.50, output: 10.00 }
  text-embedding-3-small: { input: 0.02, output: 0.00 }
```

---

## 4. Verification & Testing

```powershell
# Verify configuration loading and syntax validity
pytest tests/test_config.py
```
