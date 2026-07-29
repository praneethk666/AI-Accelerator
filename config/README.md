# Configuration Engine

The Config module manages pipeline profiles, runtime variables, and prompt templates.

## Main Profiles

* `global.yaml`: The primary active settings file. Wires system configs, model declarations, directories, and pipeline steps.
* `pipeline.example.yaml`: Reference configuration blueprint outlining all available properties.

## Configuration Structure

The configuration schema is split into the following sections:

### 1. Ingestion Pipelines
```yaml
ingestion:
  steps:
    - categorize
    - extract
    - vision_enrichment
    - chunk
    - enrich_chunks
    - embed
    - index
```
Defines step lists and route gate rules (e.g., bypassing `vision_enrichment` for text-only routes).

### 2. Query Pipelines
```yaml
query:
  steps:
    - query_planner
    - retrieval
    - answerer
```
Configures query planning, retrieval strategies (default: `hybrid_rerank`), and conversational agents.

### 3. Model Declarations
Maps providers, model versions, temperatures, and timeouts for `llm`, `vision`, and `vision_ocr` connections.

### 4. Database Setup
Defines Postgres connection limits and Qdrant collections.

### 5. Prompt Settings
Prompts for categorization, visual indexing, retrieval planning, and answering.
