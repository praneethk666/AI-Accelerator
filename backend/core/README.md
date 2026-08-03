# Core Core Module

The Core module implements shared contracts, data schemas, API client wrappers, and system utilities.

## Core Dependencies

* **pydantic**: Defines the baseline serialization schemas.
* **yaml**: Loads configuration profiles.
* **dotenv**: Manages environment variables.
* **requests / httpx**: External API integrations.

## Core Files & Architecture

* `schemas.py`: Relational schemas and serialization wrappers.
  * `NormalizedBlock`: Structured page segments (`heading`, `text`, `table`, `image_caption`).
  * `Chunk`: Refined vector database elements (retaining `vector` fields, `sparse_vector` mappings, and metadata tags).
  * `SourceRef`: Citation details (file name, page number, slide index, sheet name, bounding box coordinates).
* `tool.py`: Defines the `Tool` interface (`name` + `run(state, config)`).
* `config.py`: Handles configuration parsing (`load_config`). Resolves environment variables (`${VAR}`) dynamically without schema enforcement, returning a standard python dictionary.
* `llm_client.py` / `vision_client.py`: Client wrappers for Google AI Studio, OpenAI, and local Ollama integrations. Implements backoff retries to manage rate limits (429 errors).
* `models.py`: Caches local singletons (reranker, dense, and sparse embedding models) and implements the model `warm_up()` startup sequence.
* `pacing.py`: Implements rate-limit pacing delays for API calls.
* `tracing.py` / `usage.py`: Integrates tracing backends (Langfuse) and logs LLM token usage.
