# Retrieval & Answering Module

The Retrieval module rewrites user queries, executes hybrid database searches, reranks results, and synthesizes cited, grounded answers.

## Dependencies

* **fastembed** / **sentence-transformers**: Runs the local reranking model (`BAAI/bge-reranker-large`).
* **psycopg**: Queries PostgreSQL database stores.
* **qdrant-client**: Queries Qdrant dense/sparse collections.
* **backend.core.llm_client**: Wires LLM completions for query planning and answering.

## Execution Flows

### 1. Query Planner (`query_planner.py`)
* Receives the user question and the past 10 conversation turns.
* Sends a JSON prompt to the LLM to perform two tasks:
  1. **Rewrite**: Resolve pronouns (e.g. "it", "they") using the conversation history to make the query self-contained.
  2. **Decompose**: Split multi-part questions into individual search queries (up to `max_sub_questions`, default: 4).
* Returns `standalone_query` and `sub_questions`.

### 2. Retrieval Search (`retrieval.py`)
* Iterates through the list of sub-questions.
* Applies hard filters (`document_id`) and soft filters (`doc_type`, `industry`).
* Runs the search query:
  * **Dense**: Vector similarity search in Qdrant.
  * **Sparse**: BM25 keyword matching in Qdrant.
  * **RRF**: Merges dense and sparse ranks using Reciprocal Rank Fusion ($k=60$):
    $$\text{RRF Score}(d) = \frac{1}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{Rank}_{\text{sparse}}(d)}$$
  * **Reranker**: Evaluates the top results against the query using `bge-reranker-large` to compute final matching scores.
* **Soft Filter Recovery**: If soft filters restrict the search too much and return 0 results, the query is run again without the soft filters.

### 3. Answer Generation & Context Expansion (`answerer.py`)
* **Context Expansion (`_expand_thin_chunks`)**: Reviews retrieved chunks. If a chunk is short (e.g., headings, short tables, or snippets under 120 tokens), the tool queries Postgres for the complete page text, replacing the short snippet with full page context.
* **Grounding Synthesis**: Sends the expanded page contexts to the LLM. Enforces strict rules: copy technical specs verbatim, use inline citations in the format `[filename, p.N]`, and respond with "I could not find this..." if the context is insufficient.
* **Conversation Sync**: Saves the query and generated response to the PostgreSQL `conversations` table.
