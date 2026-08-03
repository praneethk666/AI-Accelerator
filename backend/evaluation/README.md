# Evaluation Module

The Evaluation module provides a testing framework to measure the quality of the retrieval and generation pipelines.

## Dependencies

* **pytest**: Runs the test suites.
* **numpy / pandas**: Aggregates retrieval metrics.
* **backend.core.llm_client**: Used to run LLM-as-a-judge prompts.

## Metrics & Implementation

### 1. Retrieval Evaluation (`run_eval.py`)
Matches search results against a golden QA set of query-document-page pairs:
* **Hit Rate (Recall@K)**: Checks if the correct document exists in the top $K$ retrieved chunks:
  $$\text{Hit Rate} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{source}_q \in \text{retrieved}_q[:K])$$
* **MRR (Mean Reciprocal Rank)**: Calculates the reciprocal rank of the first correct chunk:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **mAP (Mean Average Precision)**: Measures rank quality across multiple relevant chunks.

### 2. Generator Evaluation (LLM-as-a-Judge)
Queries a judge model using structural prompts:
* **Faithfulness Prompt**: Asks the judge model to verify if every fact in the generated answer exists in the retrieved source text, identifying hallucinations.
* **Relevance Prompt**: Rates if the generated response directly answers the query.
* **Citation Grounding**: Validates that all inline brackets `[filename, p.N]` point to pages that actually contain the asserted facts.
