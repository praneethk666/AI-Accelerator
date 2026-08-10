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

### 3. Intent Classification (`intent_eval.py` + `intent_dataset.py`)
Scores the agent's intent classifier (`backend/agent/intent_classifier.py`), which decides
whether a turn is forced to search the corpus (`document_question`, `action`) or may answer
directly (`follow_up`, `general`). `intent_dataset.py` holds 75 labelled messages; every
judgement call carries a `note` explaining the label.

Three numbers, because they answer different questions:
* **Label accuracy**: did it pick the exact intent? Used for prompt tuning.
* **Routing accuracy**: did it pick the right *path*? Four labels collapse into two routes,
  so a label can be wrong while agent behaviour stays correct.
* **Grounding misses**: a message that needed the documents was allowed to answer without
  them. Reported separately and expected to be **0** — this is the failure mode that yields
  confident, unsourced answers, and it can regress while accuracy stays flat.

```bash
# uses config query.agent.intent by default; --provider/--model to override
python -m backend.evaluation.intent_eval --provider groq --model llama-3.3-70b-versatile \
    --delay 2.1 --json results.json
```
`--delay` paces requests for rate-limited free tiers. Exits non-zero if accuracy falls below
95% or any grounding miss occurs, so it can gate a release. Needs a provider key and is run by
hand; `tests/test_intent_dataset.py` validates the dataset and metric maths offline in CI.
