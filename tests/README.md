# Pytest Test Automation Suite

The Tests module implements the validation and verification test suite for the accelerator pipelines.

## Test Structures

* `test_smoke.py`: Fast verification test that executes without database dependencies. Validates basic schemas and checks if environment files load successfully.
* `test_ingestion.py`: End-to-end ingestion test. Uploads mock test files to verify categorization, extraction, chunking, and database indexing pipelines.
* `test_retrieval.py`: Tests retrieval methods (naive, hybrid, reranked) and queries Qdrant collections.
* `test_agent.py`: Exercises the LangGraph agent executor loop, verifying native tool calling and human-in-the-loop write approvals.

## Running Tests

Run the full test suite using `pytest`:
```bash
# Run all tests
pytest

# Skip tests requiring active database connections
pytest -m "not needs_db"

# Run a specific test file
pytest tests/test_smoke.py
```
