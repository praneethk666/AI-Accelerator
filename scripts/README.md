# Operational Scripts Module

The Scripts module contains operational CLI utilities for managing data, running tests, and debugging pipeline steps.

## Core Utilities

* `agent_chat.py`: An interactive, multi-turn terminal interface for communicating with the LangGraph agent. Logs history to Postgres.
* `reset_state.sh` / `reset_state.bat`: Resets local development states by truncating Postgres tables, deleting Qdrant collections, and wiping visual crops under `uploads/`.
* `init_db.sql`: Database schema definition file loaded during database initialization.

## Research & Development Utilities (`scripts/dev/`)

The `scripts/dev/` subfolder contains research tools for testing and comparing model performance:
* `make_scanned.py`: Converts digital PDFs to scanned images to test OCR engines.
* `ocr_bakeoff.py`: Benchmarks PaddleOCR, Surya, and VLM engines against ground-truth texts.
* `ocr_compare.py` / `ocr_diff.py`: Measures OCR accuracy metrics.
* `table_compare.py`: Compares Docling TableFormer against VLMs for structured table parsing.
