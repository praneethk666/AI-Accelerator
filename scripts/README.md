# Operational CLIs & Database Utilities Module

The **Scripts Module** (`scripts/`) provides operational command-line interfaces (CLIs), database migration utilities, and development state resets for the AI-Accelerator platform.

---

## 1. Operational Scripts Catalog

| Script | Purpose & Execution Flow |
|---|---|
| [`agent_chat.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/agent_chat.py) | **Terminal Agent CLI**: Interactive multi-turn CLI for chatting with the LangGraph agent. Supports human-in-the-loop write confirmations in the terminal and logs turns to PostgreSQL. |
| [`run_ingest.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/run_ingest.py) | **CLI Batch Ingestion**: Ingests files directly via the command line without launching the HTTP API. |
| [`migrate_token_costs.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/migrate_token_costs.py) | **Cost Migration**: Backfills token usage and cost accounting records into the database. |
| [`check_detection.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/check_detection.py) | **PDF Detector Verification**: Analyzes vector text vs scanned bitmap page distributions across PDF sets. |
| [`test_jina_reranker_live.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/test_jina_reranker_live.py) | **Live Jina Reranker Test**: Verifies active Jina API keys, multi-key failover rotation, and response latency. |
| [`test_docling_server.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/test_docling_server.py) | **Docling Server Test**: Health checks and validates remote Docling server connectivity. |
| [`test_excel.py`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/test_excel.py) | **Excel Extraction Test**: Runs extraction on test spreadsheets and verifies markdown table formatting. |
| [`reset_state.sh`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/reset_state.sh) / `reset_state.bat` | **Development Reset**: Truncates PostgreSQL tables (`documents`, `chunks`, `conversations`), wipes Qdrant collections, and empties `uploads/`. |
| [`init_db.sql`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/init_db.sql) | **Database DDL**: Authoritative relational PostgreSQL schema definition loaded during startup. |

---

## 2. Research & Development Tools (`scripts/dev/`)

The [`scripts/dev/`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/dev/README.md) directory contains research scripts for benchmarking OCR models, evaluating TableFormer vs VLMs, and measuring character error rates (CER). See [`scripts/dev/README.md`](file:///d:/AI-Acc-updated/AI-Accelerator/scripts/dev/README.md).

---

## 3. Usage Examples

```powershell
# Interactive terminal chat with agent
python scripts/agent_chat.py

# Run standalone document ingestion
python scripts/run_ingest.py --file path/to/document.pdf

# Test live Jina reranker with key rotation
python scripts/test_jina_reranker_live.py

# Reset local development environment
./scripts/reset_state.sh
```
