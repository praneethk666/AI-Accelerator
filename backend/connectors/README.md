# Enterprise Connectors Subsystem

The **Connectors Module** (`backend/connectors/`) integrates external enterprise data sources and relational databases into the agent tool harness.

---

## 1. Key Capabilities & Features

- **SQL Read Connector (`sql_read` Tool)**:
  - Executes read-only queries against external PostgreSQL databases with strict AST safety constraints.
  - Automatically restricts queries to `SELECT` operations, blocks mutation keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`), and caps returned row sets.
- **Enterprise Extensibility**:
  - Modular connector architecture designed to integrate ERP platforms, cloud object stores, Google Drive repositories, and operational data lakes into conversational agent loops.

---

## 2. Dependencies & Testing

- **psycopg**: Connection management and query execution.
- **Verification**:
  ```powershell
  pytest tests/test_sql_read_tool.py
  ```
