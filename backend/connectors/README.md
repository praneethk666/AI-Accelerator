# Database Connectors Module

The Connectors module implements drivers for external data sources, exposing them as tools for the agent layer.

## Core Dependencies

* **psycopg**: PostgreSQL adapter.
* **re**: Parses sql command patterns to enforce security policies.

## Submodules & Drivers

* `sql_read.py`: Wires read-only database query tools.
* `google_drive.py` (Planned): Handles OAuth2 integrations and folder downloads.

## Security & Read-Only Enforcement (`sql_read.py`)

To prevent the agent from performing destructive queries (such as SQL injections or accidental drop statements), the connector runs a multi-stage validation check:

### 1. Prefix Whitelisting
Before execution, the SQL string is checked against allowed read-only command prefixes:
$$\text{Whitelisted Prefixes} = \{\text{SELECT}, \text{WITH}, \text{SHOW}, \text{DESCRIBE}, \text{DESC}, \text{EXPLAIN}\}$$
The query must begin with one of these terms (ignoring comments).

### 2. Keyword Blacklisting
The parser scans the tokenized SQL string for blacklisted mutation keywords:
$$\text{Blacklist} = \{\text{INSERT}, \text{UPDATE}, \text{DELETE}, \text{DROP}, \text{ALTER}, \text{CREATE}, \text{TRUNCATE}, \text{INTO}, \dots\}$$
If any forbidden keyword is found, the connection is blocked, raising a `ValueError`.

### 3. Function Sanitization
Blocks administrative database functions like `pg_read_file`, `lo_import`, or `dblink` to prevent local file reads or external networking calls from within a database query.

### 4. Query Limits
The system automatically appends a `LIMIT` clause (default: 200 rows) to prevent the database connection from loading oversized datasets into memory.
