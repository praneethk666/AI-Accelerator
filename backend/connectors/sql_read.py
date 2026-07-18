"""Read-only SQL agent tool.

Runs a SELECT-style query against a configured database and returns a structured
result for agent use. Connection comes from env; writes/destructive statements are
blocked before execution.
"""
from __future__ import annotations

import datetime
import decimal
import os
import re
from typing import Any

_READ_ONLY_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")
_FORBIDDEN_TOKENS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "grant",
    "revoke",
    "merge",
    "copy",
    "call",
    "execute",
    "comment",
    "lock",
    "set",
    "reset",
    "begin",
    "commit",
    "rollback",
    "vacuum",
    "analyze",
    "refresh",
    "into",
)
_FORBIDDEN_FUNCTIONS = (
    "nextval",
    "setval",
    "pg_advisory_lock",
    "pg_advisory_xact_lock",
    "pg_notify",
)


def sql_dsn_from_env() -> str:
    """Connector DSN. Prefer a dedicated env var; fall back to POSTGRES_URL."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return (
        os.getenv("SQL_READONLY_URL")
        or os.getenv("SQL_TOOL_URL")
        or os.getenv("POSTGRES_URL")
        or ""
    )


def read_sql(query: str, *, limit: int = 200) -> dict[str, Any]:
    """Execute a read-only SQL query and return columns + rows."""
    normalized = _validate_read_only_query(query)
    safe_limit = _safe_limit(limit)
    dsn = sql_dsn_from_env()
    if not dsn:
        raise RuntimeError("SQL_READONLY_URL or POSTGRES_URL must be set")

    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_apply_limit(normalized, safe_limit))
            columns = [col.name for col in (cur.description or [])]
            raw_rows = cur.fetchall() if cur.description else []
        rows = [
            {col: _json_safe(value) for col, value in zip(columns, row)}
            for row in raw_rows
        ]
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    finally:
        conn.close()


class SQLReadTool:
    name = "sql_read"
    description = (
        "Run a read-only SQL query against the configured database. "
        "Available tables:\n"
        "- documents (document_id UUID, filename TEXT, file_type TEXT, file_path TEXT, document_type TEXT, industry TEXT, route TEXT, confidence REAL, status TEXT, errors JSONB, progress REAL)\n"
        "- chunks (chunk_id UUID, document_id UUID REFERENCES documents, text TEXT, token_count INTEGER, tags JSONB, source_ref JSONB, table_data JSONB, image_path TEXT)\n"
        "- document_pages (document_id UUID REFERENCES documents, page INTEGER, image_path TEXT, width INTEGER, height INTEGER)\n"
        "- conversations (id BIGSERIAL, session_id TEXT, role TEXT, content TEXT, metadata JSONB)"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Read-only SQL query to run.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum rows to return.",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, limit: int = 200) -> dict[str, Any]:
        return read_sql(query, limit=limit)

    __call__ = run


def _connect(dsn: str):
    import psycopg

    return psycopg.connect(dsn, autocommit=True, options="-c default_transaction_read_only=on")


def _validate_read_only_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    stripped = query.strip()
    if ";" in stripped.rstrip(";"):
        raise ValueError("multiple SQL statements are not allowed")

    collapsed = re.sub(r"\s+", " ", stripped).strip()
    inspected = _inspection_text(collapsed)
    lowered = inspected.lower().lstrip("(")

    if not lowered.startswith(_READ_ONLY_PREFIXES):
        raise ValueError("only read-only queries are allowed")

    for token in _FORBIDDEN_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            raise ValueError(f"forbidden SQL keyword: {token}")

    for function in _FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(function)}\s*\(", lowered):
            raise ValueError(f"forbidden SQL function: {function}")

    return collapsed.rstrip(";")


def _safe_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    return max(1, min(parsed, 1000))


def _apply_limit(query: str, limit: int) -> str:
    lowered = query.lower()
    if re.search(r"\blimit\s+\d+\b", lowered):
        return query
    if lowered.startswith("show") or lowered.startswith("describe") or lowered.startswith("desc"):
        return query
    return f"{query} LIMIT {limit}"


def _inspection_text(query: str) -> str:
    """Normalize query text for safety inspection.

    Removes comments and quoted literals so checks focus on executable SQL rather
    than words that only appear in strings or comments.
    """
    no_block_comments = re.sub(r"/\*.*?\*/", " ", query, flags=re.S)
    no_line_comments = re.sub(r"--[^\n]*", " ", no_block_comments)
    no_single_quotes = re.sub(r"'(?:''|[^'])*'", "''", no_line_comments)
    no_double_quotes = re.sub(r'"(?:""|[^"])*"', '""', no_single_quotes)
    return re.sub(r"\s+", " ", no_double_quotes).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    return value
