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
    # server-side filesystem / external access — blocked even inside a SELECT so a
    # prompt-injected query can't read/write host files or reach out via dblink.
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "dblink",
    "dblink_exec",
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

    # Read-only transaction + hard statement/idle timeouts so an LLM-authored
    # pg_sleep, accidental cross join, or huge scan can't pin a worker thread.
    return psycopg.connect(
        dsn, autocommit=True,
        prepare_threshold=None,
        options=(
            "-c default_transaction_read_only=on "
            "-c statement_timeout=15000 "
            "-c idle_in_transaction_session_timeout=15000"
        ),
    )


def _validate_read_only_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    stripped = query.strip().rstrip(";").strip()
    # Validate against the comment/literal-stripped form so keywords in strings or
    # comments don't false-trigger, and a real second statement is still caught.
    inspected = _inspection_text(stripped)
    if ";" in inspected.rstrip(";").strip():
        raise ValueError("multiple SQL statements are not allowed")
    lowered = inspected.lower().lstrip("(")

    if not lowered.startswith(_READ_ONLY_PREFIXES):
        raise ValueError("only read-only queries are allowed")

    for token in _FORBIDDEN_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            raise ValueError(f"forbidden SQL keyword: {token}")

    for function in _FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(function)}\s*\(", lowered):
            raise ValueError(f"forbidden SQL function: {function}")

    # Return with ORIGINAL newlines preserved (do NOT collapse). A collapsed query
    # turns a mid-query '-- comment' into a trailing one that would comment out the
    # appended LIMIT; keeping newlines (and appending LIMIT on its own line) prevents it.
    return stripped


def _safe_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    return max(1, min(parsed, 1000))


def _apply_limit(query: str, limit: int) -> str:
    # Decide on the comment/literal-stripped text so a commented-out '-- limit 5' or a
    # LIMIT inside a string can't fool the guards.
    inspected = _inspection_text(query).lower()
    if inspected.startswith(("show", "describe", "desc", "explain")):
        return query
    if re.search(r"\blimit\s+\d+\b", inspected):
        return query
    # Append on a NEW line: a trailing '-- comment' on the query's last line would
    # otherwise swallow the LIMIT and dump the whole table (AGENT-1).
    return f"{query}\nLIMIT {limit}"


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
