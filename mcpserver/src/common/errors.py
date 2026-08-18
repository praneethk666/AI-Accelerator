"""
errors.py — Standard JSON-RPC 2.0 error representations and sanitization helpers.
Ensures zero stack trace leakage to clients.
"""

from typing import Any, Optional


class JSONRPCErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom Application & Security error codes
    UNAUTHORIZED = -32001
    FORBIDDEN = -32002
    RATE_LIMITED = -32003
    SECURITY_VIOLATION = -32004
    SESSION_MISMATCH = -32005


def make_jsonrpc_error(
    code: int,
    message: str,
    data: Optional[Any] = None,
    req_id: Optional[Any] = None,
) -> dict:
    """
    Constructs a spec-compliant JSON-RPC 2.0 error response object.
    Never includes raw stack traces or internal implementation details.
    """
    error_payload = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error_payload["data"] = data

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": error_payload,
    }


def make_jsonrpc_success(
    result: Any,
    req_id: Optional[Any] = None,
) -> dict:
    """Constructs a spec-compliant JSON-RPC 2.0 success response object."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }
