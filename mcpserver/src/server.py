"""
server.py — Production-grade Streamable HTTP + SSE MCP Server.
Compliant with the Model Context Protocol (MCP) Streamable HTTP Transport Specification.

Features:
  1. POST /mcp (JSON-RPC 2.0 protocol endpoint: initialize, ping, tools/list, tools/call)
  2. GET  /mcp (SSE stream endpoint with Last-Event-ID resumability)
  3. Mcp-Session-Id lifecycle and identity binding
  4. Zero stack trace error sanitization
  5. Health & Readiness probe endpoints
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
import uvicorn

from src.auth.middleware import AuthMiddleware
from src.auth.session_manager import session_manager
from src.common.errors import JSONRPCErrorCodes, make_jsonrpc_error, make_jsonrpc_success
from src.common.logging import setup_logging
from src.config import load_config
from src.registry import execute_tool, get_tool_definitions

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "streamable-mcp-server",
    "version": "0.2.0",
}


# ── Health & Readiness Probes ─────────────────────────────────────────────────

async def health_check(request: Request) -> JSONResponse:
    """Liveness probe: verifies the HTTP server is responsive."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "streamable-mcp-server",
            "protocol_version": MCP_PROTOCOL_VERSION,
        }
    )


async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe: verifies tools and configuration readiness."""
    cfg = load_config()
    return JSONResponse(
        {
            "status": "ready",
            "auth_enabled": cfg.auth.enabled,
            "tools_count": len(get_tool_definitions()),
            "smtp_mode": cfg.smtp.mode,
        },
        status_code=200,
    )


# ── JSON-RPC 2.0 Request Router ───────────────────────────────────────────────

async def handle_jsonrpc_request(payload: Dict[str, Any], caller: str) -> Dict[str, Any]:
    """
    Processes a single JSON-RPC 2.0 request dictionary and returns response object.
    """
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if not method:
        return make_jsonrpc_error(
            code=JSONRPCErrorCodes.INVALID_REQUEST,
            message="Invalid Request: 'method' field is required.",
            req_id=req_id,
        )

    # 1. initialize
    if method == "initialize":
        return make_jsonrpc_success(
            result={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {
                        "listChanged": False,
                    },
                    "logging": {},
                },
                "serverInfo": SERVER_INFO,
            },
            req_id=req_id,
        )

    # 2. notifications/initialized
    if method == "notifications/initialized":
        return make_jsonrpc_success(result={}, req_id=req_id)

    # 3. ping
    if method == "ping":
        return make_jsonrpc_success(result={}, req_id=req_id)

    # 4. tools/list
    if method == "tools/list":
        tools = get_tool_definitions()
        return make_jsonrpc_success(
            result={"tools": tools},
            req_id=req_id,
        )

    # 5. tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if not tool_name:
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INVALID_PARAMS,
                message="Invalid params: 'name' is required for tools/call.",
                req_id=req_id,
            )

        try:
            result = await execute_tool(tool_name=tool_name, arguments=arguments, caller=caller)
            
            is_error = result.get("status") == "failed"
            return make_jsonrpc_success(
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                    "isError": is_error,
                },
                req_id=req_id,
            )
        except ValidationError as val_err:
            # Cleanly format Pydantic validation errors
            field_errors = [
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in val_err.errors()
            ]
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INVALID_PARAMS,
                message=f"Validation error: {', '.join(field_errors)}",
                data={"errors": field_errors},
                req_id=req_id,
            )
        except KeyError:
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.METHOD_NOT_FOUND,
                message=f"Tool '{tool_name}' not found.",
                req_id=req_id,
            )
        except Exception as exc:
            logger.error(f"Error executing tool '{tool_name}': {exc}", exc_info=True)
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INTERNAL_ERROR,
                message="Tool execution failed due to an internal error.",
                req_id=req_id,
            )

    # Unknown method
    return make_jsonrpc_error(
        code=JSONRPCErrorCodes.METHOD_NOT_FOUND,
        message=f"Method '{method}' not found.",
        req_id=req_id,
    )


# ── Streamable HTTP Endpoint (POST /mcp) ──────────────────────────────────────

async def mcp_post_endpoint(request: Request) -> JSONResponse:
    """
    Handles POST JSON-RPC 2.0 requests over Streamable HTTP transport.
    """
    caller = getattr(request.state, "caller_identity", "anonymous_caller")
    session = getattr(request.state, "session", None)

    try:
        raw_body = await request.body()
        if not raw_body:
            return JSONResponse(
                make_jsonrpc_error(
                    code=JSONRPCErrorCodes.PARSE_ERROR,
                    message="Parse error: Empty request body.",
                ),
                status_code=400,
            )
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as parse_err:
        logger.warning(f"JSON parse error from {caller}: {parse_err}")
        return JSONResponse(
            make_jsonrpc_error(
                code=JSONRPCErrorCodes.PARSE_ERROR,
                message="Parse error: Invalid JSON payload.",
            ),
            status_code=400,
        )

    # Handle batch or single request
    if isinstance(payload, list):
        responses = [await handle_jsonrpc_request(item, caller) for item in payload]
        response_data = responses
    else:
        response_data = await handle_jsonrpc_request(payload, caller)

    # Record event in session history if active
    if session:
        session.add_event(event_name="message", data=json.dumps(response_data))

    return JSONResponse(response_data)


# ── Server-Sent Events (SSE) Stream Endpoint (GET /mcp or /sse) ───────────────

async def mcp_sse_endpoint(request: Request) -> StreamingResponse:
    """
    Handles GET SSE stream connections with Last-Event-ID resumability.
    """
    session = getattr(request.state, "session", None)
    session_id = getattr(request.state, "session_id", "default")
    caller = getattr(request.state, "caller_identity", "unknown")

    # Check for Last-Event-ID header or query parameter
    last_event_id_str = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
    last_event_id = 0
    if last_event_id_str:
        try:
            last_event_id = int(last_event_id_str)
        except ValueError:
            last_event_id = 0

    logger.info(f"SSE client connected | caller={caller} | session_id={session_id} | last_event_id={last_event_id}")

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        # 1. Initial connection & endpoint declaration event
        endpoint_uri = f"http://localhost:8100/mcp?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_uri}\n\n"

        # 2. Resumability: Replay any missed events if last_event_id was provided
        if last_event_id > 0 and session:
            missed_events = session.get_events_after(last_event_id)
            for event in missed_events:
                yield f"id: {event.event_id}\nevent: {event.event_name}\ndata: {event.data}\n\n"

        # 3. Persistent Stream Loop: Continuously yield events from session queue
        try:
            while True:
                try:
                    event = await asyncio.wait_for(session.queue.get(), timeout=15.0)
                    yield f"id: {event.event_id}\nevent: {event.event_name}\ndata: {event.data}\n\n"
                except asyncio.TimeoutError:
                    # SSE keepalive comment to maintain active connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            logger.info(f"SSE client disconnected | session_id={session_id}")

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Application Assembly ──────────────────────────────────────────────────────

routes = [
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/ready", endpoint=readiness_check, methods=["GET"]),
    Route("/mcp", endpoint=mcp_post_endpoint, methods=["POST"]),
    Route("/mcp", endpoint=mcp_sse_endpoint, methods=["GET"]),
    Route("/sse", endpoint=mcp_sse_endpoint, methods=["GET"]),
]

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(AuthMiddleware),
    ],
)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    logger.info(f"Starting Streamable MCP Server | host={cfg.server.host} | port={cfg.server.port}")
    logger.info(f"MCP POST/GET endpoint  → http://localhost:{cfg.server.port}/mcp")
    logger.info(f"Health check probe     → http://localhost:{cfg.server.port}/health")

    uvicorn.run(
        "src.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_config=None,  # We use our structured JSON logger
    )


if __name__ == "__main__":
    main()
