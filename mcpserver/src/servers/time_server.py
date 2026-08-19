"""
time_server.py — Specialized Time & System MCP Server running on Port 8102.
Implements the Model Context Protocol (MCP) Streamable HTTP + SSE transport.

Tools:
1. get_current_datetime(timezone)
2. convert_timezone(time_str, from_tz, to_tz)
3. get_system_uptime()
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
import platform
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import zoneinfo
import uvicorn

from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from src.auth.middleware import AuthMiddleware
from src.auth.session_manager import session_manager
from src.common.errors import JSONRPCErrorCodes, make_jsonrpc_error, make_jsonrpc_success
from src.common.logging import setup_logging

setup_logging()
logger = logging.getLogger("TimeMCPServer")

TIME_MCP_PORT = 8102
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "time-system-mcp-server",
    "version": "1.0.0",
    "description": "Specialized Time & System Metrics MCP Server",
}

_SERVER_START_TIME = time.time()


# ── Pydantic Input Schemas ──────────────────────────────────────────────────

class GetCurrentDateTimeInput(BaseModel):
    timezone: Optional[str] = Field(
        None,
        description="Optional IANA timezone name (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata', 'Europe/London'). Defaults to host local time.",
    )


class ConvertTimezoneInput(BaseModel):
    time_str: str = Field(..., description="Timestamp or time string to convert (e.g. '2026-08-19 14:30:00' or '14:30').")
    from_tz: str = Field("UTC", description="Source timezone (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata').")
    to_tz: str = Field(..., description="Target timezone (e.g. 'Europe/London', 'Asia/Tokyo').")


class GetSystemUptimeInput(BaseModel):
    pass


# ── Tool Implementations ────────────────────────────────────────────────────

async def handle_get_current_datetime(data: GetCurrentDateTimeInput, caller: Optional[str] = None) -> dict:
    req_tz = (data.timezone or "").strip()

    if req_tz:
        try:
            tz = zoneinfo.ZoneInfo(req_tz)
            now = datetime.now(tz)
            tz_name = req_tz
        except Exception:
            # Fallback to UTC
            now = datetime.now(timezone.utc)
            tz_name = "UTC (fallback from invalid timezone)"
    else:
        # System local time
        now = datetime.now().astimezone()
        tz_name = str(now.tzinfo) or "System Local"

    return {
        "status": "success",
        "datetime_iso": now.isoformat(),
        "timezone": tz_name,
        "formatted": now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
        "epoch_timestamp": now.timestamp(),
        "server": "Time & System MCP Server (:8102)",
    }


TZ_ABBREVIATIONS: Dict[str, str] = {
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "IST": "Asia/Kolkata",
    "JST": "Asia/Tokyo",
    "GMT": "UTC",
    "UTC": "UTC",
    "BST": "Europe/London",
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
}

def resolve_tz(tz_name: Optional[str], default_name: str = "UTC") -> zoneinfo.ZoneInfo:
    if not tz_name or not tz_name.strip():
        tz_name = default_name
    cleaned = tz_name.strip()
    
    if cleaned.upper() in TZ_ABBREVIATIONS:
        cleaned = TZ_ABBREVIATIONS[cleaned.upper()]
        
    try:
        return zoneinfo.ZoneInfo(cleaned)
    except Exception:
        # Fallback entirely for Windows if IANA DB is missing
        if default_name == "UTC" or cleaned.upper() == "UTC":
            return timezone.utc
        return zoneinfo.ZoneInfo("UTC")


async def handle_convert_timezone(data: ConvertTimezoneInput, caller: Optional[str] = None) -> dict:
    try:
        from_zone = resolve_tz(data.from_tz, "UTC")
        to_zone = resolve_tz(data.to_tz, "UTC")

        # Parse common 12-hour AM/PM and 24-hour time formats
        time_formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %I:%M:%S %p",
            "%Y-%m-%d %I:%M %p",
            "%Y-%m-%dT%H:%M:%S",
            "%I:%M:%S %p",
            "%I:%M %p",
            "%I %p",
            "%H:%M:%S",
            "%H:%M",
        )
        parsed = None
        cleaned_time = data.time_str.strip()
        for fmt in time_formats:
            try:
                parsed = datetime.strptime(cleaned_time, fmt)
                if fmt in ("%I:%M:%S %p", "%I:%M %p", "%I %p", "%H:%M:%S", "%H:%M"):
                    today = datetime.now().date()
                    parsed = parsed.replace(year=today.year, month=today.month, day=today.day)
                break
            except ValueError:
                continue

        if not parsed:
            # Try ISO parse as final fallback
            parsed = datetime.fromisoformat(cleaned_time)

        # Localize and convert
        localized = parsed.replace(tzinfo=from_zone)
        converted = localized.astimezone(to_zone)

        return {
            "status": "success",
            "original_time": data.time_str,
            "from_timezone": data.from_tz,
            "to_timezone": data.to_tz,
            "converted_time_iso": converted.isoformat(),
            "converted_formatted": converted.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
            "server": "Time & System MCP Server (:8102)",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": "timezone_conversion_failed",
            "message": f"Could not convert time across timezones: {exc}",
        }


async def handle_get_system_uptime(data: GetSystemUptimeInput, caller: Optional[str] = None) -> dict:
    uptime_seconds = int(time.time() - _SERVER_START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "status": "success",
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": f"{hours}h {minutes}m {seconds}s",
        "system_platform": platform.platform(),
        "python_version": platform.python_version(),
        "host_node": platform.node(),
        "server_started_at": datetime.fromtimestamp(_SERVER_START_TIME, timezone.utc).isoformat(),
        "server": "Time & System MCP Server (:8102)",
    }


# ── Tool Definitions Registry ───────────────────────────────────────────────

TIME_TOOLS = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current live host system date and time with optional IANA timezone conversion (e.g. 'UTC', 'Asia/Kolkata', 'America/New_York').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Optional IANA timezone name (e.g. 'UTC', 'Asia/Kolkata', 'America/New_York', 'Europe/London')."},
            },
            "required": [],
        },
        "model_cls": GetCurrentDateTimeInput,
        "handler": handle_get_current_datetime,
    },
    {
        "name": "convert_timezone",
        "description": "Converts a specific time string or timestamp from one timezone to another target timezone.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_str": {"type": "string", "description": "Time string to convert (e.g. '2026-08-19 14:30:00' or '14:30')."},
                "from_tz": {"type": "string", "description": "Source timezone (e.g. 'America/New_York', 'UTC')."},
                "to_tz": {"type": "string", "description": "Destination timezone (e.g. 'Asia/Kolkata', 'Asia/Tokyo')."},
            },
            "required": ["time_str", "to_tz"],
        },
        "model_cls": ConvertTimezoneInput,
        "handler": handle_convert_timezone,
    },
    {
        "name": "get_system_uptime",
        "description": "Returns current host server uptime, OS platform, and diagnostic runtime metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "model_cls": GetSystemUptimeInput,
        "handler": handle_get_system_uptime,
    },
]


# ── JSON-RPC Request Router ─────────────────────────────────────────────────

async def handle_jsonrpc(payload: Dict[str, Any], caller: str) -> Dict[str, Any]:
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return make_jsonrpc_success(
            result={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}, "logging": {}},
                "serverInfo": SERVER_INFO,
            },
            req_id=req_id,
        )

    if method in ("ping", "notifications/initialized"):
        return make_jsonrpc_success(result={}, req_id=req_id)

    if method == "tools/list":
        return make_jsonrpc_success(
            result={
                "tools": [
                    {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                    for t in TIME_TOOLS
                ]
            },
            req_id=req_id,
        )

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments") or {}

        for tool_meta in TIME_TOOLS:
            if tool_meta["name"] == tool_name:
                try:
                    validated = tool_meta["model_cls"](**args)
                    res = await tool_meta["handler"](validated, caller=caller)
                    is_err = res.get("status") == "failed"
                    return make_jsonrpc_success(
                        result={
                            "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                            "isError": is_err,
                        },
                        req_id=req_id,
                    )
                except Exception as exc:
                    return make_jsonrpc_error(
                        code=JSONRPCErrorCodes.INVALID_PARAMS,
                        message=f"Execution error: {exc}",
                        req_id=req_id,
                    )

        return make_jsonrpc_error(
            code=JSONRPCErrorCodes.METHOD_NOT_FOUND,
            message=f"Tool '{tool_name}' not found on Time MCP Server.",
            req_id=req_id,
        )

    return make_jsonrpc_error(
        code=JSONRPCErrorCodes.METHOD_NOT_FOUND,
        message=f"Method '{method}' not found.",
        req_id=req_id,
    )


# ── Server Routes & App ──────────────────────────────────────────────────────

async def mcp_post(request: Request) -> JSONResponse:
    caller = getattr(request.state, "caller_identity", "anonymous_caller")
    session = getattr(request.state, "session", None)
    try:
        body = await request.body()
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return JSONResponse(make_jsonrpc_error(JSONRPCErrorCodes.PARSE_ERROR, "Invalid JSON payload"), status_code=400)

    if isinstance(payload, list):
        res = [await handle_jsonrpc(item, caller) for item in payload]
    else:
        res = await handle_jsonrpc(payload, caller)

    if session:
        session.add_event(event_name="message", data=json.dumps(res))
    return JSONResponse(res)


async def mcp_sse(request: Request) -> StreamingResponse:
    session = getattr(request.state, "session", None)
    session_id = getattr(request.state, "session_id", "default")
    caller = getattr(request.state, "caller_identity", "unknown")

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        endpoint_uri = f"http://localhost:{TIME_MCP_PORT}/mcp?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_uri}\n\n"
        try:
            while True:
                if session:
                    try:
                        event = await asyncio.wait_for(session.queue.get(), timeout=15.0)
                        yield f"id: {event.event_id}\nevent: {event.event_name}\ndata: {event.data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                else:
                    await asyncio.sleep(15.0)
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "server": "time-system-mcp-server",
        "port": TIME_MCP_PORT,
        "tools_count": len(TIME_TOOLS),
    })


routes = [
    Route("/health", endpoint=health, methods=["GET"]),
    Route("/mcp", endpoint=mcp_post, methods=["POST"]),
    Route("/mcp", endpoint=mcp_sse, methods=["GET"]),
    Route("/sse", endpoint=mcp_sse, methods=["GET"]),
]

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        Middleware(AuthMiddleware),
    ],
)


def start_server():
    logger.info(f"Starting Time & System MCP Server on http://0.0.0.0:{TIME_MCP_PORT}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=TIME_MCP_PORT, log_level="warning")


if __name__ == "__main__":
    start_server()
