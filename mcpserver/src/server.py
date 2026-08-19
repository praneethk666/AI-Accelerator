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
import os
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

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
from src.registry import execute_tool, get_tool_definitions, global_tool_registry
from src.config_schema_loader import ConfigLoader
from src.server_registry import ServerRegistry
import src.globals as g
import contextlib

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

async def handle_jsonrpc_request(payload: Dict[str, Any], caller: str, identity: Dict[str, Any] = None) -> Dict[str, Any]:
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

    # 4. tools/list (Filtered by caller RBAC permissions)
    if method == "tools/list":
        all_tools = get_tool_definitions()
        
        # In Phase 2, we can filter visibly by role allowlist and agent allowlist.
        # But wait, we evaluate via PolicyEngine per tool to see if they can use it.
        if g.policy_engine and identity and caller != "anonymous_caller":
            visible_tools = []
            for t in all_tools:
                # tools/list resolution logic is: only show tools that evaluate to ALLOW or REQUIRE_APPROVAL
                # We need to map tool to a capability/server? 
                # If they are remote tools, they are registered with names. Local tools evaluate on 'local'.
                # For discovery, let's allow all that are not explicitly DENY.
                res = g.policy_engine.evaluate(identity, "local", t["name"])
                if res != "DENY":
                    visible_tools.append(t)
        else:
            # Fallback legacy behavior
            cfg = load_config()
            caller_perms = cfg.auth.permissions.get(caller, ["*"]) if cfg.auth.enabled else ["*"]
            if "*" in caller_perms:
                visible_tools = all_tools
            else:
                visible_tools = [t for t in all_tools if t["name"] in caller_perms]

        return make_jsonrpc_success(
            result={"tools": visible_tools},
            req_id=req_id,
        )

    # 5. tools/call (Enforce RBAC execution permissions)
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if not tool_name:
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INVALID_PARAMS,
                message="Invalid params: 'name' is required for tools/call.",
                req_id=req_id,
            )

        target_server = "local"
        target_tool = tool_name
        
        if g.orchestration_resolver:
            target_server, target_tool = g.orchestration_resolver.resolve(tool_name)

        if g.policy_engine and identity and caller != "anonymous_caller":
            policy_res = g.policy_engine.evaluate(identity, target_server, target_tool)
            if policy_res == "DENY":
                logger.warning(
                    f"Policy authorization denied | agent={identity.get('agentId')} | target_server={target_server} | tool={target_tool}"
                )
                return make_jsonrpc_error(
                    code=JSONRPCErrorCodes.FORBIDDEN,
                    message=f"Forbidden: Agent identity '{identity.get('agentId')}' is not authorized to execute capability '{tool_name}' (resolved to '{target_tool}' on '{target_server}').",
                    req_id=req_id,
                )
        else:
            # Enforce legacy caller RBAC authorization
            cfg = load_config()
            if cfg.auth.enabled:
                caller_perms = cfg.auth.permissions.get(caller, ["*"])
                if "*" not in caller_perms and tool_name not in caller_perms:
                    logger.warning(
                        f"RBAC authorization denied | caller={caller} | tool={tool_name} | allowed={caller_perms}"
                    )
                    return make_jsonrpc_error(
                        code=JSONRPCErrorCodes.FORBIDDEN,
                        message=f"Forbidden: Caller identity '{caller}' is not authorized to execute tool '{tool_name}'.",
                        req_id=req_id,
                    )

        try:
            # We proxy execution to `execute_tool`. `execute_tool` handles remote proxying 
            # if `target_server` != local? Wait, `execute_tool` in `ToolRegistry` relies on tools being 
            # prefixed or handles them internally. 
            # But the OrchestrationResolver already tells us the physical tool name. Let's pass the 
            # physical tool to `execute_tool`.
            result = await execute_tool(tool_name=target_tool, arguments=arguments, caller=caller)
            
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

    # 6. agent_request (New Phase 4 Dynamic Agent Entrypoint)
    if method == "agent_request":
        user_request = params.get("request")
        approval_step_id = params.get("approval_step_id")
        
        if not user_request:
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INVALID_PARAMS,
                message="Invalid params: 'request' is required for agent_request.",
                req_id=req_id,
            )
            
        try:
            role = "admin"
            if identity:
                role = identity.get("profile", {}).get("role", "admin")
                
            state = await g.agent_runtime.process_request(
                request_id=str(req_id), 
                request=user_request, 
                user_identity=identity or {}, 
                user_approval_for_step=approval_step_id
            )
            
            return make_jsonrpc_success(
                result={
                    "state": state.model_dump()
                },
                req_id=req_id,
            )
            
        except Exception as exc:
            logger.error(f"Error in agent_request phase: {exc}", exc_info=True)
            return make_jsonrpc_error(
                code=JSONRPCErrorCodes.INTERNAL_ERROR,
                message=f"Agent workflow execution failed: {str(exc)}",
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

    identity = getattr(request.state, "identity", None)

    # Handle batch or single request
    if isinstance(payload, list):
        responses = [await handle_jsonrpc_request(item, caller, identity) for item in payload]
        response_data = responses
    else:
        response_data = await handle_jsonrpc_request(payload, caller, identity)

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
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
        },
    )


# ── Web UI Endpoint ───────────────────────────────────────────────────────────

from starlette.responses import HTMLResponse
from pathlib import Path

async def serve_ui(request: Request) -> Response:
    """Serves the Gemini AI + MCP Chat Web Interface."""
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if ui_path.exists():
        content = ui_path.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse("<h1>Gemini + MCP Web UI</h1><p>ui/index.html not found.</p>", status_code=404)


# ── Application Assembly ──────────────────────────────────────────────────────

from starlette.middleware.cors import CORSMiddleware

@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    logger.info("Initializing MCP Gateway Configuration...")
    try:
        config_loader = ConfigLoader(config_dir=".", schema_dir=".")
        configs = config_loader.load_and_validate()
        
        from src.security.identity_service import IdentityService
        from src.security.policy_engine import PolicyEngine
        from src.orchestration.resolver import OrchestrationResolver
        
        from src.agent.runtime import AgentRuntime
        from src.agent.planner import Planner
        from src.agent.decision import DecisionEngine
        from src.agent.execution import ExecutionEngine
        from src.agent.evaluator import ResultEvaluator
        from src.security.audit import AuditLogger
        from src.agent.llm import GroqClient

        g.identity_service = IdentityService(configs.get("agents", {}), configs.get("roles", {}))
        g.policy_engine = PolicyEngine(configs.get("roles", {}))
        g.orchestration_resolver = OrchestrationResolver(configs.get("orchestration", {}))
        
        server_registry = ServerRegistry()
        server_registry.load_from_config(configs.get("servers", {}).get("servers", {}))
        
        global_tool_registry.set_server_registry(server_registry)
        
        from src.agent.llm import GroqClient, GeminiClient
        import os
        
        # Initialize Agent Runtime Components
        llm_client = GroqClient(model_name="qwen/qwen3.6-27b")
            
        planner = Planner(llm_client)
        decision_engine = DecisionEngine(llm_client, get_tool_definitions())
        
        # Async execution bridge
        async def run_tool(name: str, args: dict) -> str:
            val = await execute_tool(name, args, "System")
            return json.dumps(val)
            
        execution_engine = ExecutionEngine(run_tool)
        evaluator = ResultEvaluator()
        audit_logger = AuditLogger()
        
        def rbac_policy_checker(user_identity: dict, tool_name: str) -> str:
            res = g.policy_engine.evaluate(user_identity, "local", tool_name)
            return res
            
        g.agent_runtime = AgentRuntime(
            planner=planner,
            decision_engine=decision_engine,
            execution_engine=execution_engine,
            evaluator=evaluator,
            policy_checker=rbac_policy_checker,
            audit_logger=audit_logger,
            idempotent_tools=["get_current_datetime"]
        )
        
        await server_registry.initialize_all()
        for name, client in server_registry.clients.items():
            try:
                tools_response = await client.list_tools()
                tools = tools_response.get("tools", [])
                global_tool_registry.register_remote_tools(name, tools)
                logger.info(f"Registered {len(tools)} remote tools from server '{name}'.")
            except Exception as e:
                logger.error(f"Failed to fetch tools from '{name}': {e}")
    except Exception as e:
        logger.error(f"Failed to initialize remote servers: {e}")
        
    yield
    
    # Shutdown logic
    for name, client in server_registry.clients.items():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"Error closing client '{name}': {e}")

routes = [
    Route("/", endpoint=serve_ui, methods=["GET"]),
    Route("/ui", endpoint=serve_ui, methods=["GET"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/ready", endpoint=readiness_check, methods=["GET"]),
    Route("/mcp", endpoint=mcp_post_endpoint, methods=["POST"]),
    Route("/mcp", endpoint=mcp_sse_endpoint, methods=["GET"]),
    Route("/sse", endpoint=mcp_sse_endpoint, methods=["GET"]),
]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        ),
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
