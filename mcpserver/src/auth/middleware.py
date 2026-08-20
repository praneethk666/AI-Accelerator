"""
middleware.py — Authentication, Origin validation, and Session validation layer.
"""

import logging
import uuid
from typing import Callable, Optional, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.auth.session_manager import session_manager
from src.common.errors import JSONRPCErrorCodes, make_jsonrpc_error
from src.config import load_config

logger = logging.getLogger(__name__)


def extract_token(request: Request) -> Optional[str]:
    """Extracts bearer token from Authorization header, X-API-Key, or query param."""
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()

    # Query param fallback (useful for SSE EventSource connections in browsers)
    query_token = request.query_params.get("token")
    if query_token:
        return query_token.strip()

    return None


def validate_origin(origin: Optional[str], allowed_origins: list[str]) -> bool:
    """Validates the Origin header against configured allowed origins."""
    if not origin:
        return True  # Non-browser clients / CLI tools do not send Origin
    if "*" in allowed_origins:
        return True
    return origin in allowed_origins


class AuthMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware enforcing:
    1. Origin header checks
    2. Bearer token authentication
    3. Session identity binding
    4. Clean generic JSON-RPC error responses (no stack trace leaks)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Allow open access to health and readiness probes, favicon, and Web UI
        if request.url.path in ("/", "/ui", "/health", "/ready", "/healthz", "/favicon.ico"):
            return await call_next(request)

        # Handle CORS preflight OPTIONS requests cleanly
        if request.method == "OPTIONS":
            return Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, Mcp-Session-Id, Last-Event-ID, X-API-Key",
                },
            )

        cfg = load_config()
        correlation_id = uuid.uuid4().hex
        request.state.correlation_id = correlation_id

        # 1. Origin header validation
        origin = request.headers.get("origin")
        if not validate_origin(origin, cfg.server.allowed_origins):
            logger.warning(f"Rejected request with disallowed Origin: {origin}")
            return JSONResponse(
                make_jsonrpc_error(
                    code=JSONRPCErrorCodes.FORBIDDEN,
                    message="Forbidden: Origin not allowed",
                ),
                status_code=403,
            )

        # 2. Authentication check
        if cfg.auth.enabled:
            token = extract_token(request)
            if not token:
                logger.warning(
                    f"Unauthorized access attempt to {request.url.path} | token_provided=False",
                    extra={"event_type": "authentication_failure", "reason": "missing_token", "correlation_id": correlation_id}
                )
                return JSONResponse(
                    make_jsonrpc_error(
                        code=JSONRPCErrorCodes.UNAUTHORIZED,
                        message="Unauthorized: Missing authentication credential.",
                    ),
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )

            import src.globals as g
            
            # If JWT is enabled in config, we strictly validate it
            if cfg.auth.jwt and cfg.auth.jwt.enabled:
                from src.auth.jwt_validator import default_jwt_validator
                
                auth_ctx = default_jwt_validator.validate(token)
                if not auth_ctx:
                    logger.warning(
                        f"Unauthorized access attempt to {request.url.path} | invalid JWT",
                        extra={"event_type": "authentication_failure", "reason": "invalid_jwt", "correlation_id": correlation_id}
                    )
                    return JSONResponse(
                        make_jsonrpc_error(
                            code=JSONRPCErrorCodes.UNAUTHORIZED,
                            message="Unauthorized: Invalid or expired JWT.",
                        ),
                        status_code=401,
                    )
                
                subject = auth_ctx.client_id
            else:
                # Fallback purely for legacy demo/tests mode if JWT not enabled
                subject = token

            if g.identity_service:
                identity = g.identity_service.resolve(subject)
                if not identity:
                    logger.warning(
                        f"Unauthorized access attempt to {request.url.path} | no identity for subject '{subject}'",
                        extra={"event_type": "authentication_failure", "reason": "identity_not_found", "subject": subject, "correlation_id": correlation_id}
                    )
                    return JSONResponse(
                        make_jsonrpc_error(
                            code=JSONRPCErrorCodes.UNAUTHORIZED,
                            message="Unauthorized: Invalid authentication credentials.",
                        ),
                        status_code=401,
                    )
                caller_identity = identity["agentId"]
                request.state.identity = identity
            else:
                # Fallback for tests if identity service is not loaded
                if subject not in cfg.auth.tokens:
                    logger.warning(
                        f"Unauthorized access attempt to {request.url.path} | invalid token/subject",
                        extra={"event_type": "authentication_failure", "reason": "invalid_subject", "subject": subject, "correlation_id": correlation_id}
                    )
                    return JSONResponse(
                        make_jsonrpc_error(
                            code=JSONRPCErrorCodes.UNAUTHORIZED,
                            message="Unauthorized: Missing or invalid authentication credentials.",
                        ),
                        status_code=401,
                    )
                caller_identity = cfg.auth.tokens[subject]
                request.state.identity = {"agentId": caller_identity}

            logger.info(
                f"Authentication successful for {caller_identity}",
                extra={"event_type": "authentication_success", "subject": subject, "correlation_id": correlation_id}
            )

            request.state.caller_identity = caller_identity
            request.state.caller_token = token
        else:
            caller_identity = "anonymous_caller"
            request.state.caller_identity = caller_identity
            request.state.caller_token = None
            request.state.identity = {"agentId": "anonymous_caller"}

        # 3. Session validation and identity binding
        session_id = request.headers.get("mcp-session-id") or request.query_params.get("session_id")
        if session_id:
            valid, session, err_msg = session_manager.validate_or_bind_session(
                session_id=session_id,
                caller_identity=caller_identity,
            )
            if not valid:
                logger.warning(
                    f"Session identity mismatch | session_id={session_id} | "
                    f"caller={caller_identity} | reason={err_msg}"
                )
                return JSONResponse(
                    make_jsonrpc_error(
                        code=JSONRPCErrorCodes.SESSION_MISMATCH,
                        message="Forbidden: Session ID belongs to another caller identity.",
                    ),
                    status_code=403,
                )
            request.state.session = session
            request.state.session_id = session.session_id
        else:
            # Create a new session for this caller
            new_session = session_manager.create_session(caller_identity=caller_identity)
            request.state.session = new_session
            request.state.session_id = new_session.session_id

        # 4. Process the request through downstream handlers
        try:
            response = await call_next(request)
            # Ensure Mcp-Session-Id header is echoed back on every response
            response.headers["Mcp-Session-Id"] = request.state.session_id
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id, Last-Event-ID"
            return response
        except Exception as exc:
            logger.error(
                f"Unhandled server error processing {request.url.path}: {exc}",
                exc_info=True,
            )
            # Mask internal error — do not leak stack traces
            return JSONResponse(
                make_jsonrpc_error(
                    code=JSONRPCErrorCodes.INTERNAL_ERROR,
                    message="Internal server error. Please retry later.",
                ),
                status_code=500,
            )
