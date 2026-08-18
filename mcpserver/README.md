# Streamable HTTP + SSE MCP Server

> **Production-Ready Model Context Protocol (MCP) Server** with **Streamable HTTP + SSE**, **Bearer Authentication**, **Strict Session-Identity Binding**, **Prompt Injection Guardrails**, **Recipient Allowlisting**, and **Rate Limiting**.

---

## 👥 Work Split & Responsibilities

| Contributor | Area & Deliverables |
| :--- | :--- |
| **@Vinod** | Server scaffolding, Streamable HTTP (`POST` + `GET SSE`) transport, `Mcp-Session-Id` session management, `Last-Event-ID` resumability, hosting & tunnel setup (`scripts/run_tunnel.py`), Bearer token authentication, Origin checks, and JSON-RPC error-handling layer. |
| **@Vishal** | Tool handlers (`get_current_datetime`, `send_email`), input validation & Pydantic schemas, prompt injection detection & audit logging, recipient allowlist & sliding-window rate limiting, and 6 Live Demo verification test cases (`scripts/test_demo_cases.py`). |

---

## 🌟 Core Features & Standards

1. **Official MCP Specification Standard**: JSON-RPC 2.0 transport over Streamable HTTP (`POST /mcp` and `GET /mcp` SSE).
2. **Session Security & Resumability**:
   - `Mcp-Session-Id`: UUID sessions strictly tied to authenticated caller identities.
   - `Last-Event-ID`: SSE event ring buffer for event replay upon reconnect.
3. **Security Guardrails**:
   - **Auth Enforcement**: Every `/mcp` request requires a valid Bearer token or API key.
   - **Session Hijacking Defense**: Session IDs cannot be shared across different identities.
   - **Prompt Injection Defense**: Regex and heuristic analyzer scanning email fields with security audit logging.
   - **Recipient Allowlist & Rate Limits**: Prevents unauthorized email exfiltration and flooding.
   - **Clean Error Sanitization**: Generic, safe JSON-RPC errors (no stack trace leaks).
4. **Mock Tools**:
   - `get_current_datetime(timezone?)`: System date/time with IANA timezone validation.
   - `send_email(to, subject, body)`: Server SMTP with safe simulation demo mode.

---

## 📁 Repository Structure

```
Custom_MCP_Server/
├── config.yaml                    # Server, auth tokens, allowlists, rate limits, and SMTP config
├── pyproject.toml                 # Package definition & dependencies
├── requirements.txt               # Dependencies
├── README.md                      # Documentation & demo checklist
├── scripts/
│   ├── run_tunnel.py              # Hosting / tunnel launcher (Cloudflared, Ngrok, Localtunnel, Local)
│   └── test_demo_cases.py         # Automated verification suite for the 6 Live Demo test cases
├── src/
│   ├── __init__.py
│   ├── config.py                  # Typed configuration models
│   ├── server.py                  # Starlette Streamable HTTP + SSE application
│   ├── registry.py                # MCP tools registration & dispatcher
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── middleware.py          # Auth, Origin, and session validation middleware
│   │   └── session_manager.py     # UUID session manager with Last-Event-ID buffer
│   ├── security/
│   │   ├── __init__.py
│   │   ├── injection_detector.py  # Prompt injection detector & audit logger
│   │   ├── allowlist.py           # Email recipient & domain allowlist validator
│   │   └── rate_limiter.py        # Sliding-window rate limiter per caller
│   ├── common/
│   │   ├── __init__.py
│   │   ├── errors.py              # JSON-RPC 2.0 error codes and builders
│   │   └── logging.py             # Structured JSON logger with security tags
│   └── tools/
│       ├── __init__.py
│       ├── datetime_tool.py       # get_current_datetime tool handler
│       └── email_tool.py          # send_email tool handler with SMTP/demo mode
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_datetime_tool.py      # Datetime unit tests
    ├── test_email_tool.py         # Email & security guardrail tests
    ├── test_security_auth.py      # Auth & session isolation tests
    └── test_streamable_http.py    # Streamable HTTP / SSE transport tests
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js (optional, for MCP Inspector)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Server
```bash
python -m src.server
```
You will see:
```json
{"timestamp": "...", "level": "INFO", "message": "Starting Streamable MCP Server | host=0.0.0.0 | port=8100"}
{"timestamp": "...", "level": "INFO", "message": "MCP POST/GET endpoint  → http://localhost:8100/mcp"}
{"timestamp": "...", "level": "INFO", "message": "Health check probe     → http://localhost:8100/health"}
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8100
  allowed_origins:
    - "http://localhost:8100"
    - "*"

auth:
  enabled: true
  tokens:
    "agent-token-alpha": "agent_alpha"
    "agent-token-beta": "agent_beta"
    "vishal-test-token": "vishal_engineer"
    "vinod-test-token": "vinod_engineer"

security:
  email:
    allowlist:
      - "ops@company.com"
      - "manager@company.com"
      - "vishal@company.com"
      - "vinod@company.com"
      - "*@trusteddomain.com"
    rate_limit:
      max_calls: 10
      window_seconds: 60
    prompt_injection_guard:
      enabled: true

smtp:
  mode: "simulation"  # "simulation" (safe for testing/demos) or "smtp" (live server)
  host: "smtp.example.com"
  port: 587
  sender_address: "notifications@company.com"
```

---

## 🛠️ Tool Schemas

### Tool 1: `get_current_datetime`
- **Description:** Returns the current host system date and time with optional IANA timezone conversion.
- **Parameters:**
  ```json
  {
    "type": "object",
    "properties": {
      "timezone": {
        "type": "string",
        "description": "Optional IANA timezone name (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata')."
      }
    },
    "required": []
  }
  ```

### Tool 2: `send_email`
- **Description:** Sends an email notification via server SMTP with strict security guardrails.
- **Parameters:**
  ```json
  {
    "type": "object",
    "properties": {
      "to": {
        "type": "string",
        "description": "Recipient email address. Must be in the authorized allowlist."
      },
      "subject": {
        "type": "string",
        "description": "Email subject line."
      },
      "body": {
        "type": "string",
        "description": "Email body content."
      }
    },
    "required": ["to", "subject", "body"]
  }
  ```

---

## 🔌 Connecting with MCP Inspector & cURL

### Connect via MCP Inspector:
```bash
npx @modelcontextprotocol/inspector
```
Connect to URL: `http://localhost:8100/mcp` with Custom Header: `Authorization: Bearer vishal-test-token`.

### Sample cURL Invocations:

**1. Call `get_current_datetime`:**
```bash
curl -X POST http://localhost:8100/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vishal-test-token" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "get_current_datetime",
      "arguments": {"timezone": "Asia/Kolkata"}
    }
  }'
```

**2. Call `send_email`:**
```bash
curl -X POST http://localhost:8100/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vishal-test-token" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "send_email",
      "arguments": {
        "to": "ops@company.com",
        "subject": "System Status",
        "body": "All services operating normally."
      }
    }
  }'
```

---

## 🌐 Hosting & Tunnel Setup

To expose the MCP server to a public URL / cloud demo:

```bash
# Option 1: Direct Local / VM binding
python scripts/run_tunnel.py --mode local --port 8100

# Option 2: Cloudflare Tunnel (Free, no port-forwarding needed)
python scripts/run_tunnel.py --mode cloudflared --port 8100

# Option 3: Ngrok
python scripts/run_tunnel.py --mode ngrok --port 8100

# Option 4: Localtunnel
python scripts/run_tunnel.py --mode localtunnel --port 8100
```

---

## 🧪 Live Demo Checklist (6 Test Cases)

Run the automated test runner to verify all 6 demo cases in one command:

```bash
python scripts/test_demo_cases.py
```

### Or run with pytest:
```bash
pytest -v
```

### Verified Test Cases:
- [x] **Case 1: Authorized call to each tool succeeds** (both `get_current_datetime` and `send_email`).
- [x] **Case 2: Unauthorized caller gets a generic error** (HTTP 401, error code `-32001`).
- [x] **Case 3: Malformed input returns a clean validation error** (invalid timezone or missing schema fields).
- [x] **Case 4: Prompt injection attempts in email are blocked and logged** (e.g. system prompt overrides).
- [x] **Case 5: Sending to non-allowlisted email is rejected** (blocks external/unauthorized addresses).
- [x] **Case 6: Session ID reuse across different identities is denied** (HTTP 403, error code `-32005`).
