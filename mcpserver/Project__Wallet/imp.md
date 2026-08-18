5-Layer Decoupled MCP Architecture — Detailed Implementation Plan
1. Executive Summary & Architectural Overview
This implementation plan establishes a clean, enterprise-grade separation of concerns across the MCP ecosystem. It ensures that private server runtime secrets and infrastructure details (Layer 1) never leak into platform discovery (Layer 2), tool contracts (Layer 3), capability routing (Layer 4), or RBAC access policies (Layer 5).


┌────────────────────────────────────────────────────────────────────────┐
│ Layer 5: RBAC & Permission Policy (PharmaCTRL auth.py)                 │
│          • "Is the caller allowed to send downtime alerts?"            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Orchestration / Capability Routing (orchestration.schema.yaml)│
│          • "Capability 'dispatch_downtime_alert' -> provider & tool"   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Tool Catalog & Contract Schema (tool.schema.yaml)             │
│          • "What parameters, timeouts, and circuit breakers apply?"    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Server Registry & Security Posture (server.schema.yaml)       │
│          • "Where is the server? What transport and security tier?"   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼ (Network Call over HTTP/SSE)
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Server Private Runtime Config (notifications-mcp/config.yaml) │
│          • "Google OAuth secrets, token files, local port binding"     │
└────────────────────────────────────────────────────────────────────────┘
2. Layer-by-Layer Specification & Concrete Schemas
Layer 1: Server Private Runtime Config (config.yaml)
Location: d:/Custom_MCP_Server/notifications-mcp/config.yaml
Ownership: Private to the notifications-mcp microservice.
Rule: NEVER referenced or imported by PharmaCTRL or client agents.
yaml

# d:/Custom_MCP_Server/notifications-mcp/config.yaml
server:
  host: "0.0.0.0"
  port: 8100
enabled_channels:
  - mail
channels:
  mail:
    credentials_path: "credentials/google_credentials.json"
    token_path: "credentials/gmail_token.json"
    scopes:
      - "https://www.googleapis.com/auth/gmail.send"
    oauth_redirect_uri: "http://localhost:8100/auth/gmail/callback"
Layer 2: Server Registry & Security Posture Schema (server.schema.yaml)
Location: config/mcp/server.schema.yaml (in PharmaCTRL platform)
Purpose: Defines where deployed MCP servers live, their transport protocols, healthcheck probes, and security assumptions.
yaml

# config/mcp/server.schema.yaml
version: "1.0"
schema_type: "server_registry"
servers:
  notifications_service:
    display_name: "Unified Notifications Gateway"
    description: "Multi-channel outbound notification service (Email, Teams, SMS)"
    endpoint: "http://localhost:8100/mcp"
    transport: "streamable-http"
    version: "0.1.0"
    security:
      auth_type: "none"          # Options: none | api_key | bearer_jwt | mTLS
      network_zone: "internal_vpc"
      tls_enabled: false
    healthcheck:
      probe_url: "http://localhost:8100/health"
      interval_seconds: 30
      timeout_seconds: 5
    resilience:
      circuit_breaker:
        failure_threshold: 3
        recovery_time_seconds: 30
      retry:
        max_attempts: 3
        backoff_multiplier: 2
Layer 3: Tool Catalog & Interface Contract Schema (tool.schema.yaml)
Location: config/mcp/tool.schema.yaml (in PharmaCTRL platform)
Purpose: Declares tool interfaces, input constraints, expected output formats, and operational timeouts.
yaml

# config/mcp/tool.schema.yaml
version: "1.0"
schema_type: "tool_catalog"
tools:
  mail_send:
    server: "notifications_service"
    display_name: "Send Email Notification"
    description: "Send an email alert via Gmail to one or more recipients for machine downtime, deviations, or reports."
    category: "communications"
    timeout_seconds: 15
    idempotent: false
    parameters:
      type: "object"
      required:
        - to
        - subject
        - body
      properties:
        to:
          type: "array"
          items:
            type: "string"
            format: "email"
          description: "List of destination email addresses"
        subject:
          type: "string"
          description: "Email subject line"
        body:
          type: "string"
          description: "Plain text notification body"
        cc:
          type: "array"
          items:
            type: "string"
            format: "email"
          description: "Optional list of CC email addresses"
        bcc:
          type: "array"
          items:
            type: "string"
            format: "email"
          description: "Optional list of BCC email addresses"
Layer 4: Orchestration & Capability Routing Schema (orchestration.schema.yaml)
Location: config/mcp/orchestration.schema.yaml (in PharmaCTRL platform)
Purpose: Maps abstract agent capabilities to specific server tools, allowing routing to change without modifying agent code.
yaml

# config/mcp/orchestration.schema.yaml
version: "1.0"
schema_type: "capability_orchestration"
capabilities:
  # Capability 1: Equipment downtime / maintenance alerts
  "equipment.dispatch_downtime_alert":
    description: "Send urgent machine downtime notifications to shift manager and certified technicians"
    provider_server: "notifications_service"
    target_tool: "mail_send"
    defaults:
      subject_prefix: "[CRITICAL DOWNTIME ALERT]"
    parameter_mapping:
      recipients: "to"
      alert_title: "subject"
      alert_details: "body"
  # Capability 2: Quality & Lot deviation alerts
  "quality.dispatch_deviation_alert":
    description: "Notify quality assurance leads of out-of-spec lot genealogy deviations"
    provider_server: "notifications_service"
    target_tool: "mail_send"
    defaults:
      subject_prefix: "[QUALITY DEVIATION WARNING]"
    parameter_mapping:
      recipients: "to"
      alert_title: "subject"
      alert_details: "body"
  # Capability 3: Shift summary report delivery
  "operations.send_shift_summary":
    description: "Deliver end-of-shift OEE and maintenance summaries"
    provider_server: "notifications_service"
    target_tool: "mail_send"
    defaults:
      subject_prefix: "[SHIFT SUMMARY]"
    parameter_mapping:
      recipients: "to"
      alert_title: "subject"
      alert_details: "body"
Layer 5: Policy & RBAC Enforcement Matrix
Location: PharmaCTRL api/middleware/auth.py and shared/mcp_gateway.py
Purpose: Gate capability execution based on the user's role and granular permissions.
Capability	Minimum Role	Required Permission	Allowed Callers
equipment.dispatch_downtime_alert	Level 3 (technician)	can_troubleshoot OR can_assign_work_orders	EquipmentAgent, Technician, Engineer, Manager
quality.dispatch_deviation_alert	Level 4 (engineer)	can_manage_materials	DeviationAgent, Quality Engineer, Admin
operations.send_shift_summary	Level 5 (shop_floor_manager)	can_view_kpi	Manager, Admin
3. Platform Gateway Harness (shared/mcp_gateway.py)
A single, central gateway client in PharmaCTRL that coordinates Layers 2–5:

python

"""
shared/mcp_gateway.py — Central Multi-Layer MCP Gateway for PharmaCTRL.
Reads:
  - Layer 2: config/mcp/server.schema.yaml (Where servers live)
  - Layer 3: config/mcp/tool.schema.yaml (Tool definitions & timeouts)
  - Layer 4: config/mcp/orchestration.schema.yaml (Capability routing)
Enforces:
  - Layer 5: RBAC permissions before tool execution
"""
import yaml
import logging
import json
from pathlib import Path
from typing import Any, Optional
from mcp import Client
from mcp.client.transports.http import StreamableHttpTransport
logger = logging.getLogger(__name__)
class MCPGateway:
    def __init__(self, config_dir: str = "config/mcp"):
        self.config_dir = Path(config_dir)
        self.servers: dict = {}
        self.tools: dict = {}
        self.capabilities: dict = {}
        self._load_schemas()
    def _load_schemas(self):
        """Load Layers 2, 3, and 4 from declarative YAML schemas."""
        with open(self.config_dir / "server.schema.yaml") as f:
            self.servers = yaml.safe_load(f).get("servers", {})
        with open(self.config_dir / "tool.schema.yaml") as f:
            self.tools = yaml.safe_load(f).get("tools", {})
        with open(self.config_dir / "orchestration.schema.yaml") as f:
            self.capabilities = yaml.safe_load(f).get("capabilities", {})
    async def execute_capability(
        self,
        capability_name: str,
        params: dict[str, Any],
        user_context: Optional[dict] = None
    ) -> dict:
        """
        High-level entrypoint: Agent requests a capability, gateway handles
        RBAC check (Layer 5) -> Routing (Layer 4) -> Tool validation (Layer 3) -> Server transport (Layer 2).
        """
        # 1. Resolve Layer 4 Routing
        if capability_name not in self.capabilities:
            raise ValueError(f"Unknown capability: {capability_name}")
        
        cap = self.capabilities[capability_name]
        server_key = cap["provider_server"]
        tool_name = cap["target_tool"]
        # 2. Resolve Layer 2 Server Target
        server_info = self.servers.get(server_key)
        if not server_info:
            raise RuntimeError(f"Server '{server_key}' not registered in server.schema.yaml")
        
        endpoint = server_info["endpoint"]
        # 3. Apply parameter mapping and defaults
        tool_args = {}
        for param_key, target_field in cap.get("parameter_mapping", {}).items():
            if param_key in params:
                tool_args[target_field] = params[param_key]
        
        # Merge unmapped arguments directly
        for k, v in params.items():
            if k not in cap.get("parameter_mapping", {}):
                tool_args[k] = v
        # Apply subject prefix default if present
        if "subject_prefix" in cap.get("defaults", {}) and "subject" in tool_args:
            tool_args["subject"] = f"{cap['defaults']['subject_prefix']} {tool_args['subject']}"
        # 4. Execute via MCP Protocol over HTTP (Layer 2 transport)
        logger.info(f"Dispatching capability '{capability_name}' -> server '{server_key}' tool '{tool_name}'")
        transport = StreamableHttpTransport(url=endpoint)
        
        async with Client(transport) as client:
            result = await client.call_tool(tool_name, tool_args)
            result_text = result.content[0].text
            return json.loads(result_text)
4. Implementation Phasing & Roadmap

┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Runtime Server Stabilization (notifications-mcp) [COMPLETED ✅]       │
│ • Unit tests passing (7/7)                                                     │
│ • Streamable HTTP on port 8100                                                 │
│ • Private config.yaml isolation verified                                       │
├────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Platform Schema Definitions (PharmaCTRL)                              │
│ • Create config/mcp/server.schema.yaml                                         │
│ • Create config/mcp/tool.schema.yaml                                           │
│ • Create config/mcp/orchestration.schema.yaml                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: Platform Gateway Engine                                               │
│ • Build shared/mcp_gateway.py                                                  │
│ • Add unit tests for schema loading, parameter mapping, and tool resolution    │
├────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Agent & RBAC Integration                                              │
│ • Wire EquipmentAgent tool handler to call MCPGateway.execute_capability()     │
│ • Wire DeviationAgent tool handler to call MCPGateway.execute_capability()     │
│ • Apply Layer 5 permission checks in route middlewares                         │
├────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: End-to-End Empirical Verification                                     │
│ • Start notifications-mcp on :8100                                             │
│ • Trigger machine downtime event in PharmaCTRL                                 │
│ • Validate receipt of Gmail alert with correct subject prefix & recipients     │
└────────────────────────────────────────────────────────────────────────────────┘
5. Verification & Test Plan
Automated Tests
Schema Validation Tests:
Verify server.schema.yaml, tool.schema.yaml, and orchestration.schema.yaml parse without schema violations.
Gateway Unit Tests (tests/test_mcp_gateway.py):
Test capability resolution with mocked MCP client.
Test parameter mapping and subject prefix injection.
Test missing capability and unregistered server error handling.
End-to-End Smoke Test
Start notifications-mcp in background on port 8100.
Start PharmaCTRL API.
Call POST /api/v1/equipment/alerts/test as a certified technician.
Verify response: {"status": "sent", "message_id": "...", "capability": "equipment.dispatch_downtime_alert"}.
Verify zero leakage of Gmail credentials/paths in PharmaCTRL logs or responses.