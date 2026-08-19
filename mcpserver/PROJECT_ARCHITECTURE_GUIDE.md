# 🚀 Multi-Server Model Context Protocol (MCP) Architecture Guide

Welcome! This document provides a complete, easy-to-understand guide to the **Multi-Server MCP Architecture** and **Groq AI Agent Hub**.

---

## 📌 Table of Contents
1. [What is Model Context Protocol (MCP)?](#-what-is-model-context-protocol-mcp)
2. [High-Level Architecture Diagram](#-high-level-architecture-diagram)
3. [The 3 Microservice Servers](#-the-3-microservice-servers)
4. [Complete Catalog of the 7 MCP Tools](#-complete-catalog-of-the-7-mcp-tools)
5. [User Authentication & Access Control (RBAC)](#-user-authentication--access-control-rbac)
6. [Security Guardrails & Enterprise Protections](#-security-guardrails--enterprise-protections)
7. [How to Run & Test the Project](#-how-to-run--test-the-project)
8. [Connecting External MCP Clients (Inspector, Claude, Cursor)](#-connecting-external-mcp-clients)

---

## 🌟 What is Model Context Protocol (MCP)?

**Model Context Protocol (MCP)** is an open industry standard developed by Anthropic that allows AI models (like Groq, Claude, ChatGPT, Gemini) to securely discover and invoke local functions, databases, and APIs without exposing private server infrastructure.

Instead of hardcoding APIs into an AI prompt, MCP allows servers to expose **tools** via standard **JSON-RPC 2.0 protocol over HTTP and Server-Sent Events (SSE)**.

---

## 🗺️ High-Level Architecture Diagram

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                     User Input Prompt                       │
                        │        (e.g., "Convert 04:30 PM NYC to Kolkata time")       │
                        └──────────────────────────────┬──────────────────────────────┘
                                                       │
                                                       ▼
                        ┌─────────────────────────────────────────────────────────────┐
                        │             Groq LLM (Native Function Calling)             │
                        │        Evaluates all 7 Tool JSON-Schemas Dynamically        │
                        └──────────────┬──────────────────────────────┬───────────────┘
                                       │                              │
                    [Tool Call Required]                      [Direct Answer]
                                       │                              │
                                       ▼                              ▼
                 ┌───────────────────────────────────────┐   ┌────────────────────────┐
                 │     Dynamic Client-Side Dispatcher    │   │  Conversational Answer │
                 │   Routes tool to Port 8101 or 8102    │   │  (Cinema, Code, etc.)  │
                 └─────────────────┬─────────────────────┘   └────────────────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
     ┌───────────────────────────┐   ┌───────────────────────────┐
     │  Gmail MCP Server (:8101) │   │ Time & System MCP (:8102) │
     │  - send_email             │   │ - get_current_datetime    │
     │  - read_inbox             │   │ - convert_timezone        │
     │  - search_emails          │   │ - get_system_uptime       │
     │  - create_draft           │   └───────────────────────────┘
     └─────────────┬─────────────┘                 │
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────┐
                 │    Live JSON Result Received     │
                 └─────────────────┬────────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────┐
                 │       Agent Synthesis Pass       │
                 │   LLM synthesizes final answer   │
                 └─────────────────┬────────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────┐
                 │  Rendered to User in Web UI Hub  │
                 └──────────────────────────────────┘
```

---

## 🏛️ The 3 Microservice Servers

| Server Component | Port | Endpoint URL | Responsibilities |
|---|---|---|---|
| **🌐 AI Agent Hub & Gateway** | `8100` | `http://localhost:8100` | • Single-Page Web UI interface<br>• Master MCP Gateway consolidating all tools<br>• Session authentication & login portal |
| **📧 Gmail MCP Server** | `8101` | `http://localhost:8101/mcp` | • Real-time Gmail SMTP email dispatch<br>• IMAP inbox reader & keyword search<br>• Gmail draft creation |
| **🕒 Time & System MCP Server** | `8102` | `http://localhost:8102/mcp` | • Host hardware clock & ISO-8601 formatting<br>• Multi-timezone converter (12h/24h)<br>• Server uptime & OS diagnostic metrics |

---

## 🛠️ Complete Catalog of the 7 MCP Tools

### 📧 Domain 1: Gmail MCP Server (`Port 8101`)

#### 1. `send_email`
* **Purpose:** Dispatches an outgoing email via Gmail SMTP.
* **Input Parameters:**
  * `to` *(string, required)*: Recipient email address (e.g. `bonthumanoj999@gmail.com`).
  * `subject` *(string, required)*: Subject line of the email.
  * `body` *(string, required)*: Text body of the email.
* **Security:** Checked against the recipient allowlist, prompt-injection scanner, and rate limiter.

#### 2. `read_inbox`
* **Purpose:** Reads and summarizes the most recent incoming emails from your inbox.
* **Input Parameters:**
  * `max_results` *(integer, optional, default: 5)*: Number of emails to retrieve.
* **Output:** List of emails with Sender, Subject, Date, and message snippet.

#### 3. `search_emails`
* **Purpose:** Searches through inbox messages and drafts for specific keywords or senders.
* **Input Parameters:**
  * `query` *(string, required)*: Keyword or phrase to search (e.g. `"MCP Server"`).
* **Output:** List of matching emails and count of matches.

#### 4. `create_draft`
* **Purpose:** Prepares an email draft in Gmail without sending it immediately.
* **Input Parameters:**
  * `to` *(string, required)*: Target recipient.
  * `subject` *(string, required)*: Draft subject.
  * `body` *(string, required)*: Draft body.
* **Output:** Unique `draft_id` and confirmation timestamp.

---

### 🕒 Domain 2: Time & System MCP Server (`Port 8102`)

#### 5. `get_current_datetime`
* **Purpose:** Reads the exact live host operating system hardware clock.
* **Input Parameters:**
  * `timezone` *(string, optional)*: IANA timezone name (e.g. `"Asia/Kolkata"`, `"UTC"`, `"America/New_York"`, `"Asia/Tokyo"`). If omitted, defaults to host local time.
* **Output:** ISO-8601 timestamp, human-readable date/time string, and Unix epoch seconds.

#### 6. `convert_timezone`
* **Purpose:** Converts timestamps between international timezones.
* **Input Parameters:**
  * `time_str` *(string, required)*: Time to convert (supports `"04:30 PM"`, `"16:30"`, or `"2026-08-19 14:30:00"`).
  * `from_tz` *(string, default: "UTC")*: Source timezone (e.g. `"America/New_York"`).
  * `to_tz` *(string, required)*: Destination timezone (e.g. `"Asia/Kolkata"`).
* **Output:** Converted local time with timezone abbreviations (e.g. `02:00:00 AM IST next day`).

#### 7. `get_system_uptime`
* **Purpose:** Provides server health metrics, operating system version, and server runtime duration.
* **Input Parameters:** None (`{}`).
* **Output:** Formatted uptime (`Xh Ym Zs`), OS platform (Windows/Linux/macOS), and Python runtime version.

---

## 🔐 User Authentication & Access Control (RBAC)

The system enforces authentication using **Bearer Tokens** and pre-configured user credentials:

| Username | Password | Role / Identity | Permissions |
|---|---|---|---|
| **`vishal`** | `vishal123` | `vishal_engineer` | 🟢 **Full Admin** (All 7 Tools) |
| **`vinod`** | `vinod123` | `vinod_engineer` | 🟢 **Full Admin** (All 7 Tools) |
| **`admin`** | `admin123` | `admin_user` | 🟢 **Superuser** (All 7 Tools) |
| **`agent_alpha`** | `alpha123` | `agent_alpha` | 🟡 Standard Agent (`get_current_datetime`, `send_email`) |
| **`agent_beta`** | `beta123` | `agent_beta` | 🟠 Restricted Read-Only (`get_current_datetime`) |

---

## 🛡️ Security Guardrails & Enterprise Protections

1. **Security Allowlist**:
   * Emails can only be dispatched to explicitly verified email domains and addresses defined in `config.yaml` (`bonthumanoj999@gmail.com`, `vishalreddykonreddy@gmail.com`, `*@trusteddomain.com`).
   * Unlisted recipients are automatically blocked with a security alert.
2. **Prompt Injection Guard**:
   * Inspects outgoing email subject and body content for adversarial prompt injection signatures before dispatch.
3. **Sliding-Window Rate Limiter**:
   * Restricts callers to a maximum number of calls within a 60-second window to prevent Denial-of-Service (DoS).
4. **Dual-Port SMTP Fallback & Socket Timeouts**:
   * Automatically attempts Port 587 (TLS) and Port 465 (SSL) with a 5-second socket timeout.
   * If local ISP/firewall blocks live ports, it safely completes in verified simulation mode with zero thread hanging.
5. **Tamper-Evident SHA-256 Audit Trail**:
   * Records cryptographic transaction hashes in `audit.db` SQLite core.

---

## 🚀 How to Run & Test the Project

### 1. Launch All 3 Servers in One Command
Open your terminal in `Custom_MCP_Server` and run:

```bash
python run_servers.py
```

You will see the startup banner:
```
======================================================================
🚀 STARTING MULTI-MCP SERVER ARCHITECTURE
======================================================================
  🌐 Web UI & AI Agent Hub     → http://localhost:8100
  📧 Gmail MCP Server           → http://localhost:8101/mcp (Tools: send, read, search, draft)
  🕒 Time & System MCP Server   → http://localhost:8102/mcp (Tools: datetime, timezone, uptime)
======================================================================
```

### 2. Open the Web Hub in Your Browser
Visit **[http://localhost:8100](http://localhost:8100)** and log in with:
* **Username**: `vishal`
* **Password**: `vishal123`

### 3. Example Prompts to Try:
* 🕒 `What is the current time in Tokyo and London?`
* 🌐 `Convert 04:30 PM New York time to Asia/Kolkata timezone.`
* 💻 `Show me my server uptime and platform metrics.`
* 📧 `Send an email to bonthumanoj999@gmail.com stating that both Multi-MCP servers are operational.`
* 📝 `Create an email draft to vishalreddykonreddy@gmail.com with subject Project Update.`
* 📥 `Check and summarize my recent incoming emails from inbox.`
* 🔍 `Search my emails for keywords about MCP Server status.`

---

## 🔌 Connecting External MCP Clients

### Connect via Anthropic MCP Inspector:
Run the official inspector tool:
```bash
npx @modelcontextprotocol/inspector
```
Connect to any server port:
* **Unified Master Gateway**: `http://localhost:8100/mcp`
* **Gmail MCP Server**: `http://localhost:8101/mcp`
* **Time MCP Server**: `http://localhost:8102/mcp`

With Custom Header:
```http
Authorization: Bearer vishal-test-token
```

---

## 🧩 Developer Template Guide: Adding & Removing Tools/Servers

Our codebase includes a reusable boilerplate template in **[`src/servers/template_server.py`](file:///c:/Users/visha/OneDrive/Desktop/mcp/Custom_MCP_Server/src/servers/template_server.py)**. Any developer can add or remove tools and servers in minutes.

---

### 1️⃣ How to Add a New Tool (3 Simple Steps)

Open any server file (e.g. `src/servers/gmail_server.py` or `src/servers/time_server.py`):

#### Step 1: Define the Pydantic Input Model
```python
class WeatherCheckInput(BaseModel):
    city: str = Field(..., description="Name of the city (e.g. 'London').")
```

#### Step 2: Write the Async Handler Function
```python
async def handle_weather_check(data: WeatherCheckInput, caller: Optional[str] = None) -> dict:
    return {
        "status": "success",
        "city": data.city,
        "temperature": "22°C",
        "condition": "Partly Cloudy",
    }
```

#### Step 3: Register in the `TOOLS` List
```python
CUSTOM_TOOLS.append({
    "name": "get_weather",
    "description": "Checks the current weather for a specified city.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
        },
        "required": ["city"],
    },
    "handler": handle_weather_check,
    "model_cls": WeatherCheckInput,
})
```
*The Web UI Hub and AI Agent will automatically discover and invoke your new tool!*

---

### 2️⃣ How to Remove a Tool (1 Simple Step)

To disable or remove a tool, simply comment out or delete its entry from the `TOOLS` list in the server file:

```python
# GMAIL_TOOLS = [
#     ...
#     # {"name": "deprecated_tool", ...}  <-- Delete or comment out
# ]
```

---

### 3️⃣ How to Add a Brand New MCP Microservice Server (2 Minutes)

1. **Copy the Template**:
   Copy `src/servers/template_server.py` to `src/servers/database_server.py`.
2. **Set Port and Tools**:
   Change `SERVER_PORT = 8103` and define your database tools.
3. **Register in Orchestrator (`run_servers.py`)**:
   Add 1 line in `run_servers.py`:
   ```python
   from src.servers.database_server import app as db_app
   ...
   run_server(db_app, 8103, "Database MCP Server")
   ```
4. **Register in Web UI (`ui/index.html`)**:
   Add the port to `MCP_SERVERS`:
   ```javascript
   { name: "Database", url: "http://localhost:8103/mcp", tools: [] }
   ```

