# 🚀 Multi-Server MCP Architecture & Groq AI Agent Hub

> **Production-Grade Model Context Protocol (MCP) Multi-Server Ecosystem**
> Features **Streamable HTTP + SSE**, **Bearer Token RBAC Auth**, **Dual-Port SMTP / IMAP Resilient Delivery**, **Multi-Timezone Conversions**, and a **100% Agentic Groq AI Web Hub**.

---

## 📖 Complete Documentation & Guide

👉 **For a complete, easy-to-understand walkthrough of all servers, tools, and security features, read the [Project Architecture Guide](file:///c:/Users/visha/OneDrive/Desktop/mcp/Custom_MCP_Server/PROJECT_ARCHITECTURE_GUIDE.md).**

---

## 🗺️ High-Level Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                     User Input Prompt                       │
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
                 │   Routes tool to Port 8101 or 8102    │   └────────────────────────┘
                 └─────────────────┬─────────────────────┘
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

## 🏛️ Ports & Microservices

| Server | Port | Endpoint URL | Description |
|---|---|---|---|
| **🌐 AI Agent Web Hub** | `8100` | `http://localhost:8100` | Unified Web UI interface & Master Gateway consolidating all tools |
| **📧 Gmail MCP Server** | `8101` | `http://localhost:8101/mcp` | 4 Email Tools (`send_email`, `read_inbox`, `search_emails`, `create_draft`) |
| **🕒 Time & System Server** | `8102` | `http://localhost:8102/mcp` | 3 Clock Tools (`get_current_datetime`, `convert_timezone`, `get_system_uptime`) |

---

## ⚡ Quickstart

### 1. Launch All Servers:
```bash
python run_servers.py
```

### 2. Access the Web Hub:
Open your browser at **[http://localhost:8100](http://localhost:8100)** and log in with `vishal` / `vishal123`.

### 3. Connect to MCP Inspector:
```bash
npx @modelcontextprotocol/inspector
```
Connect to `http://localhost:8100/mcp` with header `Authorization: Bearer vishal-test-token`.
