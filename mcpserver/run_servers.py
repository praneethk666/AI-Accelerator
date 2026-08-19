"""
run_servers.py — Master Multi-MCP Server Orchestrator.
Runs all specialized MCP servers and the Web UI Hub concurrently with a single command.

Architecture:
  - Port 8100: AI Agent Hub & Web UI (http://localhost:8100)
  - Port 8101: Gmail MCP Server (http://localhost:8101/mcp)
  - Port 8102: Time & System MCP Server (http://localhost:8102/mcp)
"""

import asyncio
import logging
import signal
import sys
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] %(message)s")
logger = logging.getLogger("MCPOrchestrator")


async def run_server(app_path: str, port: int, name: str):
    logger.info(f"Starting {name} on http://0.0.0.0:{port}...")
    config = uvicorn.Config(app_path, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    print("=" * 70)
    print("🚀 STARTING MULTI-MCP SERVER ARCHITECTURE")
    print("=" * 70)
    print("  🌐 Web UI & AI Agent Hub     → http://localhost:8100")
    print("  📧 Gmail MCP Server           → http://localhost:8101/mcp (Tools: send, read, search, draft)")
    print("  🕒 Time & System MCP Server   → http://localhost:8102/mcp (Tools: datetime, timezone, uptime)")
    print("=" * 70)
    print("Press Ctrl+C to stop all servers gracefully.\n")

    servers = [
        run_server("src.server:app", 8100, "AI Agent Hub & Web UI"),
        run_server("src.servers.gmail_server:app", 8101, "Gmail MCP Server"),
        run_server("src.servers.time_server:app", 8102, "Time & System MCP Server"),
    ]

    try:
        await asyncio.gather(*servers)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down all MCP servers...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAll MCP servers stopped.")
        sys.exit(0)
