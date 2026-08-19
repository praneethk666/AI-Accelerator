import asyncio
import httpx
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MCPClientManager:
    """Base class for MCP Client."""
    def __init__(self, server_name: str, config: Dict[str, Any]):
        self.server_name = server_name
        self.config = config
        self._request_id = 0

    async def initialize(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def list_tools(self) -> Dict[str, Any]:
        """Calls tools/list on the remote server."""
        raise NotImplementedError

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Calls tools/call on the remote server."""
        raise NotImplementedError

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id


class StreamableHttpClient(MCPClientManager):
    """Client for Streamable HTTP MCP endpoints."""
    def __init__(self, server_name: str, config: Dict[str, Any]):
        super().__init__(server_name, config)
        transport = config.get("transport", {})
        self.url = transport.get("url", "")
        timeout_ms = transport.get("timeoutMs", 10000)
        self.client = httpx.AsyncClient(timeout=timeout_ms / 1000.0)

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self.client.post(self.url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error calling {self.url}: {e}")
            raise

    async def initialize(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "custom-mcp-gateway", "version": "1.0"}}
        }
        resp = await self._post(payload)
        return resp.get("result", {})

    async def list_tools(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        resp = await self._post(payload)
        return resp.get("result", {})

    async def close(self):
        await self.client.aclose()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        resp = await self._post(payload)
        return resp.get("result", {})


class StdioClient(MCPClientManager):
    """Client for STDIO MCP endpoints."""
    def __init__(self, server_name: str, config: Dict[str, Any]):
        super().__init__(server_name, config)
        self.process: Optional[asyncio.subprocess.Process] = None

    async def _start_process(self):
        if self.process is None:
            transport = self.config.get("transport", {})
            cmd = transport.get("command")
            args = transport.get("args", [])
            env = transport.get("env", None)
            
            self.process = await asyncio.create_subprocess_exec(
                cmd, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

    async def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._start_process()
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Subprocess not running")

        out_data = json.dumps(payload) + "\n"
        self.process.stdin.write(out_data.encode("utf-8"))
        await self.process.stdin.drain()

        # Read response
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("Unexpected EOF from subprocess")
            
        return json.loads(line.decode("utf-8"))

    async def initialize(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "custom-mcp-gateway", "version": "1.0"}}
        }
        resp = await self._send_request(payload)
        return resp.get("result", {})

    async def close(self):
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass


    async def list_tools(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        resp = await self._send_request(payload)
        return resp.get("result", {})

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        resp = await self._send_request(payload)
        return resp.get("result", {})
