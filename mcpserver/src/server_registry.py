import logging
from typing import Dict, Any, Optional

from src.mcp_client import MCPClientManager, StreamableHttpClient, StdioClient

logger = logging.getLogger(__name__)

class ServerRegistry:
    """Manages configured MCP servers and their clients."""
    
    def __init__(self):
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, MCPClientManager] = {}

    def load_from_config(self, servers_config: Dict[str, Any]):
        """Load servers from the parsed servers.yaml configuration."""
        for name, config in servers_config.items():
            if name == "local":
                continue # local tools are handled directly by the internal registry
            self.register_server(name, config)

    def register_server(self, name: str, config: Dict[str, Any]):
        """Registers a single server and initializes its client."""
        self.servers[name] = config
        
        transport = config.get("transport", {})
        kind = transport.get("kind")
        
        if kind == "streamable-http":
            self.clients[name] = StreamableHttpClient(name, config)
        elif kind == "stdio":
            self.clients[name] = StdioClient(name, config)
        else:
            logger.warning(f"Unknown transport kind '{kind}' for server '{name}'.")

    def get_client(self, name: str) -> Optional[MCPClientManager]:
        """Retrieves the client for a specific server name."""
        return self.clients.get(name)

    def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """Returns all registered server configs."""
        return self.servers

    async def initialize_all(self):
        """Attempts to initialize connection to all registered clients."""
        for name, client in self.clients.items():
            try:
                await client.initialize()
                logger.info(f"Initialized client for server: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize client for {name}: {e}")
