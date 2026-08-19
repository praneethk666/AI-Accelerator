import pytest
from src.server_registry import ServerRegistry
from src.mcp_client import StreamableHttpClient, StdioClient

def test_server_registry_registration():
    registry = ServerRegistry()
    config = {
        "http-server": {
            "transport": {"kind": "streamable-http", "url": "http://example.com/mcp"}
        },
        "stdio-server": {
            "transport": {"kind": "stdio", "command": "echo"}
        }
    }
    
    registry.load_from_config(config)
    
    assert len(registry.clients) == 2
    assert "http-server" in registry.clients
    assert "stdio-server" in registry.clients
    
    assert isinstance(registry.get_client("http-server"), StreamableHttpClient)
    assert isinstance(registry.get_client("stdio-server"), StdioClient)
