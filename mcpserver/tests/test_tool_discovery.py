import pytest
from src.registry import ToolRegistry

@pytest.mark.asyncio
async def test_tool_registry_discovery():
    registry = ToolRegistry()
    
    # Assuming the registry comes preloaded with the default TOOLS
    initial_count = len(registry.get_tool_definitions())
    
    remote_tools = [
        {
            "name": "remote_search",
            "description": "Searches data remotely",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    registry.register_remote_tools("remote-server", remote_tools)
    
    definitions = registry.get_tool_definitions()
    assert len(definitions) == initial_count + 1
    
    names = [d["name"] for d in definitions]
    assert "remote_search" in names
