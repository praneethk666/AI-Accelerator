import pytest
from unittest.mock import AsyncMock, patch
from src.mcp_client import StreamableHttpClient

@pytest.mark.asyncio
async def test_streamable_http_client_initialize():
    client = StreamableHttpClient("test-server", {"transport": {"url": "http://test/mcp"}})
    
    mock_post = AsyncMock()
    mock_post.return_value = {"result": {"protocolVersion": "2025-06-18"}}
    
    with patch.object(client, "_post", mock_post):
        result = await client.initialize()
        assert result.get("protocolVersion") == "2025-06-18"
        mock_post.assert_awaited_once()

@pytest.mark.asyncio
async def test_streamable_http_client_list_tools():
    client = StreamableHttpClient("test-server", {"transport": {"url": "http://test/mcp"}})
    
    mock_post = AsyncMock()
    mock_post.return_value = {"result": {"tools": [{"name": "remote_tool"}]}}
    
    with patch.object(client, "_post", mock_post):
        result = await client.list_tools()
        assert len(result.get("tools", [])) == 1
        assert result["tools"][0]["name"] == "remote_tool"
