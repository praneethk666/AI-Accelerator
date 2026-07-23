# tests/test_jina_reranker.py
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from backend.core.models import JinaRerankerAPIClient, get_reranker

def test_jina_reranker_raises_value_error_if_no_key():
    client = JinaRerankerAPIClient("jina-model", api_key=None)
    with pytest.raises(ValueError, match="JINA_API_KEY is not set"):
        client.predict([("query", "doc")])

def test_jina_reranker_predict_success():
    client = JinaRerankerAPIClient("jina-model", api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.45}
        ]
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        scores = client.predict([("query", "doc1"), ("query", "doc2")])
        
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        
        json_data = mock_post.call_args[1]["json"]
        assert json_data["model"] == "jina-model"
        assert json_data["query"] == "query"
        assert json_data["documents"] == ["doc1", "doc2"]
        
        assert scores == [0.95, 0.45]

def test_get_jina_reranker_from_config():
    config = {
        "embeddings": {
            "reranker_provider": "jina",
            "reranker_model": "jina-reranker-v2-base-multilingual",
            "reranker_api_key": "some-key"
        }
    }
    # Reset singleton to test instantiation
    with patch("backend.core.models._reranker", None):
        reranker = get_reranker(config)
        assert isinstance(reranker, JinaRerankerAPIClient)
        assert reranker.model_name == "jina-reranker-v2-base-multilingual"
        assert reranker.api_key == "some-key"
