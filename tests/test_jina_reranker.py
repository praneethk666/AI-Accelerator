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


def test_jina_reranker_key_rotation():
    keys = ["key1_123456789", "key2_123456789", "key3_123456789", "key4_123456789"]
    comma_keys = ", ".join(keys)
    client = JinaRerankerAPIClient("jina-model", api_key=comma_keys)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9}
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        scores = client.predict([("query", "doc1")])
        headers = mock_post.call_args[1]["headers"]
        auth_header = headers["Authorization"]
        assert auth_header.startswith("Bearer key")
        used_key = auth_header.replace("Bearer ", "")
        assert used_key in keys


def test_jina_reranker_failover_on_429():
    keys = ["key1_fail", "key2_success"]
    client = JinaRerankerAPIClient("jina-model", api_key=", ".join(keys))

    mock_fail = MagicMock()
    mock_fail.status_code = 429
    mock_fail.text = "Rate limit exceeded"

    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.json.return_value = {
        "results": [{"index": 0, "relevance_score": 0.88}]
    }

    with patch("requests.post", side_effect=[mock_fail, mock_ok]) as mock_post:
        scores = client.predict([("query", "doc1")])
        assert scores == [0.88]
        assert mock_post.call_count == 2
        # First attempt used key1, second attempt used key2
        first_call_key = mock_post.call_args_list[0][1]["headers"]["Authorization"]
        second_call_key = mock_post.call_args_list[1][1]["headers"]["Authorization"]
        assert first_call_key == "Bearer key1_fail"
        assert second_call_key == "Bearer key2_success"


def test_jina_reranker_all_keys_exhausted():
    keys = ["key1_fail", "key2_fail"]
    client = JinaRerankerAPIClient("jina-model", api_key=", ".join(keys))

    mock_fail1 = MagicMock()
    mock_fail1.status_code = 429
    mock_fail1.text = "Rate limit exceeded"

    mock_fail2 = MagicMock()
    mock_fail2.status_code = 429
    mock_fail2.text = "Rate limit exceeded"

    with patch("requests.post", side_effect=[mock_fail1, mock_fail2]):
        with pytest.raises(RuntimeError, match="All Jina API keys exhausted or rate-limited"):
            client.predict([("query", "doc1")])


