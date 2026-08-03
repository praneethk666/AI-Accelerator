import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.core.models import OpenAIEmbeddingsAPIClient, get_dense_model

def test_openai_embeddings_client_single_query():
    client = OpenAIEmbeddingsAPIClient(model_name="text-embedding-3-small", api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0}
        ]
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        result = client.encode("hello world")
        
        # Verify the result is a 1D numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])
        
        mock_post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": ["hello world"]
            },
            timeout=30
        )

def test_openai_embeddings_client_batch_passages():
    client = OpenAIEmbeddingsAPIClient(model_name="text-embedding-3-small", api_key="test-key", dimensions=1024)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
            {"embedding": [0.1, 0.2, 0.3], "index": 0}
        ]
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        result = client.encode(["passage 1", "passage 2"], batch_size=2)
        
        # Verify results are sorted by index
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(result[1], [0.4, 0.5, 0.6])
        
        mock_post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": ["passage 1", "passage 2"],
                "dimensions": 1024
            },
            timeout=30
        )

def test_openai_embeddings_missing_key():
    client = OpenAIEmbeddingsAPIClient(model_name="text-embedding-3-small", api_key=None)
    with pytest.raises(ValueError, match="OpenAI API Key is missing"):
        client.encode("hello")

def test_get_dense_model_resolves_openai():
    config = {
        "embeddings": {
            "dense_provider": "openai",
            "dense_model": "text-embedding-3-small",
            "dense_api_key": "test-config-key",
            "dense_dim": 1024
        }
    }
    
    # Temporarily reset the global _dense_model singleton for test isolation
    with patch("backend.core.models._dense_model", None):
        model = get_dense_model(config)
        assert isinstance(model, OpenAIEmbeddingsAPIClient)
        assert model.model_name == "text-embedding-3-small"
        assert model.api_key == "test-config-key"
        assert model.dimensions == 1024
