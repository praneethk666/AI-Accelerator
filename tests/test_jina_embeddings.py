import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.core.models import JinaEmbeddingsAPIClient, get_dense_model

def test_jina_embeddings_client_single_query():
    client = JinaEmbeddingsAPIClient(model_name="jina-embeddings-v3", api_key="test-key")
    
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
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json"
            },
            json={
                "model": "jina-embeddings-v3",
                "input": ["hello world"],
                "truncate": True
            },
            timeout=30
        )

def test_jina_embeddings_client_batch_passages():
    client = JinaEmbeddingsAPIClient(model_name="jina-embeddings-v3", api_key="test-key")
    
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

def test_jina_embeddings_missing_key():
    client = JinaEmbeddingsAPIClient(model_name="jina-embeddings-v3", api_key=None)
    with pytest.raises(ValueError, match="Jina API Key is missing"):
        client.encode("hello")

def test_get_dense_model_resolves_jina():
    config = {
        "embeddings": {
            "dense_provider": "jina",
            "dense_model": "jina-embeddings-v3",
            "dense_api_key": "test-config-key"
        }
    }
    
    # Temporarily reset the global _dense_model singleton for test isolation
    with patch("backend.core.models._dense_model", None):
        model = get_dense_model(config)
        assert isinstance(model, JinaEmbeddingsAPIClient)
        assert model.model_name == "jina-embeddings-v3"
        assert model.api_key == "test-config-key"
