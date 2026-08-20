import pytest
from unittest.mock import MagicMock
from src.credentials.providers.openbao import OpenBaoProvider
import hvac

@pytest.fixture
def mock_hvac(monkeypatch):
    mock_client = MagicMock()
    mock_client.auth.approle.login.return_value = {
        'auth': {'client_token': 'test_token'}
    }
    
    def get_test_secret(path):
        if path == "mcp/smtp":
            return {
                'data': {
                    'data': {
                        'username': 'bao-user',
                        'password': 'bao-password'
                    }
                }
            }
        else:
            raise hvac.exceptions.InvalidPath()
    
    mock_client.secrets.kv.v2.read_secret_version.side_effect = get_test_secret
    
    import hvac
    monkeypatch.setattr(hvac, "Client", lambda url: mock_client)
    return mock_client

def test_openbao_authentication(mock_hvac):
    provider = OpenBaoProvider("http://localhost:8200", "test-role", "test-secret")
    
    # client token starts empty
    mock_hvac.is_authenticated.return_value = False
    
    # it lazy loads on get_credential
    creds = provider.get_credential("mcp/smtp")
    
    assert mock_hvac.auth.approle.login.called
    assert mock_hvac.token == "test_token"
    assert creds["username"] == "bao-user"
    assert creds["password"] == "bao-password"


def test_openbao_missing_role_credentials(mock_hvac):
    # should fail gracefully
    provider = OpenBaoProvider("http://localhost:8200", "", "")
    mock_hvac.is_authenticated.return_value = False
    
    creds = provider.get_credential("mcp/smtp")
    
    assert not mock_hvac.auth.approle.login.called
    assert creds is None


def test_openbao_path_not_found(mock_hvac):
    provider = OpenBaoProvider("http://localhost:8200", "test-role", "test-secret")
    mock_hvac.is_authenticated.return_value = True # skip re-auth
    
    creds = provider.get_credential("unknown/path")
    assert creds is None

def test_openbao_auth_failure(mock_hvac):
    mock_hvac.auth.approle.login.side_effect = Exception("Auth failed")
    mock_hvac.is_authenticated.return_value = False
    
    provider = OpenBaoProvider("http://localhost:8200", "test-role", "test-secret")
    
    creds = provider.get_credential("mcp/smtp")
    assert creds is None

def test_openbao_token_renewal(mock_hvac):
    provider = OpenBaoProvider("http://localhost:8200", "test-role", "test-secret")
    mock_hvac.is_authenticated.return_value = True # initially authenticated

    # Setup the mock so first read_secret raises Forbidden, second read returns data
    call_count = 0
    def side_effect_read(path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise hvac.exceptions.Forbidden("Token expired")
        return {
            'data': {
                'data': {
                    'username': 'renewed-user'
                }
            }
        }
    
    mock_hvac.secrets.kv.v2.read_secret_version.side_effect = side_effect_read
    
    creds = provider.get_credential("mcp/smtp")
    
    # login should be called once, after the forbidden error
    assert mock_hvac.auth.approle.login.call_count == 1
    assert creds is not None
    assert creds["username"] == "renewed-user"
