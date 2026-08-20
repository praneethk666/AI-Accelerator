import pytest
from src.credentials.service import CredentialService, get_credential_service, reset_credential_service
from src.credentials.providers.env_provider import EnvProvider
from src.config import CredentialsConfig, OpenBaoConfig

import os

def test_credential_service_env_provider(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "testuser")
    monkeypatch.setenv("SMTP_PASS", "testpass")
    
    provider = EnvProvider()
    service = CredentialService(provider, path_mapping={"smtp": "mcp/smtp"})
    
    creds = service.get("smtp")
    assert creds is not None
    assert creds["username"] == "testuser"
    assert creds["password"] == "testpass"

def test_credential_service_unknown_id():
    provider = EnvProvider()
    service = CredentialService(provider)
    
    creds = service.get("unknown_id")
    assert creds is None

def test_get_credential_service_factory_env():
    reset_credential_service()
    config = CredentialsConfig(provider="env", openbao=OpenBaoConfig("", "", ""))
    service = get_credential_service(config)
    
    assert isinstance(service.provider, EnvProvider)

def test_get_credential_service_factory_unknown():
    reset_credential_service()
    config = CredentialsConfig(provider="fake_provider", openbao=OpenBaoConfig("", "", ""))
    with pytest.raises(ValueError, match="Unknown credential provider: fake_provider"):
        get_credential_service(config)

def test_credential_service_path_mapping():
    provider = EnvProvider()
    service = CredentialService(provider, path_mapping={"smtp": "secret/data/prod/smtp"})
    
    # In EnvProvider get_credential normally ignores path conceptually or uses it as env key, 
    # but the service layer maps it before invoking provider.get_credential
    # We can mock the provider to verify
    from unittest.mock import MagicMock
    mock_provider = MagicMock()
    service_with_mock = CredentialService(mock_provider, path_mapping={"smtp": "secret/data/prod/mcp/smtp"})
    
    service_with_mock.get("smtp")
    mock_provider.get_credential.assert_called_with("secret/data/prod/mcp/smtp")
    
def test_get_credential_service_factory_openbao():
    reset_credential_service()
    config = CredentialsConfig(
        provider="openbao", 
        openbao=OpenBaoConfig(
            url="http://127.0.0.1:8200", 
            role_id="role", 
            secret_id="secret",
            path_mapping={"smtp": "mapped_path"}
        )
    )
    service = get_credential_service(config)
    
    from src.credentials.providers.openbao import OpenBaoProvider
    assert isinstance(service.provider, OpenBaoProvider)
    assert service.path_mapping["smtp"] == "mapped_path"
    assert service.provider.url == "http://127.0.0.1:8200"
