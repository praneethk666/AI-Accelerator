"""
config.py — Typed dataclasses for server, authentication, security, and SMTP configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8100
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class JWTConfig:
    enabled: bool = False
    secret_key: str = ""
    issuer: str = ""
    audience: str = ""
    algorithms: List[str] = field(default_factory=lambda: ["HS256"])
    jwks_url: str = ""


@dataclass
class AuthConfig:
    enabled: bool = True
    tokens: Dict[str, str] = field(default_factory=dict)  # token -> caller_identity
    users: Dict[str, Dict[str, str]] = field(default_factory=dict)  # username -> {password, identity, token}
    permissions: Dict[str, List[str]] = field(default_factory=dict)  # caller_identity -> [allowed_tools]
    jwt: JWTConfig = field(default_factory=JWTConfig)


@dataclass
class RateLimitConfig:
    max_calls: int = 10
    window_seconds: int = 60


@dataclass
class PromptInjectionConfig:
    enabled: bool = True
    strict_mode: bool = True


@dataclass
class EmailSecurityConfig:
    allowlist: List[str] = field(default_factory=list)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    prompt_injection_guard: PromptInjectionConfig = field(default_factory=PromptInjectionConfig)


@dataclass
class SecurityConfig:
    email: EmailSecurityConfig = field(default_factory=EmailSecurityConfig)


@dataclass
class SMTPConfig:
    mode: str = "simulation"  # "simulation" or "smtp"
    host: str = "smtp.example.com"
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    sender_address: str = "notifications@company.com"


@dataclass
class OpenBaoConfig:
    url: str
    role_id: str
    secret_id: str
    path_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class CredentialsConfig:
    provider: str
    openbao: OpenBaoConfig


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    credentials: CredentialsConfig = field(default_factory=lambda: CredentialsConfig("env", OpenBaoConfig("", "", "")))


_config: Optional[AppConfig] = None


def _resolve_env(value):
    """Resolve ${env:VAR} placeholders from the current environment."""
    if isinstance(value, str) and value.startswith("${env:") and value.endswith("}"):
        env_name = value[6:-1]
        return os.environ.get(env_name, "")
    if isinstance(value, dict):
        resolved_dict = {}
        for k, v in value.items():
            new_k = _resolve_env(k) if isinstance(k, str) else k
            resolved_dict[new_k] = _resolve_env(v)
        return resolved_dict
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration dynamically from YAML file."""
    global _config
    if _config is not None:
        return _config

    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = _resolve_env(yaml.safe_load(f) or {})

    server_raw = raw.get("server", {})
    auth_raw = raw.get("auth", {})
    sec_raw = raw.get("security", {})
    email_sec_raw = sec_raw.get("email", {})
    rl_raw = email_sec_raw.get("rate_limit", {})
    pi_raw = email_sec_raw.get("prompt_injection_guard", {})
    smtp_raw = raw.get("smtp", {})
    cred_raw = raw.get("credentials", {})
    bao_raw = cred_raw.get("openbao", {})
    path_mapping = bao_raw.get("path_mapping", {})

    rate_limit = RateLimitConfig(
        max_calls=rl_raw.get("max_calls", 10),
        window_seconds=rl_raw.get("window_seconds", 60),
    )

    pi_guard = PromptInjectionConfig(
        enabled=pi_raw.get("enabled", True),
        strict_mode=pi_raw.get("strict_mode", True),
    )

    email_security = EmailSecurityConfig(
        allowlist=email_sec_raw.get("allowlist", []),
        rate_limit=rate_limit,
        prompt_injection_guard=pi_guard,
    )

    _config = AppConfig(
        server=ServerConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8100),
            allowed_origins=server_raw.get("allowed_origins", ["*"]),
        ),
        auth=AuthConfig(
            enabled=auth_raw.get("enabled", True),
            tokens=auth_raw.get("tokens", {}),
            users=auth_raw.get("users", {}),
            permissions=auth_raw.get("permissions", {}),
            jwt=JWTConfig(
                enabled=auth_raw.get("jwt", {}).get("enabled", False),
                secret_key=auth_raw.get("jwt", {}).get("secret_key", ""),
                issuer=auth_raw.get("jwt", {}).get("issuer", ""),
                audience=auth_raw.get("jwt", {}).get("audience", ""),
                algorithms=auth_raw.get("jwt", {}).get("algorithms", ["HS256"]),
                jwks_url=auth_raw.get("jwt", {}).get("jwks_url", ""),
            ),
        ),
        security=SecurityConfig(
            email=email_security,
        ),
        smtp=SMTPConfig(
            mode=smtp_raw.get("mode", "simulation"),
            host=smtp_raw.get("host", "smtp.example.com"),
            port=smtp_raw.get("port", 587),
            username=smtp_raw.get("username", ""),
            password=smtp_raw.get("password", ""),
            use_tls=smtp_raw.get("use_tls", True),
            sender_address=smtp_raw.get("sender_address", "notifications@company.com"),
        ),
        credentials=CredentialsConfig(
            provider=os.getenv("CREDENTIAL_PROVIDER", "env"),
            openbao=OpenBaoConfig(
                url=os.getenv("OPENBAO_URL", "http://127.0.0.1:8200"),
                role_id=os.getenv("OPENBAO_ROLE_ID", ""),
                secret_id=os.getenv("OPENBAO_SECRET_ID", ""),
                path_mapping=path_mapping
            )
        )
    )
    return _config


def reset_config() -> None:
    """Resets cached configuration."""
    global _config
    _config = None
