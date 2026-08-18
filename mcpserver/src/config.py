"""
config.py — Typed dataclasses for server, authentication, security, and SMTP configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8100
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AuthConfig:
    enabled: bool = True
    tokens: Dict[str, str] = field(default_factory=dict)  # token -> caller_identity


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
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)


_config: Optional[AppConfig] = None


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration dynamically from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    server_raw = raw.get("server", {})
    auth_raw = raw.get("auth", {})
    sec_raw = raw.get("security", {})
    email_sec_raw = sec_raw.get("email", {})
    rl_raw = email_sec_raw.get("rate_limit", {})
    pi_raw = email_sec_raw.get("prompt_injection_guard", {})
    smtp_raw = raw.get("smtp", {})

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
    )
    return _config


def reset_config() -> None:
    """Resets cached configuration."""
    global _config
    _config = None
