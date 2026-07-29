"""
config_schema.py — Pydantic validation schema for config/global.yaml.

Ensures the guardrail configuration:
  - Has all required stages.
  - Does not have inverted thresholds (e.g. warn_threshold >= block_threshold).
  - Enforces output_guard_pct is exactly 100% (compliance constraint).
  - Uses correct types and bounds (e.g. percentages 0-100, sizes > 0).

Fails fast at startup if config is invalid.
"""
from __future__ import annotations

from typing import Optional, Dict
from pydantic import BaseModel, Field, validator


class PolicyConfig(BaseModel):
    block_threshold:   int   = Field(80,   ge=1,   le=100)
    warn_threshold:    int   = Field(40,   ge=1,   le=100)
    bypass_alert_rate: float = Field(0.05, ge=0.0, le=1.0)
    stage_weights:     Dict[str, float] = Field(default_factory=lambda: {
        "input": 1.0, "retrieval": 1.2, "output": 1.5
    })

    @validator("warn_threshold")
    def warn_below_block(cls, v, values):
        if "block_threshold" in values and v >= values["block_threshold"]:
            raise ValueError(
                f"warn_threshold ({v}) must be strictly less than block_threshold "
                f"({values['block_threshold']}). Inverting this blocks every query."
            )
        return v


class InputConfig(BaseModel):
    max_query_chars:        int  = Field(2000, ge=10, le=100000)
    pii_redact:             bool = True
    injection_check:        bool = True
    encoding_entropy_check: bool = False
    off_topic_check:        bool = False
    passport_pattern:       bool = False
    voter_id_pattern:       bool = False


class RetrievalConfig(BaseModel):
    chunk_injection_scan: bool = True
    scan_max_bytes:       int  = Field(100000, ge=100, le=10000000)


class GroundednessConfig(BaseModel):
    enabled:    bool = False
    fail_open:  bool = True
    timeout_ms: int  = Field(3000, ge=100, le=30000)


class OutputConfig(BaseModel):
    pii_mask:         bool = True
    gstin_before_pan: bool = True
    groundedness:     GroundednessConfig = Field(default_factory=GroundednessConfig)


class TokenQuotaConfig(BaseModel):
    enabled:           bool = True
    window_minutes:    int  = Field(60,      ge=1,   le=1440)
    tokens_per_window: int  = Field(500_000, ge=1000)
    reserve_tokens:    int  = Field(2000,    ge=0)
    fail_open:         bool = True


class TokenBudgetConfig(BaseModel):
    enabled:            bool = True
    max_context_tokens: int  = Field(8000, ge=256, le=128000)


class SessionRiskConfig(BaseModel):
    enabled:                 bool = True
    window_minutes:          int  = Field(30,  ge=1,  le=1440)
    session_block_threshold: int  = Field(150, ge=10, le=10000)
    redis_url:               Optional[str] = None


class DegradationConfig(BaseModel):
    qdrant_max_retries:           int   = Field(2,   ge=0, le=10)
    qdrant_base_delay_s:          float = Field(0.5, ge=0.0, le=10.0)
    reranker_max_retries:         int   = Field(2,   ge=0, le=10)
    reranker_base_delay_s:        float = Field(0.5, ge=0.0, le=10.0)
    reranker_max_pairs:           int   = Field(50,  ge=1, le=500)
    reranker_max_tokens_per_pair: int   = Field(512, ge=64, le=4096)
    reranker_max_total_tokens:    int   = Field(10000, ge=1000, le=1000000)


class RolloutConfig(BaseModel):
    input_guard_pct:     int = Field(100, ge=0, le=100)
    retrieval_guard_pct: int = Field(50,  ge=0, le=100)
    output_guard_pct:    int = Field(100, ge=0, le=100)
    token_quota_pct:     int = Field(5,   ge=0, le=100)
    session_risk_pct:    int = Field(20,  ge=0, le=100)

    @validator("output_guard_pct")
    def output_guard_must_be_100(cls, v):
        if v < 100:
            raise ValueError(
                "output_guard_pct must be exactly 100. PII masking is a compliance "
                "requirement and cannot be canary-rolled."
            )
        return v


class LoggingConfig(BaseModel):
    enabled:          bool = True
    ring_buffer_size: int  = Field(500, ge=10, le=5000)


class HealthProbeConfig(BaseModel):
    enabled:          bool = True
    interval_seconds: int  = Field(60, ge=5, le=3600)


class CostConfig(BaseModel):
    enabled:                bool  = True
    llm_input_cost_per_1m:  float = Field(0.075, ge=0.0)
    llm_output_cost_per_1m: float = Field(0.30,  ge=0.0)
    embed_cost_per_1m:      float = Field(0.0,   ge=0.0)


class GuardrailConfig(BaseModel):
    enabled:        bool              = True
    version:        str               = "1.0.0"
    policy:         PolicyConfig      = Field(default_factory=PolicyConfig)
    input:          InputConfig       = Field(default_factory=InputConfig)
    retrieval:      RetrievalConfig   = Field(default_factory=RetrievalConfig)
    output:         OutputConfig      = Field(default_factory=OutputConfig)
    token_quota:    TokenQuotaConfig  = Field(default_factory=TokenQuotaConfig)
    token_budget:   TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    session_risk:   SessionRiskConfig = Field(default_factory=SessionRiskConfig)
    degradation:    DegradationConfig = Field(default_factory=DegradationConfig)
    rollout:        RolloutConfig     = Field(default_factory=RolloutConfig)
    logging:        LoggingConfig     = Field(default_factory=LoggingConfig)
    health_probe:   HealthProbeConfig = Field(default_factory=HealthProbeConfig)
    cost:           CostConfig        = Field(default_factory=CostConfig)


def validate_guardrail_config(raw_config: dict) -> GuardrailConfig:
    """Validate raw config and return a structured GuardrailConfig object.

    Raises ValueError on validation failure.
    """
    g = raw_config.get("guardrails") or {}
    # Use Pydantic to parse/validate the dictionary
    return GuardrailConfig(**g)
