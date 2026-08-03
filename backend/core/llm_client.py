"""Shared LLM factory. Every tool that needs an LLM calls get_llm(config).

Providers (config["llm"]["provider"]):
  groq      — ChatGroq (native)
  google    — ChatGoogleGenerativeAI (Gemini / Vertex AI Studio, native)
  ollama    — ChatOllama (local, no key)
  openai    — ChatOpenAI, AND any OpenAI-COMPATIBLE endpoint via base_url. This one
              key covers NVIDIA NIM, OpenRouter, Together, Fireworks, vLLM, LM Studio,
              even Gemini's OpenAI-compat endpoint — point base_url at the provider.
  anthropic — ChatAnthropic (Claude models). Excellent structured-JSON field discipline
              and cheap for bulk extraction. Needs ANTHROPIC_API_KEY.

config["llm"] keys (all optional except provider/model):
  provider, model
  base_url   — for openai-compatible endpoints (e.g. NVIDIA NIM:
               https://integrate.api.nvidia.com/v1)
  api_key    — usually "${SOME_ENV_VAR}" in global.yaml (resolved by load_config).
               If absent/blank, the provider SDK falls back to its own default env
               (GROQ_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY).
  temperature

Recipes (config/global.yaml):
  Groq:        provider: groq    model: llama-3.1-8b-instant
  NVIDIA NIM:  provider: openai  model: meta/llama-3.1-8b-instruct
               base_url: https://integrate.api.nvidia.com/v1  api_key: ${NVIDIA_API_KEY}
  Gemini:      provider: google  model: gemini-1.5-flash      api_key: ${GOOGLE_API_KEY}
  OpenRouter:  provider: openai  model: <org/model>
               base_url: https://openrouter.ai/api/v1         api_key: ${OPENROUTER_API_KEY}
  Local vLLM:  provider: openai  model: <served-name>  base_url: http://localhost:8000/v1
  Anthropic:   provider: anthropic  model: claude-haiku-4-5  api_key: ${ANTHROPIC_API_KEY}

LLM calls are auto-traced by opentelemetry-instrumentation-langchain (see
backend/core/tracing.py), which patches LangChain's Runnable.invoke so every
call becomes a child span of whatever OTEL span is currently active — no
per-call wiring needed here.
"""
from __future__ import annotations

import os
import random

from langchain_core.language_models import BaseChatModel


def get_llm(config: dict, max_tokens: int | None = None,
            model: str | None = None) -> BaseChatModel:
    """Build the configured chat model.

    max_tokens caps the completion length. It's provider-agnostic here — each SDK
    names the kwarg differently (groq/openai: max_tokens, google: max_output_tokens,
    ollama: num_predict) — so callers pass one number and we map it. An explicit
    arg wins; otherwise we fall back to llm.max_tokens in config if set. Used e.g.
    by enrich_chunks to guarantee a batch's JSON array isn't truncated.

    model overrides llm.model for a single call so different tasks can use different
    models on the SAME provider — e.g. a strong reasoning model for answering vs the
    cheap/fast default for bulk enrichment. None -> the configured llm.model.
    """
    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = model or llm_cfg["model"]
    api_key = _clean(llm_cfg.get("api_key"))
    temperature = llm_cfg.get("temperature")
    if max_tokens is None:
        max_tokens = llm_cfg.get("max_tokens")

    common = {}
    if temperature is not None:
        common["temperature"] = temperature

    if provider == "groq":
        from langchain_groq import ChatGroq
        kw = dict(common)
        if max_tokens:
            kw["max_tokens"] = max_tokens
        return ChatGroq(model=model, **_with_key(kw, "api_key", api_key))

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kw = dict(common)
        if max_tokens:
            kw["max_output_tokens"] = max_tokens
        # Reasoning is ON by default and the parameter to disable it is MODEL-FAMILY
        # SPECIFIC (thinking_budget: Gemini 2.5 family; thinking_level: Gemma family) —
        # config-driven, not hardcoded, since get_llm() builds the client lazily and
        # can't try-then-fall-back the way _describe_google() does at call time.
        # Validated live (21-Jul): without this, gemini-2.5-flash burned 424 of 443
        # output tokens on hidden reasoning for a ONE-SENTENCE summary — 96% waste,
        # silently inflating both cost and latency on every categorize/enrichment call.
        if llm_cfg.get("thinking_budget") is not None:
            kw["thinking_budget"] = llm_cfg["thinking_budget"]
        if llm_cfg.get("thinking_level") is not None:
            kw["thinking_level"] = llm_cfg["thinking_level"]
        return ChatGoogleGenerativeAI(
            model=model, **_with_key(kw, "google_api_key", api_key)
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kw = dict(common)
        if llm_cfg.get("base_url"):
            kw["base_url"] = llm_cfg["base_url"]
        if max_tokens:
            kw["num_predict"] = max_tokens
        return ChatOllama(model=model, **kw)

    if provider == "openai":
        # native OpenAI or ANY OpenAI-compatible endpoint (base_url).
        from langchain_openai import ChatOpenAI
        kw = dict(common)
        if llm_cfg.get("base_url"):
            kw["base_url"] = llm_cfg["base_url"]
        if max_tokens:
            kw["max_tokens"] = max_tokens
        # Passthrough for provider-specific request fields that aren't part of the
        # OpenAI schema (e.g. GLM's `thinking: {type: disabled}` — GLM has a reasoning
        # mode ON by default that otherwise burns the whole max_tokens budget on hidden
        # reasoning tokens and returns EMPTY content; validated live against z.ai).
        # Belongs in config, not hardcoded here, since it's specific to whichever
        # OpenAI-compatible provider is configured.
        if llm_cfg.get("extra_body"):
            kw["extra_body"] = llm_cfg["extra_body"]
        if llm_cfg.get("top_p") is not None:
            kw["top_p"] = llm_cfg["top_p"]
        if "streaming" in llm_cfg:
            kw["streaming"] = llm_cfg["streaming"]
        elif "stream" in llm_cfg:
            kw["streaming"] = llm_cfg["stream"]
        kw["timeout"] = 30.0  # Set standard 30s timeout to protect against indefinite hangs
        llm = ChatOpenAI(model=model, **_with_key(kw, "api_key", api_key))
        # Structured Outputs: if a json_schema is configured and we're on a native OpenAI
        # endpoint (no base_url = not vLLM/OpenRouter), bind the schema so the model is
        # GUARANTEED to return valid JSON matching our invoice field structure.
        schema = llm_cfg.get("json_schema")
        if schema and not llm_cfg.get("base_url"):
            try:
                llm = llm.with_structured_output(schema, method="json_schema")
            except Exception:
                pass  # schema binding not available on this langchain version — ignore
        return llm

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kw = dict(common)
        if max_tokens:
            kw["max_tokens"] = max_tokens
        return ChatAnthropic(model=model, **_with_key(kw, "api_key", api_key))

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        "Use 'groq', 'google', 'ollama', 'openai' (openai + base_url covers any "
        "OpenAI-compatible API such as NVIDIA NIM / OpenRouter / vLLM), or 'anthropic'."
    )


def get_llm_for(config: dict, section: dict | None = None, *,
                max_tokens: int | None = None,
                default_model: str | None = None) -> BaseChatModel:
    """Build an LLM for ONE pipeline step, with an optional per-step override.

    `section` is that step's own config dict (e.g. config["enrichment"],
    config["categorization"], config["query"]["planner"]). If it carries any of
    model / provider / api_key / base_url / temperature / extra_body /
    thinking_budget / thinking_level, those override the global `llm` block FOR THIS
    CALL ONLY; anything unset inherits the global default. extra_body (openai
    provider) and thinking_budget/thinking_level (google provider) are all
    provider/model-family-specific reasoning-suppression knobs — if a step points at
    a DIFFERENT provider or Google model family than the global block, override the
    relevant one explicitly to avoid inheriting a field the new target doesn't
    recognize (validated live: NVIDIA 400s on z.ai's `thinking` key; Gemini 2.5
    rejects thinking_level, Gemma rejects thinking_budget).
    This is what lets categorize / enrich / plan / answer each run a different model
    (e.g. a cheap model for bulk enrichment, a stronger one for answering).

    Model resolution order: section.model -> default_model (e.g. llm.answer_model)
    -> global llm.model. When section switches provider without giving its own
    api_key, the key is cleared so the SDK reads that provider's default env var
    (mirrors the agent executor's provider-override handling).
    """
    section = section or {}
    global_llm = config.get("llm", {}) or {}
    llm_cfg = dict(global_llm)

    chosen_model = section.get("model") or default_model
    if chosen_model:
        llm_cfg["model"] = chosen_model
    for k in ("provider", "api_key", "base_url", "temperature", "extra_body",
              "thinking_budget", "thinking_level"):
        if section.get(k) is not None:
            llm_cfg[k] = section[k]

    ov_provider = section.get("provider")
    if ov_provider and ov_provider != global_llm.get("provider") and section.get("api_key") is None:
        llm_cfg["api_key"] = None

    return get_llm({**config, "llm": llm_cfg}, max_tokens=max_tokens)

def resolve_model_provider(config: dict, section: dict | None = None,
                            default_model: str | None = None) -> tuple[str, str]:
    """Same model/provider precedence get_llm_for() applies internally, exposed
    so callers can log the ACTUAL model/provider a call used (e.g. for
    usage.record_from_message's model=/provider= kwargs) without duplicating
    get_llm_for's resolution order and risking the two drifting apart. Keep
    this in sync with get_llm_for()'s resolution comment if that ever changes.
    """
    section = section or {}
    global_llm = config.get("llm", {}) or {}
    model = section.get("model") or default_model or global_llm.get("model")
    provider = section.get("provider") or global_llm.get("provider")
    return model, provider

def _clean(value):
    """Treat blank or unresolved ${VAR} placeholders as 'not set'.
    If multiple keys are provided via comma-separation, randomly select one for load balancing."""
    if not value or (isinstance(value, str) and value.startswith("${")):
        return None
    if isinstance(value, str) and "," in value:
        keys = [k.strip() for k in value.split(",") if k.strip()]
        if keys:
            return random.choice(keys)
    return value


def _with_key(kwargs: dict, key_name: str, api_key) -> dict:
    # only pass an explicit key when we actually have one; else let the SDK read
    # its own default env var (GROQ_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY).
    if api_key:
        kwargs = dict(kwargs)
        kwargs[key_name] = api_key
    return kwargs


def clean_message_content(content) -> str:
    """Standardize the LLM message content representation.
    In some integrations (like langchain_google_genai), content can be returned
    as a list of dictionaries/parts rather than a raw string.
    """
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or "")
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return str(content or "")