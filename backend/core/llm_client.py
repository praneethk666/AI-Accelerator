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

Langfuse tracing is wired automatically when LANGFUSE_* env vars are set; absent,
calls still work (tracing silently skipped).
"""
from __future__ import annotations

import os

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
    callbacks = _langfuse_callbacks()

    common = {"callbacks": callbacks}
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


def _clean(value):
    """Treat blank or unresolved ${VAR} placeholders as 'not set'."""
    if not value or (isinstance(value, str) and value.startswith("${")):
        return None
    return value


def _with_key(kwargs: dict, key_name: str, api_key) -> dict:
    # only pass an explicit key when we actually have one; else let the SDK read
    # its own default env var (GROQ_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY).
    if api_key:
        kwargs = dict(kwargs)
        kwargs[key_name] = api_key
    return kwargs


def _langfuse_callbacks() -> list:
    # langfuse v3 -> v4 moved the handler to langfuse.langchain; try the new path
    # first, fall back to the old one. Absent/misconfigured -> no callbacks (calls
    # still work, just untraced).
    # Only attach when both keys are set — otherwise langfuse v4 starts an OTEL
    # exporter against LANGFUSE_HOST (default localhost:3001) and floods the logs.
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        try:
            from langfuse.callback import CallbackHandler
            return [CallbackHandler()]
        except Exception:
            return []


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