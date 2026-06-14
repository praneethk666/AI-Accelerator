"""Shared LLM factory. Every tool that needs an LLM calls get_llm(config).

Providers (config["llm"]["provider"]):
  groq    — ChatGroq (native)
  google  — ChatGoogleGenerativeAI (Gemini / Vertex AI Studio, native)
  ollama  — ChatOllama (local, no key)
  openai  — ChatOpenAI, AND any OpenAI-COMPATIBLE endpoint via base_url. This one
            key covers NVIDIA NIM, OpenRouter, Together, Fireworks, vLLM, LM Studio,
            even Gemini's OpenAI-compat endpoint — point base_url at the provider.

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

Langfuse tracing is wired automatically when LANGFUSE_* env vars are set; absent,
calls still work (tracing silently skipped).
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def get_llm(config: dict) -> BaseChatModel:
    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    api_key = _clean(llm_cfg.get("api_key"))
    temperature = llm_cfg.get("temperature")
    callbacks = _langfuse_callbacks()

    common = {"callbacks": callbacks}
    if temperature is not None:
        common["temperature"] = temperature

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, **_with_key(common, "api_key", api_key))

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, **_with_key(common, "google_api_key", api_key)
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kw = dict(common)
        if llm_cfg.get("base_url"):
            kw["base_url"] = llm_cfg["base_url"]
        return ChatOllama(model=model, **kw)

    if provider == "openai":
        # native OpenAI or ANY OpenAI-compatible endpoint (base_url).
        from langchain_openai import ChatOpenAI
        kw = dict(common)
        if llm_cfg.get("base_url"):
            kw["base_url"] = llm_cfg["base_url"]
        return ChatOpenAI(model=model, **_with_key(kw, "api_key", api_key))

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        "Use 'groq', 'google', 'ollama', or 'openai' (openai + base_url covers any "
        "OpenAI-compatible API such as NVIDIA NIM / OpenRouter / vLLM)."
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
    try:
        from langfuse.callback import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []
