"""Shared LLM factory. Every tool that needs an LLM calls get_llm(config).

Supported providers (set via config["llm"]["provider"]):
  groq    — ChatGroq (fast, good for production)
  ollama  — ChatOllama (fully local, no API key needed)
  openai  — ChatOpenAI (or any OpenAI-compatible endpoint)

Langfuse tracing is wired automatically — every call through get_llm() is
traced without the caller doing anything extra. Tracing requires
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST in the environment
(see .env.example). If those vars are absent, calls still work — tracing is
silently skipped.
"""
from __future__ import annotations
from langchain_core.language_models import BaseChatModel


def get_llm(config: dict) -> BaseChatModel:
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    callbacks = _langfuse_callbacks()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, callbacks=callbacks)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, callbacks=callbacks)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        base_url = config["llm"].get("base_url")
        return ChatOpenAI(model=model, base_url=base_url, callbacks=callbacks)

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        "Set config['llm']['provider'] to 'groq', 'ollama', or 'openai'."
    )


def _langfuse_callbacks() -> list:
    try:
        from langfuse.callback import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []
