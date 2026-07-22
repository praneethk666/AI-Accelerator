"""Shared vision helper. Both categorize_tool and vision_enrichment_tool
call describe_image() — vision-model access lives here and nowhere else.

Providers (config["vision"]["provider"]):
  google  — Gemma/Gemini via Google AI Studio (GOOGLE_API_KEY)
  ollama  — self-hosted VLM via Ollama (local, no key)
  openai  — any OpenAI-COMPATIBLE multimodal endpoint via base_url. Covers OpenAI
            GPT-4o-vision, NVIDIA NIM VLMs, OpenRouter vision, vLLM — point
            config["vision"]["base_url"] at the provider.

config["vision"] keys: provider, model, base_url (openai), api_key ("${ENV}").

Usage:
    from backend.core.vision_client import describe_image
    text = describe_image(image_bytes, "Describe this diagram.", config)
"""
from __future__ import annotations
import logging
import os
import random
import re
import time
from backend.core.tracing import traced_tool

logger = logging.getLogger(__name__)

# Substrings that mark a transient/retryable vision error: provider rate limits
# (429 / quota) and transient server errors (500/503). Google's free tier is only
# 15 requests/min, so bursts of image calls WILL hit 429 — we back off and retry.
_RETRYABLE = ("429", "quota", "rate limit", "resource_exhausted", "resourceexhausted",
              "500", "internal error", "503", "unavailable", "overloaded")


def describe_image(image_bytes: bytes, prompt: str, config: dict) -> str:
    # Accept either the FULL config (categorize passes this) or the vision
    # sub-dict (vision_enrichment passes config["vision"]). Without this, the
    # sub-dict caller falls through to the default model and 404s.
    vcfg = config.get("vision") or config
    provider = vcfg.get("provider", "google")
    model = vcfg.get("model", "gemma-4-31b-it")
    max_retries = int(vcfg.get("max_retries", 5))

    # Proactively pace calls to stay under the free-tier RPM (e.g. Gemma ~15/min ->
    # ~4s). Spaces distinct calls across the vision worker threads; the retry backoff
    # below remains the safety net for any 429 that slips through.
    from backend.core import pacing
    pacing.pace("vision", float(vcfg.get("min_interval_s", 0) or 0))

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if provider == "google":
                return _describe_google(image_bytes, prompt, model, config)
            if provider == "ollama":
                return _describe_ollama(image_bytes, prompt, model, config)
            if provider == "openai":
                return _describe_openai(image_bytes, prompt, model, config)
            raise ValueError(
                f"Unknown vision provider {provider!r}. "
                "Use 'google', 'ollama', or 'openai' (openai + base_url covers any "
                "OpenAI-compatible vision API such as NVIDIA NIM / GPT-4o-vision)."
            )
        except ValueError:
            raise  # config error — not retryable
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = _retry_after(exc, attempt)
            logger.warning(
                "vision call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, max_retries, delay, str(exc)[:140],
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in _RETRYABLE)


def _retry_after(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt. Honor the server's own
    `retry_delay { seconds: N }` (Google sends one on 429); otherwise exponential
    backoff with jitter, capped so a stuck doc can't hang forever."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
    if m:
        return min(float(m.group(1)) + random.uniform(0, 1.5), 45.0)
    return min((2 ** attempt) + random.uniform(0, 1.0), 30.0)


def _describe_openai(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    """OpenAI-compatible multimodal: send the image as a base64 data URL."""
    import base64
    from openai import OpenAI

    vcfg = config.get("vision") or config
    api_key = os.environ.get("OPENAI_API_KEY")
    cfg_key = vcfg.get("api_key")
    if cfg_key and not str(cfg_key).startswith("${"):
        api_key = cfg_key

    client = OpenAI(api_key=api_key, base_url=vcfg.get("base_url") or None)
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()

    with _trace(prompt, model) as t:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        result = resp.choices[0].message.content
        try:
            from backend.core import usage
            u = getattr(resp, "usage", None)
            if u is not None:
                usage.record("vision", getattr(u, "prompt_tokens", 0),
                             getattr(u, "completion_tokens", 0))
        except Exception:
            pass
        t["output"] = result
    return result


def _describe_google(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    # New google-genai SDK (the old google.generativeai is deprecated AND can't set
    # thinking config). Gemma 4 has a built-in reasoning mode that's ON by default on
    # AI Studio, so it "thinks out loud" before the JSON; thinking_level=MINIMAL
    # suppresses it at the source → pure JSON, and fewer tokens (faster/cheaper).
    # block_builder._extract_json stays as a backup for any reply that still narrates.
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    contents = [prompt, image_part]

    with _trace(prompt, model) as t:
        try:
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                ),
            )
        except Exception:
            # A model may reject thinking_config or the mime type (e.g. non-Gemma) —
            # fall back to a plain call; the JSON extractor still cleans the reply.
            resp = client.models.generate_content(model=model, contents=contents)
        result = resp.text
        _record_google_usage(resp)
        t["output"] = result
    return result


def _record_google_usage(resp) -> None:
    """Record Gemini/Gemma token usage (response.usage_metadata) into the run sink."""
    try:
        from backend.core import usage
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage.record(
                "vision",
                getattr(um, "prompt_token_count", 0),
                getattr(um, "candidates_token_count", 0),
            )
    except Exception:
        pass


def _describe_ollama(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    import base64
    import ollama

    b64 = base64.b64encode(image_bytes).decode()
    with _trace(prompt, model) as t:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
        )
        result = response["message"]["content"]
        t["output"] = result
    return result

def _trace(prompt: str, model: str):
    """Span for one vision call (thread-safe — vision runs many in parallel).
    Delegates to traced_tool, which is a no-op unless
    OTEL_EXPORTER_OTLP_ENDPOINT is set. Never logs the image bytes, only
    prompt+model+output, matching the old Langfuse version's privacy behavior."""
    return traced_tool(f"vision:{model}", input=prompt)