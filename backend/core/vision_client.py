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
import os
from contextlib import contextmanager


def describe_image(image_bytes: bytes, prompt: str, config: dict) -> str:
    vcfg = config.get("vision", {})
    provider = vcfg.get("provider", "google")
    model = vcfg.get("model", "gemma-3-27b-it")

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


def _describe_openai(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    """OpenAI-compatible multimodal: send the image as a base64 data URL."""
    import base64
    from openai import OpenAI

    vcfg = config.get("vision", {})
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
        t["output"] = result
    return result


def _describe_google(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    import google.generativeai as genai
    import PIL.Image
    import io

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    image = PIL.Image.open(io.BytesIO(image_bytes))

    with _trace(prompt, model) as t:
        result = client.generate_content([prompt, image]).text
        t["output"] = result
    return result


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


@contextmanager
def _trace(prompt: str, model: str):
    """Best-effort Langfuse generation span for one vision call (langfuse v4).

    Thread-safe (each call owns its observation — vision runs many in parallel)
    and never logs the image bytes, only prompt+model+output. No-op when Langfuse
    isn't configured.
    """
    box = {"output": None}
    cm = obs = None
    try:
        from langfuse import get_client
        cm = get_client().start_as_current_observation(
            as_type="generation", name="vision_describe_image",
            model=model, input=prompt,
        )
        obs = cm.__enter__()
    except Exception:
        cm = None
    try:
        yield box
    finally:
        try:
            if obs is not None and box["output"] is not None:
                obs.update(output=box["output"])
            if cm is not None:
                cm.__exit__(None, None, None)
        except Exception:
            pass
