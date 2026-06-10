"""Shared vision helper. Both categorize_tool and vision_enrichment_tool
call describe_image() — Gemma access lives here and nowhere else.

Supported providers (set via config["vision"]["provider"]):
  google  — Gemma via Google AI Studio (requires GOOGLE_API_KEY in env)
  ollama  — self-hosted Gemma via Ollama (no API key, local only)

Usage:
    from backend.core.vision_client import describe_image
    text = describe_image(image_bytes, "Describe this diagram.", config)
"""
from __future__ import annotations
import os


def describe_image(image_bytes: bytes, prompt: str, config: dict) -> str:
    provider = config.get("vision", {}).get("provider", "google")
    model = config.get("vision", {}).get("model", "gemma-4-26b-a4b-it")

    if provider == "google":
        return _describe_google(image_bytes, prompt, model, config)
    if provider == "ollama":
        return _describe_ollama(image_bytes, prompt, model, config)

    raise ValueError(
        f"Unknown vision provider {provider!r}. "
        "Set config['vision']['provider'] to 'google' or 'ollama'."
    )


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

    _trace_start(prompt, model, config)
    response = client.generate_content([prompt, image])
    result = response.text
    _trace_end(result, config)
    return result


def _describe_ollama(image_bytes: bytes, prompt: str, model: str, config: dict) -> str:
    import base64
    import ollama

    b64 = base64.b64encode(image_bytes).decode()
    _trace_start(prompt, model, config)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
    )
    result = response["message"]["content"]
    _trace_end(result, config)
    return result


def _trace_start(prompt: str, model: str, config: dict) -> None:
    try:
        from langfuse import Langfuse
        _trace_start._lf = Langfuse()
        _trace_start._gen = _trace_start._lf.generation(
            name="vision_describe_image",
            model=model,
            input=prompt,
        )
    except Exception:
        pass


def _trace_end(result: str, config: dict) -> None:
    try:
        gen = getattr(_trace_start, "_gen", None)
        if gen:
            gen.end(output=result)
    except Exception:
        pass
