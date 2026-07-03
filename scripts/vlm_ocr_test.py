"""Test a vision model as page-OCR on specific PDF pages, vs the digital text layer.

Renders each page to an image, asks a VLM (via our describe_image, provider from
args) to transcribe the page (text + tables as markdown), and prints it next to the
PDF's own native text — so we can see whether the VLM fixes the font-garble + tables
that pymupdf/Paddle mishandle.

    python scripts/vlm_ocr_test.py ollama qwen3-vl:4b <pdf> 2,8        # pages (1-based)
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from backend.core.vision_client import describe_image

PROMPT = (
    "You are an OCR + document-structure engine. Transcribe ALL text on this page "
    "EXACTLY as written, preserving reading order. Render any tables as GitHub "
    "markdown tables (| col | col |). Preserve bullet lists. Do NOT summarize, "
    "explain, or add commentary — output only the transcription."
)


def render_png(page, dpi=200) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")


def run(provider: str, model: str, pdf: str, pages: list[int]) -> None:
    # ad-hoc config: point the vision client at the chosen provider/model, no pacing
    vis = {"provider": provider, "model": model, "min_interval_s": 0,
           "max_retries": 2, "timeout_s": 600}
    if provider == "openrouter":   # shortcut: OpenAI-compatible via OpenRouter base_url
        vis["provider"] = "openai"
        vis["base_url"] = "https://openrouter.ai/api/v1"
        vis["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    elif provider == "zai":        # shortcut: Z.ai (GLM) free, OpenAI-compatible (paas/v4)
        vis["provider"] = "openai"
        vis["base_url"] = "https://api.z.ai/api/paas/v4"
        vis["api_key"] = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY", "")
    cfg = {"vision": vis}
    doc = fitz.open(pdf)
    for p in pages:
        page = doc[p - 1]
        native = " ".join(page.get_text().split())
        print("\n" + "=" * 80)
        print(f"PAGE {p}")
        print("=" * 80)
        print("\n--- PDF NATIVE TEXT (first 400 chars) ---")
        print(native[:400])
        print(f"\n--- {provider}:{model} VLM TRANSCRIPTION ---")
        try:
            out = describe_image(render_png(page), PROMPT, cfg)
            print(out.strip())
        except Exception as e:
            print(f"[VLM error] {type(e).__name__}: {e}")
    doc.close()


if __name__ == "__main__":
    provider, model, pdf = sys.argv[1], sys.argv[2], sys.argv[3]
    pages = [int(x) for x in sys.argv[4].split(",")]
    run(provider, model, pdf, pages)
