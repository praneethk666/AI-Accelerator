from google import genai
from google.genai import types

_client = genai.Client()

def describe_image_with_gemma(
    image_bytes: bytes,
    prompt: str | None = None,
    model: str = "gemini-3.5-flash"
) -> str:
    # Detect MIME type
    if image_bytes[:4] == b'\x89PNG':
        mime = "image/png"
    else:
        mime = "image/jpeg"

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

    if prompt is None:
        prompt = (
            "Describe this image in detail. "
            "Include the type of diagram/figure, main components, labels, and relationships."
        )

    contents = [prompt, image_part]

    try:
        response = _client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"    ⚠️ Vision API error: {e}")
        return ""