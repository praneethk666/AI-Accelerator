import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .prompts import VISION_PROMPT

load_dotenv()

class VisionClient:
    def __init__(self, model_name="gemma-4-26b-a4b-it"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def describe(self, image_bytes: bytes, config: dict | None = None) -> str:
        mime = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[VISION_PROMPT, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            raw = response.text.strip() if response.text else ""
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return json.dumps(parsed)
            return raw
        except Exception as e:
            print(f"\n⚠️ Vision API error: {e}")
            return ""