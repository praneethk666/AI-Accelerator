# backend/vision/vision_client.py

import os

from google import genai
from dotenv import load_dotenv

from .prompts import VISION_PROMPT

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY not found in .env or environment"
    )

client = genai.Client(api_key=api_key)


class VisionClient:

    def __init__(self, model_name=None):

        if model_name is None:
            model_name = os.getenv(
                "VISION_MODEL",
                "gemma-4-26b-a4b-it"
            )

        self.model_name = model_name

    def describe(
        self,
        image_bytes,
        config,
    ):
        image_part = {
            "mime_type": "image/png",
            "data": image_bytes,
        }

        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                VISION_PROMPT,
                image_part,
            ],
        )

        return response.text