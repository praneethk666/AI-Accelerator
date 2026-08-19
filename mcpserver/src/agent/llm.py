import json
import os
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.model_name = model_name 
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json"}
            )
        except ImportError:
            raise RuntimeError("google-generativeai is not installed. Please run: pip install google-generativeai")

    def generate_json(self, prompt: str) -> dict:
        """Uses native Gemini library to generate a JSON response."""
        response = self.model.generate_content(prompt)
        text_content = response.text
        # Sometimes models wrap JSON in markdown block, strip it if so
        if text_content.startswith("```json"):
            text_content = text_content[7:-3].strip()
        return json.loads(text_content) 
 
class GroqClient:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        self.model_name = model_name
        try:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY environment variable not set.")
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise RuntimeError("groq is not installed. Please run: pip install groq")

    def generate_json(self, prompt: str) -> dict:
        """Uses native Groq library to generate a JSON response."""
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        # Strip potential markdown code blocks returned by LLM
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        try:
            return json.loads(content)
        except json.JSONDecodeError as err:
            logger.error(f"Groq JSON parse failed on string: {content}")
            raise err
