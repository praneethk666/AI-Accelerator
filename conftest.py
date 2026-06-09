"""pytest configuration. Ensures the repo root is on sys.path so all tests
can do `from backend.core.xxx import ...` without any install step."""
import os
import sys
from dotenv import load_dotenv

# Load .env file so GOOGLE_API_KEY and other env vars are available
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
