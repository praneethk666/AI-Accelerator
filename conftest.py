"""pytest configuration. Ensures the repo root is on sys.path so all tests
can do `from backend.core.xxx import ...` without any install step."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
