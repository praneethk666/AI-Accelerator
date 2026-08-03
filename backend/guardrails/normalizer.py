"""
normalizer.py — text normalization pipeline before any regex or heuristic check.

Runs on a COPY of the input. The original is passed through the pipeline unchanged.
Never mutates data the LLM will actually use.

Steps (in order):
  1. NFKC Unicode normalization — converts visually similar chars to canonical form
     e.g. Ｉｇｎｏｒｅ → Ignore, Cyrillic "о" → Latin "o"
  2. Zero-width character removal — strips invisible characters used to bypass regex
     (U+200B, U+200C, U+200D, U+FEFF, etc.)
  3. HTML entity unescape — &#105;&#103;&#110;&#111;&#114;&#101; → ignore
  4. Markdown code fence strip — removes ```...``` wrappers from injection attempts
     embedded in code blocks
"""
from __future__ import annotations

import html
import re
import unicodedata

# Zero-width + direction override + soft-hyphen characters
_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad]"
)

# Markdown/code-fence wrappers
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9]*\n?(.*?)```", re.DOTALL)


def normalize(text: str) -> str:
    """Return normalized copy of *text* suitable for safety scanning.

    This must be called on a COPY — never on the string that will be sent to
    the LLM or saved to the database.
    """
    if not isinstance(text, str):
        return text

    # 1. NFKC — canonical decomposition + compatibility composition
    out = unicodedata.normalize("NFKC", text)

    # 2. Strip zero-width / direction-override characters
    out = _ZERO_WIDTH.sub("", out)

    # 3. Unescape HTML entities
    out = html.unescape(out)

    # 4. Flatten code-fence wrappers (keep inner content for scanning)
    out = _CODE_FENCE.sub(lambda m: m.group(1), out)

    return out
