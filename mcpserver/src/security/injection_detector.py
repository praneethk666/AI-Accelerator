"""
injection_detector.py — Prompt injection detection and security event auditing.
"""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Heuristic & regex patterns targeting direct & indirect prompt injection techniques
PROMPT_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", "IGNORE_PREVIOUS_INSTRUCTIONS"),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", "DISREGARD_INSTRUCTIONS"),
    (r"override\s+(all\s+)?(system|safety|security)\s+(prompts?|instructions?|rules?)", "OVERRIDE_SYSTEM_PROMPT"),
    (r"system\s*:\s*override", "SYSTEM_OVERRIDE_COLON"),
    (r"you\s+are\s+now\s+(in\s+)?(dan|unrestricted|god)\s+mode", "DAN_MODE_INJECTION"),
    (r"act\s+as\s+an?\s+unrestricted\s+ai", "UNRESTRICTED_AI_JAILBREAK"),
    (r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions|internal\s+rules)", "PROMPT_LEAK_ATTEMPT"),
    (r"output\s+(the\s+)?(system\s+prompt|initial\s+instructions)", "OUTPUT_SYSTEM_PROMPT"),
    (r"<\s*\|\s*im_start\s*\|\s*>", "IM_START_DELIMITER"),
    (r"<\s*\|\s*im_end\s*\|\s*>", "IM_END_DELIMITER"),
    (r"\[\s*INST\s*\]|\[\s*/\s*INST\s*\]", "LLAMA_INST_DELIMITER"),
    (r"<<\s*SYS\s*>>|<\s*/\s*SYS\s*>>", "LLAMA_SYS_DELIMITER"),
    (r"###\s*(System|Instruction|Assistant|Human)\s*:", "MARKDOWN_PROMPT_DELIMITER"),
    (r"bypass\s+(all\s+)?(safety|guardrails?|filters?)", "BYPASS_SAFETY_FILTERS"),
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), name) for p, name in PROMPT_INJECTION_PATTERNS]


def detect_prompt_injection(
    text: str,
    field_name: str = "text",
    caller: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Scans the given text for prompt injection signatures.
    Returns:
        (is_injection, reason_message)
    """
    if not text:
        return False, None

    for pattern, name in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_snippet = match.group(0)
            logger.warning(
                f"Prompt injection attempt detected in '{field_name}' | "
                f"caller={caller or 'unknown'} | pattern={name} | snippet={matched_snippet!r}",
                extra={
                    "security_alert": True,
                    "event_type": "PROMPT_INJECTION_BLOCKED",
                    "caller": caller or "unknown",
                    "field": field_name,
                    "pattern": name,
                    "snippet": matched_snippet,
                },
            )
            return True, f"Prompt injection detected in field '{field_name}' (rule: {name})"

    return False, None
