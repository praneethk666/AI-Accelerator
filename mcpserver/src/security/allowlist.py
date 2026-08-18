"""
allowlist.py — Strict recipient email and domain allowlist validation.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def is_valid_email_syntax(email: str) -> bool:
    """Checks if email string matches standard email syntax."""
    return bool(EMAIL_REGEX.match(email.strip()))


def is_email_allowlisted(email: str, allowlist: List[str]) -> bool:
    """
    Checks if an email address is in the allowlist.
    Supports exact matches (user@domain.com) and wildcards (*@domain.com or @domain.com).
    """
    email_clean = email.strip().lower()
    if "@" not in email_clean:
        return False

    _, domain = email_clean.split("@", 1)

    for entry in allowlist:
        entry_clean = entry.strip().lower()
        if not entry_clean:
            continue

        # Exact match
        if email_clean == entry_clean:
            return True

        # Wildcard domain match (*@example.com or @example.com)
        if entry_clean.startswith("*@"):
            allowed_domain = entry_clean[2:]
            if domain == allowed_domain:
                return True
        elif entry_clean.startswith("@"):
            allowed_domain = entry_clean[1:]
            if domain == allowed_domain:
                return True

    return False


def validate_email_allowlist(
    email: str,
    allowlist: List[str],
    caller: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validates that an email address is syntactically valid and allowed.
    Returns (is_allowed, error_reason).
    """
    if not email or not isinstance(email, str):
        return False, "Recipient email address must be a non-empty string."

    email_clean = email.strip()
    if not is_valid_email_syntax(email_clean):
        return False, f"Invalid email format: '{email_clean}'"

    if not is_email_allowlisted(email_clean, allowlist):
        logger.warning(
            f"Email send rejected: Recipient '{email_clean}' is not in the allowlist | caller={caller or 'unknown'}"
        )
        return (
            False,
            f"Recipient '{email_clean}' is not in the authorized recipient allowlist.",
        )

    return True, None
