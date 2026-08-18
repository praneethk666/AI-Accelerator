"""
rate_limiter.py — Sliding window rate limiter per caller identity.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory sliding window rate limiter."""

    def __init__(self):
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check_and_record(
        self,
        caller: str,
        action: str = "send_email",
        max_calls: int = 10,
        window_seconds: int = 60,
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if the caller has exceeded their rate limit.
        If under limit, records the action timestamp and returns (True, None).
        If exceeded, returns (False, error_message).
        """
        key = f"{caller}:{action}"
        now = time.time()
        window_start = now - window_seconds

        # Prune timestamps older than window
        timestamps = [t for t in self._history[key] if t > window_start]
        self._history[key] = timestamps

        if len(timestamps) >= max_calls:
            logger.warning(
                f"Rate limit exceeded | caller={caller} | action={action} | "
                f"calls={len(timestamps)}/{max_calls} in {window_seconds}s"
            )
            return (
                False,
                f"Rate limit exceeded for action '{action}'. Maximum {max_calls} requests per {window_seconds}s.",
            )

        self._history[key].append(now)
        return True, None

    def reset(self) -> None:
        """Clears all rate limit history."""
        self._history.clear()


# Global singleton instance
rate_limiter = RateLimiter()
