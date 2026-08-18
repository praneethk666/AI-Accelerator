"""
datetime_tool.py — Tool handler for get_current_datetime(timezone?).
Provides host system date and time with strict timezone validation.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from pydantic import BaseModel, Field
import pytz

logger = logging.getLogger(__name__)


class GetCurrentDateTimeInput(BaseModel):
    timezone: Optional[str] = Field(
        default=None,
        description="Optional IANA timezone name (e.g. 'UTC', 'America/New_York', 'Asia/Kolkata', 'Europe/London'). Defaults to system local time if omitted.",
    )


def validate_and_resolve_timezone(tz_name: Optional[str]):
    """
    Validates timezone name and returns a tuple: (timezone_info, display_name, source_label).
    If tz_name is omitted or empty, reads directly from the host laptop's local system clock.
    """
    if not tz_name or tz_name.strip() == "":
        local_now = datetime.now().astimezone()
        local_tz = local_now.tzinfo
        local_tz_name = local_now.tzname() or "Local"
        offset_str = local_now.strftime("%z")
        display = f"Host System Local Time ({local_tz_name}, UTC{offset_str[:3]}:{offset_str[3:]})"
        return local_tz, display, "Host Laptop System Clock"

    tz_clean = tz_name.strip()
    try:
        # Common aliases handling
        if tz_clean.upper() in ("UTC", "GMT", "Z"):
            return timezone.utc, "UTC", "UTC Standard"
        tz = pytz.timezone(tz_clean)
        return tz, tz_clean, f"IANA Timezone ({tz_clean})"
    except Exception:
        raise ValueError(
            f"Invalid timezone '{tz_clean}'. Must be a valid IANA timezone name such as 'UTC', 'America/New_York', 'Asia/Kolkata', or 'Europe/London'."
        )


async def get_current_datetime_handler(input_data: GetCurrentDateTimeInput, caller: Optional[str] = None) -> dict:
    """
    Core business logic for get_current_datetime.
    Directly queries host laptop hardware clock when called without parameters.
    """
    try:
        tz, tz_display, source_label = validate_and_resolve_timezone(input_data.timezone)
        now = datetime.now(tz)

        logger.info(
            f"get_current_datetime invoked | source={source_label} | timezone={tz_display} | caller={caller or 'unknown'}"
        )

        return {
            "status": "success",
            "source": source_label,
            "datetime_iso": now.isoformat(),
            "timezone": tz_display,
            "utc_offset": now.strftime("%z"),
            "epoch_timestamp": now.timestamp(),
            "formatted": now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
        }
    except ValueError as val_err:
        logger.warning(f"get_current_datetime validation error: {val_err}")
        return {
            "status": "failed",
            "error": "invalid_timezone",
            "message": str(val_err),
        }
    except Exception as exc:
        logger.error(f"Unexpected error in get_current_datetime: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": "internal_error",
            "message": "Failed to calculate current datetime.",
        }
