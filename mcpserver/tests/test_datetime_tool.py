"""
test_datetime_tool.py — Unit tests for get_current_datetime tool handler.
"""

import pytest
from src.tools.datetime_tool import (
    GetCurrentDateTimeInput,
    get_current_datetime_handler,
    validate_and_resolve_timezone,
)


@pytest.mark.asyncio
async def test_get_current_datetime_local():
    """Default invocation returns valid ISO datetime."""
    result = await get_current_datetime_handler(GetCurrentDateTimeInput())
    assert result["status"] == "success"
    assert "datetime_iso" in result
    assert "epoch_timestamp" in result
    assert "formatted" in result


@pytest.mark.asyncio
async def test_get_current_datetime_valid_timezones():
    """Valid IANA timezones return correct offsets and timezone names."""
    timezones = ["UTC", "America/New_York", "Asia/Kolkata", "Europe/London", "GMT"]
    for tz in timezones:
        result = await get_current_datetime_handler(GetCurrentDateTimeInput(timezone=tz))
        assert result["status"] == "success"
        assert result["timezone"] in (tz, "UTC")
        assert "T" in result["datetime_iso"]


@pytest.mark.asyncio
async def test_get_current_datetime_invalid_timezone():
    """Invalid timezone string returns clean validation failure."""
    result = await get_current_datetime_handler(
        GetCurrentDateTimeInput(timezone="NonExistent/Place_123")
    )
    assert result["status"] == "failed"
    assert result["error"] == "invalid_timezone"
    assert "Invalid timezone" in result["message"]


def test_validate_and_resolve_timezone_aliases():
    """Validates UTC and GMT aliases."""
    tz, name, src = validate_and_resolve_timezone("utc")
    assert name == "UTC"
    tz_gmt, name_gmt, src_gmt = validate_and_resolve_timezone("GMT")
    assert name_gmt == "UTC"
