"""Tests for describe_image()'s OpenAI-compatible client timeout wiring.

Real gap found live, 4-Aug: timeout_s has been set in config (e.g.
extraction.cad.vision.timeout_s) in multiple places for a while, but the
client never read it -- a hung provider call fell back to the openai SDK's
own default (600s) instead of the configured value, silently.
"""
from unittest.mock import MagicMock, patch

from backend.core.vision_client import describe_image


def _fake_openai_client():
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="described"))]
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


def test_timeout_s_passed_through_to_openai_client():
    fake_client = _fake_openai_client()
    with patch("openai.OpenAI", return_value=fake_client) as mock_openai_cls:
        describe_image(b"fakepng", "describe this",
                       {"vision": {"provider": "openai", "model": "gpt-4o-mini",
                                  "timeout_s": 45, "min_interval_s": 0}})
    _, kwargs = mock_openai_cls.call_args
    assert kwargs["timeout"] == 45.0


def test_no_timeout_configured_leaves_client_default():
    fake_client = _fake_openai_client()
    with patch("openai.OpenAI", return_value=fake_client) as mock_openai_cls:
        describe_image(b"fakepng", "describe this",
                       {"vision": {"provider": "openai", "model": "gpt-4o-mini",
                                  "min_interval_s": 0}})
    _, kwargs = mock_openai_cls.call_args
    assert kwargs["timeout"] is None


def test_client_level_retries_disabled_to_avoid_double_retry():
    # describe_image() already retries at its own level; letting the openai
    # SDK retry too would mean retrying twice for one logical attempt.
    fake_client = _fake_openai_client()
    with patch("openai.OpenAI", return_value=fake_client) as mock_openai_cls:
        describe_image(b"fakepng", "describe this",
                       {"vision": {"provider": "openai", "model": "gpt-4o-mini",
                                  "min_interval_s": 0}})
    _, kwargs = mock_openai_cls.call_args
    assert kwargs["max_retries"] == 0
