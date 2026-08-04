"""Tests for describe_image()'s token-budget pacing wiring (backend/core/
token_pacer.py) -- real finding, 3-Aug: gpt-4o-mini figure captioning hit 429
"tokens per min" even with request-interval pacing tuned down, since that
mechanism has no notion of per-call size. Only the NEW pacing wiring is tested
here (opt-in via vision.tpm_limit, estimate math, key construction) -- provider
dispatch/retry logic is unchanged and not re-tested.

Run: pytest tests/test_vision_client_pacing.py
"""
from unittest.mock import patch

from backend.core.vision_client import describe_image


def _config(tpm_limit=None, **extra):
    vcfg = {"provider": "openai", "model": "gpt-4o-mini", "min_interval_s": 0}
    if tpm_limit is not None:
        vcfg["tpm_limit"] = tpm_limit
    vcfg.update(extra)
    return {"vision": vcfg}


def test_token_pacer_not_invoked_when_tpm_limit_unset():
    with patch("backend.core.vision_client._describe_openai", return_value="ok"), \
         patch("backend.core.token_pacer.wait_for_tokens") as mock_wait:
        describe_image(b"fakepng", "describe this", _config())
    mock_wait.assert_not_called()


def test_token_pacer_invoked_with_provider_model_key_when_tpm_limit_set():
    with patch("backend.core.vision_client._describe_openai", return_value="ok"), \
         patch("backend.core.token_pacer.wait_for_tokens") as mock_wait:
        describe_image(b"fakepng", "describe this", _config(tpm_limit=200_000))

    mock_wait.assert_called_once()
    key, est_tokens, tpm_limit = mock_wait.call_args[0]
    assert key == "vision:openai:gpt-4o-mini"
    assert tpm_limit == 200_000
    assert est_tokens > 0


def test_larger_image_and_prompt_produce_larger_estimate():
    estimates = []

    def _capture(key, est_tokens, tpm_limit):
        estimates.append(est_tokens)

    with patch("backend.core.vision_client._describe_openai", return_value="ok"), \
         patch("backend.core.token_pacer.wait_for_tokens", side_effect=_capture):
        describe_image(b"x" * 100, "short prompt", _config(tpm_limit=200_000))
        describe_image(b"x" * 100_000, "a much longer prompt " * 50, _config(tpm_limit=200_000))

    assert estimates[1] > estimates[0]


def test_flat_est_tokens_per_call_overrides_the_computed_estimate():
    with patch("backend.core.vision_client._describe_openai", return_value="ok"), \
         patch("backend.core.token_pacer.wait_for_tokens") as mock_wait:
        describe_image(b"x" * 1_000_000, "a very long prompt " * 200,
                       _config(tpm_limit=200_000, est_tokens_per_call=1300))

    _, est_tokens, _ = mock_wait.call_args[0]
    assert est_tokens == 1300   # NOT derived from the huge image/prompt above


def test_est_response_tokens_configurable():
    estimates = []

    def _capture(key, est_tokens, tpm_limit):
        estimates.append(est_tokens)

    with patch("backend.core.vision_client._describe_openai", return_value="ok"), \
         patch("backend.core.token_pacer.wait_for_tokens", side_effect=_capture):
        describe_image(b"x" * 10, "p", _config(tpm_limit=200_000, est_response_tokens=100))
        describe_image(b"x" * 10, "p", _config(tpm_limit=200_000, est_response_tokens=5000))

    assert estimates[1] - estimates[0] == 4900
