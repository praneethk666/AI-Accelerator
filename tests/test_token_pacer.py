"""Tests for backend.core.token_pacer -- sliding-window TOKEN-budget pacing,
built to fix a real 3-Aug finding: gpt-4o-mini figure captioning hit 429 "tokens
per min" even with request-interval pacing (backend/core/pacing.py) tuned down,
because that mechanism paces REQUEST rate and has no notion of per-call size.

Run: pytest tests/test_token_pacer.py
"""
import time
from unittest.mock import patch

from backend.core import token_pacer


def test_disabled_when_tpm_limit_unset():
    t0 = time.monotonic()
    token_pacer.wait_for_tokens("k1", 999_999, tpm_limit=0)
    assert time.monotonic() - t0 < 0.05   # no blocking at all


def test_disabled_when_tpm_limit_negative():
    t0 = time.monotonic()
    token_pacer.wait_for_tokens("k2", 999_999, tpm_limit=-1)
    assert time.monotonic() - t0 < 0.05


def test_noop_for_zero_or_negative_estimate():
    t0 = time.monotonic()
    token_pacer.wait_for_tokens("k3", 0, tpm_limit=100)
    token_pacer.wait_for_tokens("k3", -5, tpm_limit=100)
    assert time.monotonic() - t0 < 0.05


def test_calls_within_budget_do_not_block():
    key = "k4"
    t0 = time.monotonic()
    for _ in range(5):
        token_pacer.wait_for_tokens(key, 1000, tpm_limit=100_000)
    assert time.monotonic() - t0 < 0.1


def test_call_exceeding_remaining_budget_blocks_until_window_frees_up():
    key = "k5"
    budget = token_pacer._TokenBudget(window_s=0.3)   # short window for a fast test
    with patch.object(token_pacer, "_budgets", {key: budget}):
        t0 = time.monotonic()
        token_pacer.wait_for_tokens(key, 80, tpm_limit=100)   # consumes 80/100
        token_pacer.wait_for_tokens(key, 50, tpm_limit=100)   # needs the window to age out first
        elapsed = time.monotonic() - t0
    assert elapsed >= 0.25   # actually waited for (most of) the window


def test_independent_keys_do_not_share_budget():
    budget_a = token_pacer._TokenBudget(window_s=5.0)
    budget_b = token_pacer._TokenBudget(window_s=5.0)
    with patch.object(token_pacer, "_budgets", {"a": budget_a, "b": budget_b}):
        t0 = time.monotonic()
        token_pacer.wait_for_tokens("a", 100, tpm_limit=100)   # fills key "a" only
        token_pacer.wait_for_tokens("b", 100, tpm_limit=100)   # key "b" is fresh -> no block
        assert time.monotonic() - t0 < 0.1


def test_concurrent_callers_serialize_through_the_same_budget():
    import threading
    key = "k6"
    budget = token_pacer._TokenBudget(window_s=0.3)
    results = []
    with patch.object(token_pacer, "_budgets", {key: budget}):
        def call():
            token_pacer.wait_for_tokens(key, 60, tpm_limit=100)
            results.append(time.monotonic())

        threads = [threading.Thread(target=call) for _ in range(3)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 3 calls of 60 tokens each against a 100-token budget -> at least one pair
    # must be separated by close to a full window (can't both fit at once).
    assert max(results) - t0 >= 0.2
