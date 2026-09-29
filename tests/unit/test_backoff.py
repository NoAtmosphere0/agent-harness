import random

import pytest

from harness.tools.spec import RetryPolicy


@pytest.mark.parametrize(("attempt", "cap"), [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (8, 4.0)])
def test_backoff_delay_stays_within_capped_exponential_bound(attempt: int, cap: float):
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0)
    rng = random.Random(0)

    delays = [policy.backoff_delay(attempt, rng) for _ in range(200)]

    assert all(0 <= d <= cap for d in delays)
    # Full jitter: delays spread across the range rather than clustering at the cap.
    assert min(delays) < cap * 0.2
    assert max(delays) > cap * 0.8


def test_backoff_delay_is_deterministic_for_seeded_rng():
    policy = RetryPolicy()

    first = [policy.backoff_delay(a, random.Random(42)) for a in (1, 2, 3)]
    second = [policy.backoff_delay(a, random.Random(42)) for a in (1, 2, 3)]

    assert first == second


def test_backoff_delay_rejects_attempt_zero():
    with pytest.raises(ValueError, match="1-based"):
        RetryPolicy().backoff_delay(0, random.Random())
