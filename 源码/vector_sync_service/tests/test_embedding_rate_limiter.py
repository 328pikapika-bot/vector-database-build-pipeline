from __future__ import annotations

import pytest

from app.services.embedding_rate_limiter import (
    AsyncTextRateLimiter,
    build_text_rate_limiter,
)


@pytest.mark.asyncio
async def test_token_bucket_waits_for_text_items():
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = AsyncTextRateLimiter(
        10.0,
        burst=2,
        clock=lambda: now[0],
        sleep=fake_sleep,
    )

    await limiter.acquire(2)
    await limiter.acquire(1)

    assert sleeps == [pytest.approx(0.1)]


def test_rate_limiter_is_disabled_by_default():
    assert build_text_rate_limiter(enabled=False, rate_per_second=80) is None
    assert build_text_rate_limiter(enabled=True, rate_per_second=0) is None

