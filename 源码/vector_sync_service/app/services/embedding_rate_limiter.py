"""Per-worker text-embedding rate limiting.

The three ingestion workers run in separate containers, so each worker needs
an explicit budget. The budgets configured in the three Stack files are
summed below the measured capacity of the shared embedding endpoint.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional


class AsyncTextRateLimiter:
    """Token bucket measured in embedding input texts per second."""

    def __init__(
        self,
        rate_per_second: float,
        *,
        burst: int = 32,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.rate_per_second = max(float(rate_per_second or 0.0), 0.0)
        self.capacity = max(int(burst or 0), 1)
        self._tokens = float(self.capacity)
        self._updated_at = clock()
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_second > 0.0

    async def acquire(self, amount: int) -> None:
        """Wait until ``amount`` input-text tokens are available."""
        requested = max(int(amount or 0), 0)
        if requested <= 0 or not self.enabled:
            return

        # A batch larger than burst must still be serviceable. Waiting for the
        # full batch keeps the configured average rate while avoiding spikes.
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(now - self._updated_at, 0.0)
                capacity = max(float(self.capacity), float(requested))
                self._tokens = min(
                    capacity,
                    self._tokens + elapsed * self.rate_per_second,
                )
                self._updated_at = now
                if self._tokens >= requested:
                    self._tokens -= requested
                    return
                wait_seconds = (requested - self._tokens) / self.rate_per_second
            await self._sleep(max(wait_seconds, 0.001))


def build_text_rate_limiter(
    *,
    enabled: bool,
    rate_per_second: float,
    burst: int = 32,
) -> Optional[AsyncTextRateLimiter]:
    """Build a limiter only when a Stack explicitly enables a text budget."""
    if not enabled or float(rate_per_second or 0.0) <= 0.0:
        return None
    return AsyncTextRateLimiter(rate_per_second, burst=burst)

