"""Token-bucket rate limiter for GitHub API requests."""

from __future__ import annotations

import asyncio
import time


class TokenBucketLimiter:
    """Async token-bucket rate limiter.

    Defaults to GitHub's 5000 requests/hour budget, reserving capacity
    for the polling workload (~1200 req/hr target).
    """

    def __init__(
        self,
        rate: float = 1200 / 3600,  # tokens per second
        capacity: int = 30,
    ) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        async with self._lock:
            self._refill()
            while self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now
