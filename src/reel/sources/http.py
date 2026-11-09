"""Shared HTTP plumbing: async client, token-bucket rate limiting, retries with backoff."""

from __future__ import annotations

import asyncio
import time

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

USER_AGENT = "reel/0.1 (personal film research project)"


class RateLimiter:
    """Simple async token bucket: at most `rate` acquisitions per second, burst of `burst`."""

    def __init__(self, rate: float, burst: int | None = None):
        self.rate = rate
        self.capacity = burst or max(1, int(rate))
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class RetryableStatus(Exception):
    def __init__(self, response: httpx.Response):
        super().__init__(f"HTTP {response.status_code} for {response.request.url}")
        self.response = response


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableStatus | httpx.TransportError)


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: RateLimiter | None = None,
    params: dict | None = None,
    attempts: int = 5,
) -> httpx.Response:
    """GET with retry on 429/5xx/transport errors. Honors Retry-After. Returns 4xx as-is."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            if limiter:
                await limiter.acquire()
            resp = await client.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(min(int(retry_after), 60))
                raise RetryableStatus(resp)
            return resp
    raise AssertionError("unreachable")
