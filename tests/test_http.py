import asyncio
import time

from reel.sources.http import RateLimiter


def test_rate_limiter_respects_window():
    """burst=1 at rate r => at most r*window + 1 calls in any window (TMDB: 40 / 10s)."""
    limiter = RateLimiter(rate=200, burst=1)
    stamps: list[float] = []

    async def run():
        await asyncio.gather(*(limiter.acquire() for _ in range(41)))

    async def timed():
        for _ in range(41):
            await limiter.acquire()
            stamps.append(time.monotonic())

    asyncio.run(run())  # also proves reuse across event loops works
    asyncio.run(timed())
    # 41 calls at 200/s with no burst take at least 40/200 s.
    assert stamps[-1] - stamps[0] >= 40 / 200 * 0.95
