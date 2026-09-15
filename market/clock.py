"""Clock abstraction.

The minute cadence is timing-critical, and timing bugs are the hardest kind to
find by running the bot for an hour. So the scheduler, the candle aggregator
and the feed all take a ``Clock`` instead of calling ``time.time()`` — tests
drive a ``VirtualClock`` and step through hours of candles in milliseconds.

``RealClock`` is what production uses: wall-clock time, real sleeps.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Current time in epoch seconds."""
        ...

    async def sleep_until(self, ts: float) -> None:
        """Sleep until epoch second ``ts`` (returns immediately if already past)."""
        ...


class RealClock:
    """Wall-clock time."""

    def now(self) -> float:
        return time.time()

    async def sleep_until(self, ts: float) -> None:
        delay = ts - self.now()
        if delay > 0:
            await asyncio.sleep(delay)


class VirtualClock:
    """A clock that only moves when the test advances it.

    ``sleep_until`` parks the caller on a future that ``advance`` completes once
    virtual time passes the deadline — so a scheduler loop awaiting a minute
    boundary blocks exactly as it would in production, but a test can jump it
    forward instantly.
    """

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start
        self._waiters: list[tuple[float, asyncio.Future]] = []

    def now(self) -> float:
        return self._now

    async def sleep_until(self, ts: float) -> None:
        if ts <= self._now:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append((ts, fut))
        await fut

    def _wake_due(self) -> None:
        due = [(ts, f) for ts, f in self._waiters if ts <= self._now]
        for ts, fut in due:
            self._waiters.remove((ts, fut))
            if not fut.done():
                fut.set_result(None)

    async def _drain(self, rounds: int = 12) -> None:
        """Give woken tasks enough loop turns to run to their next await."""
        for _ in range(rounds):
            await asyncio.sleep(0)

    async def advance(self, seconds: float, steps: int = 1) -> None:
        """Move virtual time forward by ``seconds``, waking sleepers on the way."""
        steps = max(1, steps)
        step = seconds / steps
        for _ in range(steps):
            self._now += step
            self._wake_due()
            await self._drain()

    async def advance_to(self, ts: float) -> None:
        """Advance in 1-second steps until virtual time reaches ``ts``."""
        while self._now < ts:
            await self.advance(min(1.0, ts - self._now))
