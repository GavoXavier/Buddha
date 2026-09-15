"""Simulated tick feed for offline runs and tests.

Produces a price for each symbol from a slow cycle (which creates genuine
trends and reversals for the indicators to find) plus a mean-reverting random
walk. It pushes *ticks* rather than candles, exactly like the live feed, so the
whole pipeline — aggregation, engine, ranking, scheduler — runs unmodified.

Driven by an injected ``Clock``, so tests can pair it with a ``VirtualClock``
and replay hours of market in milliseconds.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Optional, Sequence

from market.clock import Clock, RealClock
from market.universe import AssetMeta
from signals.engine import Candle

from .base import DataFeed, TickHandler

TWO_PI = 2.0 * math.pi


class _SymbolState:
    def __init__(self, symbol: str, base: float, offset: float) -> None:
        self.symbol = symbol
        self.base = base
        self.offset = offset
        self.walk = 0.0


class SimulatedFeed(DataFeed):
    def __init__(self, symbols: Optional[Sequence[str]] = None, seed: int = 7,
                 clock: Optional[Clock] = None, ticks_per_second: float = 2.0,
                 cycle_minutes: float = 17.0, amplitude_bps: float = 9.0,
                 noise_bps: float = 0.7, payout: int = 85) -> None:
        self._symbols = list(symbols or ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"])
        self._rng = random.Random(seed)
        self._clock = clock or RealClock()
        self._tick_interval = 1.0 / max(ticks_per_second, 0.01)
        self._cycle_seconds = cycle_minutes * 60.0
        self._amplitude_bps = amplitude_bps
        self._noise_bps = noise_bps
        self._payout = payout
        self._handler: Optional[TickHandler] = None
        self._running = False
        self._start: Optional[float] = None
        self._last_tick_at = 0.0
        self._state: dict[str, _SymbolState] = {}
        self._build_state()

    def _build_state(self) -> None:
        for i, symbol in enumerate(self._symbols):
            rng = random.Random(f"{symbol}:{self._rng.random()}")
            if symbol.startswith("USD") or symbol.startswith("EUR"):
                base = 1.0 + rng.random() * 0.6
            elif "JPY" in symbol:
                base = 100.0 + rng.random() * 60.0
            else:
                base = 0.7 + rng.random() * 0.8
            self._state[symbol] = _SymbolState(symbol, base, rng.random())

    # -- DataFeed ------------------------------------------------------------
    def set_tick_handler(self, handler: TickHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        self._start = self._clock.now()
        self._last_tick_at = self._start
        self._running = True

    async def close(self) -> None:
        self._running = False

    async def asset_meta(self) -> dict[str, AssetMeta]:
        return {
            symbol: AssetMeta(symbol=symbol, payout=self._payout, digits=5,
                              is_otc=symbol.endswith("_otc"), active=True)
            for symbol in self._symbols
        }

    async def subscribe(self, symbols: Sequence[str]) -> None:
        for symbol in symbols:
            if symbol not in self._state:
                rng = random.Random(f"{symbol}:{self._rng.random()}")
                self._state[symbol] = _SymbolState(symbol, 1.0 + rng.random(), rng.random())

    async def get_candles(self, symbol: str, period: int, count: int) -> list[Candle]:
        """Synthetic bars from the same process, for backtest smoke tests."""
        now = self._clock.now()
        candles: list[Candle] = []
        for i in range(count, 0, -1):
            bucket = int(math.floor((now - i * period) / period) * period)
            prices = [self._price_at(symbol, bucket + j) for j in
                      range(0, period, max(1, period // 10))]
            candles.append(Candle(float(bucket), prices[0], max(prices),
                                  min(prices), prices[-1]))
        return candles

    # -- tick production -----------------------------------------------------
    def _price_at(self, symbol: str, at: float) -> float:
        state = self._state.get(symbol)
        if state is None:
            state = self._state[symbol] = _SymbolState(symbol, 1.0, 0.0)
        start = self._start if self._start is not None else at
        phase = TWO_PI * ((at - start) / self._cycle_seconds + state.offset)
        cycle = state.base * self._amplitude_bps / 10000.0 * math.sin(phase)
        state.walk = state.walk * 0.995 + self._rng.gauss(
            0.0, state.base * self._noise_bps / 10000.0)
        return max(1e-6, state.base + cycle + state.walk)

    def tick_once(self, at: Optional[float] = None) -> None:
        """Emit one tick per symbol at time ``at`` (defaults to now)."""
        now = self._clock.now() if at is None else at
        self._last_tick_at = now
        handler = self._handler
        if handler is None:
            return
        for symbol in self._symbols:
            handler(symbol, now, self._price_at(symbol, now))

    async def run(self, stop: Optional[asyncio.Event] = None) -> None:
        """Emit ticks at ``ticks_per_second`` until stopped.

        If the clock jumps forward by more than one tick interval (a virtual
        clock stepping in coarse jumps), missed ticks are skipped rather than
        replayed: the price is a function of time, so bars still come out right.
        """
        if self._start is None:
            self._start = self._clock.now()
        next_at = self._clock.now() + self._tick_interval
        while self._running and (stop is None or not stop.is_set()):
            await self._clock.sleep_until(next_at)
            now = self._clock.now()
            self.tick_once(now)
            next_at = max(next_at + self._tick_interval, now + self._tick_interval)
