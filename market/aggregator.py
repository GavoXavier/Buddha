"""Live tick -> OHLC candle aggregation, aligned to wall-clock boundaries.

Every bar is built from ticks this process actually received. Broker *history*
is deliberately not used: for Pocket Option's OTC assets the history endpoint
and the live tick stream are unrelated price paths (measured drift of 6-18 pips
per 5 minutes on the same minutes, versus a constant offset on real assets), so
warming indicators from history produces signals about a series that never
traded. See the README for the measurements.

Buckets are ``floor(timestamp / period) * period`` — the same convention the
official SDK uses — and are labelled by their OPEN time. A bar is closed when
either the first tick of the next bucket arrives, or ``finalize()`` sees that
wall-clock time has passed its end; the second path is what makes the bot keep
producing bars when ticks are briefly late.

No synthetic candles are ever invented: a bucket that received no ticks simply
does not become a bar, and the asset is instead reported as stale so it can be
skipped.

A hole longer than ``MAX_GAP_BARS`` bars is a different matter, and the series
is *dropped* across it rather than left to span it. The engine's indicators
assume evenly spaced bars, so a window straddling a long hole reads the whole
outage as a single bar's move: after a 92-minute feed outage in which EURUSD
moved 48 pips, RSI and Stochastic were pinned to the same extreme and voted
together, producing a signal every minute off an outage instead of a market.
Only the bars *before* the hole go — the ones after it are still good — so this
costs warm-up time, not data.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Iterator, Optional, Sequence

from signals.engine import Candle

from .clock import Clock, RealClock

log = logging.getLogger("pocket.market")

# How long a hole in the feed has to be before the bars either side of it are
# treated as two different series. A stray missing bar is tolerated: a
# two-minute move read as a one-minute one is noise, not a broken series.
GAP_TOLERANCE_BARS = 5


class CandleSeries:
    """Rolling tick-built candle series for one symbol."""

    def __init__(self, symbol: str, period: int, max_bars: int,
                 clock: Optional[Clock] = None,
                 max_gap_bars: int = GAP_TOLERANCE_BARS) -> None:
        self.symbol = symbol
        self.period = period
        self.max_bars = max_bars
        self.max_gap_bars = max_gap_bars
        self._clock = clock or RealClock()

        self._closed: deque[Candle] = deque(maxlen=max_bars)
        self._cur_bucket: Optional[int] = None
        self._cur_finalized = False
        self._o = self._h = self._l = self._c = 0.0
        self._ticks = 0

        self.last_tick_ts: float = 0.0
        self.last_price: float = 0.0
        self.gaps: int = 0                # buckets with no ticks at all
        self.last_gap_seconds: int = 0

    # -- ingestion -----------------------------------------------------------
    def add_tick(self, ts: float, price: float) -> None:
        """Fold one price update in. Ignores out-of-range or non-finite ticks."""
        if not math.isfinite(price) or price <= 0 or not math.isfinite(ts):
            return
        now = self._clock.now()
        if ts > now + self.period:
            return  # implausible timestamp (clock skew) - don't invent a bucket
        if ts > self.last_tick_ts:
            self.last_tick_ts = ts
            self.last_price = price

        bucket = int(math.floor(ts / self.period) * self.period)

        if self._cur_bucket is None:
            self._open_bucket(bucket, price)
        elif bucket == self._cur_bucket:
            if self._cur_finalized:
                return  # late tick for a bar already closed; too late to matter
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = price
            self._ticks += 1
        elif bucket > self._cur_bucket:
            self._close_bucket()
            missing = (bucket - self._cur_bucket) // self.period - 1
            if missing > 0:
                self.gaps += 1
                self.last_gap_seconds = bucket - self._cur_bucket
                if missing > self.max_gap_bars:
                    self._drop_broken_history(missing)
            self._open_bucket(bucket, price)
        # bucket < current: a genuinely late tick, already accounted for.

    def _drop_broken_history(self, missing: int) -> None:
        """Forget every bar before a hole too long to be one series with them.

        Called when the feed comes back after a gap. The bars after the hole are
        sound and are kept; the ones before it can never be contiguous with
        them, so an indicator window spanning the two would be reading an outage.
        """
        dropped = len(self._closed)
        self._closed.clear()
        log.warning("%s: %d-minute hole in the feed — dropped %d bar(s) that "
                    "cannot be one series with what follows, warming up again",
                    self.symbol, missing, dropped)

    def _open_bucket(self, bucket: int, price: float) -> None:
        self._cur_bucket = bucket
        self._cur_finalized = False
        self._o = self._h = self._l = self._c = price
        self._ticks = 1

    def _close_bucket(self) -> None:
        if self._cur_bucket is None or self._cur_finalized:
            return
        if self._ticks > 0:
            self._closed.append(Candle(
                time=float(self._cur_bucket),
                open=self._o, high=self._h, low=self._l, close=self._c,
            ))
        self._cur_finalized = True

    def finalize(self, now: Optional[float] = None) -> None:
        """Close the in-progress bar once wall-clock time has passed its end."""
        now = self._clock.now() if now is None else now
        if (self._cur_bucket is not None and not self._cur_finalized
                and self._cur_bucket + self.period <= now):
            self._close_bucket()

    # -- reads ---------------------------------------------------------------
    @property
    def bar_count(self) -> int:
        return len(self._closed)

    def closed(self) -> list[Candle]:
        return list(self._closed)

    def forming(self) -> Optional[Candle]:
        """Snapshot of the in-progress bar, or None if there isn't one."""
        if self._cur_bucket is None or self._cur_finalized:
            return None
        return Candle(time=float(self._cur_bucket), open=self._o,
                      high=self._h, low=self._l, close=self._c)

    def ticks_in_forming(self) -> int:
        return 0 if self._cur_finalized else self._ticks

    def buffer_for_boundary(self, boundary: float) -> list[Candle]:
        """Bars to judge for a trade entered at ``boundary``.

        That is every closed bar up to the one *before* the bar ending at
        ``boundary``, plus the bar ending at ``boundary`` itself — as a
        snapshot of the bar still in progress, which is exactly the state the
        engine needs a few seconds before it closes.
        """
        last_open = boundary - self.period
        out = [c for c in self._closed if c.time <= last_open]
        if out and abs(out[-1].time - last_open) < 1e-6:
            return out  # already finalized (late wake) - use the real bar
        if self._cur_bucket is not None and abs(self._cur_bucket - last_open) < 1e-6:
            snap = self.forming()
            if snap is not None:
                out.append(snap)
        return out

    def price_at_boundary(self, boundary: float) -> Optional[float]:
        """Close of the bar that ended at ``boundary`` — the entry/exit price."""
        want = boundary - self.period
        for c in reversed(self._closed):
            if abs(c.time - want) < 1e-6:
                return c.close
            if c.time < want:
                break
        return None

    def tick_age(self, now: Optional[float] = None) -> float:
        """Seconds since the last tick (``inf`` if none seen yet)."""
        if self.last_tick_ts <= 0:
            return float("inf")
        now = self._clock.now() if now is None else now
        return now - self.last_tick_ts

    def is_stale(self, stale_after: float, now: Optional[float] = None) -> bool:
        return self.tick_age(now) > stale_after

    # -- restore -------------------------------------------------------------
    def restore(self, candles: Sequence[Candle]) -> None:
        """Seed the buffer from a persisted series (restart warm-up).

        Only closed bars are ever persisted, so no bucket is in progress: the
        first live tick opens a fresh one. That first bar is partial (it starts
        mid-bucket, so its open/high/low miss the ticks we were not running
        for) — the alternative would be inventing them.

        A persisted series can hold a hole, because the process was not running
        or the feed died mid-session, so only the newest run of contiguous bars
        is restored. Without that, the first indicator windows after a restart
        straddle the hole and read the whole outage as one bar's move.
        """
        ordered = sorted(candles, key=lambda x: x.time)[-self.max_bars:]
        start = self._newest_contiguous_start(ordered)
        if start:
            log.warning("%s: persisted bars span a %d-minute hole — restored "
                        "only the %d bar(s) after it",
                        self.symbol,
                        int((ordered[start].time - ordered[start - 1].time)
                            // self.period) - 1,
                        len(ordered) - start)
        for c in ordered[start:]:
            self._closed.append(c)

    def _newest_contiguous_start(self, candles: Sequence[Candle]) -> int:
        """Index of the first bar in the newest unbroken run, 0 if none is broken."""
        start = 0
        for i in range(1, len(candles)):
            gap = (candles[i].time - candles[i - 1].time) / self.period
            if int(round(gap)) - 1 > self.max_gap_bars:
                start = i
        return start


class MarketState:
    """All tracked symbols' candle series, fed by one tick handler."""

    def __init__(self, period: int, max_bars: int, stale_after: float,
                 clock: Optional[Clock] = None,
                 max_gap_bars: int = GAP_TOLERANCE_BARS) -> None:
        self.period = period
        self.max_bars = max_bars
        self.stale_after = stale_after
        self.max_gap_bars = max_gap_bars
        self._clock = clock or RealClock()
        self._series: dict[str, CandleSeries] = {}

    def track(self, symbol: str) -> CandleSeries:
        series = self._series.get(symbol)
        if series is None:
            series = CandleSeries(symbol, self.period, self.max_bars, self._clock,
                                  self.max_gap_bars)
            self._series[symbol] = series
        return series

    def symbols(self) -> list[str]:
        return list(self._series)

    def on_tick(self, symbol: str, ts: float, price: float) -> None:
        self.track(symbol).add_tick(ts, price)

    def finalize(self, now: Optional[float] = None) -> None:
        for series in self._series.values():
            series.finalize(now)

    def healthy(self, now: Optional[float] = None) -> list[str]:
        """Symbols with a live feed and at least one real bar."""
        return [
            s for s, series in self._series.items()
            if series.bar_count > 0 and not series.is_stale(self.stale_after, now)
        ]

    def stalest(self) -> list[tuple[str, float]]:
        return sorted(
            ((s, series.tick_age()) for s, series in self._series.items()),
            key=lambda x: -x[1],
        )

    def iter_series(self) -> Iterator[tuple[str, CandleSeries]]:
        return iter(self._series.items())
