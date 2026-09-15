"""Persistence for the live-aggregated candle series.

Because broker history cannot be trusted as the traded price path (see
``aggregator`` and the README), the only truthful warm-up data is what this bot
has already watched go by. Persisting it means a restart resumes with a warm
buffer in seconds instead of spending another hour blind.

Files are one JSON document per symbol, written atomically (temp file +
``os.replace``) so a crash mid-write cannot corrupt the store. A document
records its schema version, bar period **and the feed it was built from**;
anything that does not match the current configuration is discarded rather than
mixed into the buffer.

Writes are *additive*: a save unions the series it is given with the one already
persisted, so a bar that has been watched is never lost to a later, shorter
series. See ``save`` for why that is not merely tidy.

That last one matters more than it looks. The period check stops 1-minute bars
being read as 5-minute ones; the feed check stops *fabricated* bars being read as
real ones. ``FEED=simulated`` writes to the same directory with the same symbols,
so switching to ``FEED=pocket_option`` without it would quietly seed the live
indicators with a synthetic price path — and the resulting signals would look
exactly like real ones. A store whose origin cannot be established (no ``feed``
field, i.e. written before this check existed) is discarded too: "unknown" is not
"trusted".
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from signals.engine import Candle

log = logging.getLogger("pocket.store")

_SCHEMA_VERSION = 1


class CandleStore:
    def __init__(self, directory: str | Path, period: int, max_bars: int,
                 feed: str = "") -> None:
        self.directory = Path(directory)
        self.period = period
        self.max_bars = max_bars
        # Which feed produced these bars. Empty means "not stated", which is
        # only ever a match against another empty — a store with a real feed
        # stamped on it will not be read by a caller that did not name one.
        self.feed = str(feed or "")
        self._last_save: dict[str, float] = {}
        self.save_interval = 30.0  # seconds between writes per symbol

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe}.json"

    def load(self, symbol: str) -> list[Candle]:
        """Return the persisted bars for ``symbol`` (empty if missing/unusable)."""
        path = self._path(symbol)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable candle store %s: %s", path.name, exc)
            return []

        if doc.get("version") != _SCHEMA_VERSION:
            log.warning("candle store %s has version %s (want %s) - discarding",
                        path.name, doc.get("version"), _SCHEMA_VERSION)
            return []
        if int(doc.get("period", 0)) != self.period:
            log.warning("candle store %s is for period %ss, not %ss - discarding",
                        path.name, doc.get("period"), self.period)
            return []
        stored_feed = str(doc.get("feed", ""))
        if stored_feed != self.feed:
            # Simulated bars must never become a live engine's history: they are
            # a fabricated price path, and signals read off them are worthless
            # while looking exactly like real ones. An unstated origin is not a
            # match either — it cannot be shown to be the right feed.
            log.warning("candle store %s came from the %s feed, not %s - discarding",
                        path.name, stored_feed or "unknown", self.feed or "unknown")
            return []

        candles: list[Candle] = []
        for row in doc.get("candles", []):
            try:
                t, o, h, l, c = (float(x) for x in row)
            except (TypeError, ValueError):
                continue
            candles.append(Candle(t, o, h, l, c))
        candles.sort(key=lambda x: x.time)
        return candles[-self.max_bars:]

    def load_all(self, symbols: Iterable[str]) -> dict[str, list[Candle]]:
        out: dict[str, list[Candle]] = {}
        for symbol in symbols:
            candles = self.load(symbol)
            if candles:
                out[symbol] = candles
        return out

    def _merge_with_disk(self, symbol: str, candles: Sequence[Candle]) -> list[Candle]:
        """``candles`` unioned with the persisted series, by timestamp.

        ``load`` is what filters by version, period and feed, so a store written
        by a different configuration contributes nothing here and the caller's
        series is written on its own — the same discarding rule as everywhere
        else, applied rather than duplicated.

        On a timestamp present in both, the caller's bar wins: it is the one
        built from ticks this process received, and a closed bar never changes
        anyway, so the choice only matters if a bar was somehow written twice.
        """
        by_time: dict[float, Candle] = {c.time: c for c in self.load(symbol)}
        for candle in candles:
            by_time[candle.time] = candle
        return sorted(by_time.values(), key=lambda c: c.time)

    def save(self, symbol: str, candles: Sequence[Candle], force: bool = False) -> None:
        """Write ``candles`` for ``symbol``, at most once per ``save_interval``.

        What is written is the series handed in *unioned with whatever is already
        on disk*, not that series alone. A closed bar is immutable, so merging by
        timestamp can only ever add bars.

        It has to, because the live series is not monotonic in what it holds. A
        gap longer than ``MAX_GAP_BARS`` makes the aggregator drop everything
        before it — correctly, since an indicator window must not be left to span
        an outage — and a plain overwrite would then write that truncation
        through to disk. That turns a temporary blindness into permanent loss:
        measured on 2026-09-15, one teardown flush after a churn-induced gap took
        AUDUSD from 19 bars to 2, EURCHF 20 to 2 and GBPUSD 21 to 2, and no
        later session could recover them. Refusing to read across a hole is a
        judgement about *analysis*; deleting the bars is a judgement about
        *history*, and the two do not have to be the same one.
        """
        now = time.time()
        if not force and now - self._last_save.get(symbol, 0.0) < self.save_interval:
            return
        self._last_save[symbol] = now
        merged = self._merge_with_disk(symbol, candles)[-self.max_bars:]
        doc = {
            "version": _SCHEMA_VERSION,
            "period": self.period,
            "feed": self.feed,
            "asset": symbol,
            "saved_at": now,
            "candles": [[c.time, c.open, c.high, c.low, c.close]
                        for c in merged],
        }
        path = self._path(symbol)
        tmp = path.with_suffix(".json.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not persist candles for %s: %s", symbol, exc)

    def save_all(self, series_by_symbol: dict, force: bool = False) -> None:
        """Persist every series given as ``{symbol: CandleSeries}``."""
        for symbol, series in series_by_symbol.items():
            self.save(symbol, series.closed(), force=force)

    def newest_saved_at(self, symbols: Iterable[str]) -> Optional[float]:
        """Timestamp of the most recent save across ``symbols`` (for logging)."""
        newest: Optional[float] = None
        for symbol in symbols:
            path = self._path(symbol)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        return newest
