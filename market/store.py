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

That last one matters more than it looks, and it is applied to writing as well as
reading. The period check stops 1-minute bars being read as 5-minute ones; the
feed check stops *fabricated* bars being read as real ones. ``FEED=simulated``
writes to the same directory with the same symbols, so switching to
``FEED=pocket_option`` without it would quietly seed the live indicators with a
synthetic price path — and the resulting signals would look exactly like real
ones. A store whose origin cannot be established (no ``feed`` field, i.e. written
before this check existed) is discarded too: "unknown" is not "trusted".

The same rule governs ``save``: a file this store would not read from is a file it
will not write over. Reading a fabricated series costs a bad signal; overwriting a
real one costs the history, which is the only input here the broker cannot
re-supply — so a refused write is much the cheaper failure. See ``_foreign_feed``.
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

    def _foreign_feed(self, symbol: str) -> Optional[str]:
        """The feed stamped on the persisted file when it is not ours, else ``None``.

        A file that is missing, unreadable or not a store document has no history
        to protect, so it is not foreign — the caller may write over it. A file
        carrying another feed's name is, and the one rule is the one ``load``
        already applies: **a file this store would not read from it will not write
        over either.**
        """
        path = self._path(symbol)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict):
            return None
        stored = str(doc.get("feed", ""))
        return stored if stored != self.feed else None

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
        foreign = self._foreign_feed(symbol)
        if foreign is not None:
            # The check in ``load`` is why this bot will not *read* fabricated
            # bars as history. This one is why it will not destroy real history
            # with them: without it, running a simulated session in the live
            # store's directory replaces every file with a fabricated series —
            # measured on 2026-09-16, a 50-bar live store became a 3-bar simulated
            # one and none of the 50 could be read back. Nothing else in the bot
            # can undo that, and watched bars are the one input the feed cannot
            # re-supply, so refusing the write is the cheap side of this trade.
            #
            # Logged at most once per ``save_interval`` per symbol, because the
            # gate above has just stamped the clock.
            log.error(
                "refusing to write %s bars over %s bars in %s — one feed per "
                "store directory. Give the %s run its own CANDLE_STORE_DIR, or "
                "delete the file to start this feed's series from scratch",
                self.feed or "unnamed", foreign or "unnamed",
                self._path(symbol).name, self.feed or "unnamed")
            return
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


class CandleTiers:
    """The store the engine reads, plus a deeper copy nothing reads but a replay.

    ``MAX_BARS`` answers two questions at once, and they are not the same one: how
    much history the live buffer may hold in memory, and how much history a
    backtest can ever be run against. The first has to stay small — 20,000 bars
    across twenty markets is a lot of objects to hold for indicators that read
    dozens — while the second is the whole sample a selectivity dial can be
    checked against, and it is what runs out first. Measured on 2026-09-16 the
    shipped engine fires 0.2 times an hour, so the 30 held-out trades
    ``calibrate.py`` wants need roughly 150h of history, three and a half times
    what a 500-bar store can hold. Raising ``MAX_BARS`` to reach it would make the
    live buffer exactly as heavy as the archive.

    So the bars go to two places. The live tier is written and read exactly as
    before, and is still the only one the engine ever sees; the archive tier takes
    the same series with a much larger cap and a much slower write interval, and
    is read only by ``backtest.py --dir``. A signal cannot depend on it, because
    ``load`` here is the live store's ``load`` and nothing else — which is the
    property worth keeping, so it is structural rather than a matter of care.

    The archive's own writes are the same union-with-disk as the live store's, so
    the bars survive the live store's truncation into it: a session that drops a
    market, or a series that a gap truncates, still leaves the archive holding
    everything that was ever watched.
    """

    def __init__(self, live: CandleStore, archive: Optional[CandleStore] = None,
                 archive_interval: float = 900.0) -> None:
        self.live = live
        self.archive = archive
        if archive is not None:
            # The archive is a copy, not a buffer waiting to be resumed: it wants
            # to be written rarely (the union makes the writes additive, so
            # nothing is lost between them) and read never.
            archive.save_interval = archive_interval
        self.directory = live.directory
        self.period = live.period
        self.max_bars = live.max_bars

    @property
    def store(self) -> CandleStore:
        """The tier the engine reads — kept for callers that name ``store``."""
        return self.live

    def load(self, symbol: str) -> list[Candle]:
        return self.live.load(symbol)

    def load_all(self, symbols: Iterable[str]) -> dict[str, list[Candle]]:
        return self.live.load_all(symbols)

    def seed(self, symbol: str, candles: Sequence[Candle]) -> bool:
        """Give an empty archive the bars the live store already holds.

        Without this the archive only ever learns bars from the moments it is
        running, so a bot upgraded tonight would have an archive holding one
        night's bars while the live store beside it held a day of them — the
        deeper copy starting out shallower than the original, which is the one
        thing it must not be. A symbol the archive already knows is left alone:
        its own series is the union of everything ever watched, which is more
        than the live store can show.
        """
        if self.archive is None or len(candles) < 2:
            return False
        if self.archive.load(symbol):
            return False
        self.archive.save(symbol, candles, force=True)
        return True

    def save(self, symbol: str, candles: Sequence[Candle], force: bool = False) -> None:
        self.live.save(symbol, candles, force=force)
        if self.archive is not None:
            try:
                self.archive.save(symbol, candles, force=force)
            except Exception as exc:      # never let a copy end a session
                log.warning("could not archive candles for %s: %s", symbol, exc)

    def save_all(self, series_by_symbol: dict, force: bool = False) -> None:
        for symbol, series in series_by_symbol.items():
            self.save(symbol, series.closed(), force=force)

    def archived(self, symbols: Iterable[str]) -> dict[str, list[Candle]]:
        """What the archive holds, for a replay that asks for it directly."""
        if self.archive is None:
            return {}
        return self.archive.load_all(symbols)
