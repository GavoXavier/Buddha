"""Replay the bot's own candle store through the live signal path.

    python backtest.py                 # every market in the store
    python backtest.py EURUSD_otc      # one market
    python backtest.py --hours 6       # only the most recent 6 hours

**What this is for.** Not profit forecasting. It answers the two questions that
cannot be answered by reading the code: *how often does this configuration
actually fire*, and *which gate is throwing the rest away*. That is what the
selectivity dials in ``.env`` (MIN_SCORE, MIN_COMPONENTS, TREND_FLAT_MIN_SCORE,
MIN_CONFIDENCE) trade against each other, and guessing at them is how the bot
ends up either silent or trading noise.

**What it reuses.** The engine (`signals.engine.evaluate`), the ranking
(`signals.ranking.select_best`) and the outcome rule (`engine.scheduler
.outcome_of`) are the *same functions the live loop calls*, so this cannot drift
away from production behaviour.

**What it deliberately does not use: broker history.** Pocket Option's history
endpoint and its live tick stream are unrelated price paths for OTC assets (see
the README for the measurements), so replaying history tests the engine against
a market that was never traded. The only honest input is the candle store the
bot built from ticks it actually received — one JSON file per market under
CANDLE_STORE_DIR.

**Known optimism.** A live signal is judged 10s before the bar closes, on a bar
that still has 10 seconds of ticks to come; here the bar is complete. So the
backtest sees slightly more information than the bot does, and its win rate is a
ceiling rather than a prediction. Signals also carry no payout here (the store
does not record one), so ranking ties fall through to the symbol name.

**Sample size.** The store only holds what the bot has watched — an hour of
uptime is an hour of sample. Treat anything under a few days as a rumour.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import Config, ConfigError, load_config
from engine.scheduler import outcome_of
from market.store import CandleStore
from signals.engine import Candle, _analyze, evaluate, vote_components
from signals.ranking import Candidate, select_best
from telegram.sender import configure_clock

# Why a judgement produced nothing, in the order the engine checks.
_THIN = "not enough indicators warm"
_TIE = "indicators disagree (tie)"
_WEAK = "agreement below MIN_SCORE"
_GATED = "rejected by a gate (trend / MTF / ATR / S-R)"


class Store:
    """The persisted candle store, indexed for replay."""

    def __init__(self, directory: str | Path, cfg: Config, hours: float = 0.0):
        self.cfg = cfg
        # Match the feed this config would connect to, so replaying a simulated
        # store is not silently presented as a measurement of the live market.
        self.store = CandleStore(directory, cfg.candle_period, cfg.max_bars,
                                 feed=cfg.feed)
        self.bars: dict[str, list[Candle]] = {}
        self.times: dict[str, list[float]] = {}
        for symbol in self._symbols(directory):
            candles = self.store.load(symbol)
            if hours > 0 and candles:
                cutoff = candles[-1].time - hours * 3600
                candles = [c for c in candles if c.time >= cutoff]
            if len(candles) >= 2:
                self.bars[symbol] = candles
                self.times[symbol] = [c.time for c in candles]

    @staticmethod
    def _symbols(directory: str | Path) -> list[str]:
        path = Path(directory)
        if not path.is_dir():
            return []
        found = []
        for file in sorted(path.glob("*.json")):
            try:
                with open(file, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("candles"), list):
                found.append(doc.get("asset") or file.stem)
        return found

    def boundaries(self) -> list[int]:
        """Every bar boundary any market has data up to (the loop's minutes)."""
        period = self.cfg.candle_period
        marks: set[int] = set()
        for candles in self.bars.values():
            marks.update(int(c.time) + period for c in candles)
        return sorted(marks)

    def close_at(self, symbol: str, time: float) -> float | None:
        candles, times = self.bars[symbol], self.times[symbol]
        i = bisect_right(times, time) - 1
        if i >= 0 and abs(times[i] - time) < 1e-6:
            return candles[i].close
        return None

    def buffer_for(self, symbol: str, boundary: int) -> list[Candle]:
        """Bars the engine would judge for an entry at ``boundary``.

        Reproduces what the live loop can see at the signal, which depends on the
        lead. With a short lead the live aggregator feeds the bar closing on the
        boundary as a *snapshot* of the bar still in progress; replaying its
        finished close instead is a small, accepted optimism — ten seconds of it.

        With a full-bar lead the bar closing at ``boundary`` has not even started
        when the signal goes out, so reading it back is not optimism, it is one
        whole bar of hindsight: the replay would judge on the very bar the trade
        opens on and report an edge the live loop cannot have. There the window
        stops one bar earlier, matching the live ``series.closed()``.
        """
        candles, times = self.bars[symbol], self.times[symbol]
        period = self.cfg.candle_period
        bars_back = period if self.cfg.lead_seconds < period else 2 * period
        i = bisect_right(times, boundary - bars_back)
        if i < 2:
            return []
        return candles[max(0, i - self.cfg.max_bars):i]


class Replay:
    """One pass of the whole universe, exactly one signal per boundary at most."""

    def __init__(self, store: Store):
        self.store = store
        self.cfg = store.cfg
        self.last_signal_at: dict[str, float] = {}
        self.trades: list[dict] = []
        self.reasons: dict[str, int] = {}
        self.judgements = 0
        self.candidates = 0
        self.boundaries = 0
        self.scores: dict[int, int] = {}
        self.trends: dict[str, int] = {}

    def run(self) -> "Replay":
        cfg = self.cfg
        for boundary in self.store.boundaries():
            candidates = self._collect(boundary)
            self.boundaries += 1
            self.candidates += len(candidates)
            chosen = select_best(candidates, self.last_signal_at,
                                 boundary - cfg.lead_seconds,
                                 cfg.cooldown_seconds, cfg.signal.min_confidence)
            if chosen is None:
                continue
            self.last_signal_at[chosen.asset] = boundary - cfg.lead_seconds
            self._settle(chosen, boundary)
        return self

    def _collect(self, boundary: int) -> list[Candidate]:
        out: list[Candidate] = []
        for symbol in self.store.bars:
            buffer = self.store.buffer_for(symbol, boundary)
            if len(buffer) < 2:
                continue
            self.judgements += 1
            signal = evaluate(buffer, self.cfg.signal)
            if signal is not None:
                self.scores[signal.score] = self.scores.get(signal.score, 0) + 1
                out.append(Candidate(asset=symbol, signal=signal, bars=len(buffer)))
                continue
            self._blame(buffer)
        return out

    def _blame(self, buffer: list[Candle]) -> None:
        """Classify a rejection without re-implementing the gates.

        The vote tally comes from the engine's own analysis, so this cannot
        disagree with ``evaluate`` about *why* there was no direction; only the
        final bucket is a catch-all for the gates that ``evaluate`` applies.
        """
        a = _analyze(buffer, self.cfg.signal)
        self.trends[a.trend or "n/a"] = self.trends.get(a.trend or "n/a", 0) + 1
        if len(vote_components(a)) < self.cfg.signal.min_components:
            reason = _THIN
        elif a.bull == a.bear:
            reason = _TIE
        elif max(a.bull, a.bear) < self.cfg.signal.min_score:
            reason = _WEAK
        else:
            reason = _GATED
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def _settle(self, chosen: Candidate, boundary: int) -> None:
        cfg = self.cfg
        period = cfg.candle_period
        entry_price = self.store.close_at(chosen.asset, boundary - period)
        exit_price = self.store.close_at(
            chosen.asset, boundary - period + cfg.expiry_seconds)
        if entry_price is None or exit_price is None:
            return  # a gap in the store: the live loop would wait for the tick
        self.trades.append({
            "asset": chosen.asset,
            "direction": chosen.signal.direction,
            "entry_at": boundary,
            "entry": entry_price,
            "exit": exit_price,
            "score": chosen.signal.score,
            "confidence": chosen.signal.confidence,
            "outcome": outcome_of(chosen.signal.direction, entry_price, exit_price),
        })

    # -- reporting -----------------------------------------------------------
    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t["outcome"] == "WIN")

    @property
    def losses(self) -> int:
        return len(self.trades) - self.wins

    def hours(self) -> float:
        marks = [t for b in self.store.bars.values() for t in (b[0].time, b[-1].time)]
        if len(marks) < 2:
            return 0.0
        return max(0.0, (max(marks) - min(marks)) / 3600.0)

    def report(self) -> None:
        cfg = self.cfg
        print(f"Period {cfg.candle_period}s · expiry {cfg.expiry} · "
              f"cooldown {cfg.cooldown_seconds}s · lead {cfg.lead_seconds}s")
        print(f"Selectivity: MIN_SCORE={cfg.signal.min_score} "
              f"MIN_COMPONENTS={cfg.signal.min_components} "
              f"TREND_FLAT_MIN_SCORE={cfg.signal.trend_flat_min_score} "
              f"MIN_CONFIDENCE={cfg.signal.min_confidence}")
        print()

        if not self.store.bars:
            print("No usable candles in the store yet.")
            print("Run the bot (FEED=simulated python main.py) to start collecting —")
            print("bars are built from live ticks, never from broker history.")
            return

        hours = self.hours()
        print(f"{'market':<16}{'bars':>7}  {'from':<12}{'to':<12}{'span':>6}")
        print("-" * 55)
        for symbol, candles in sorted(self.store.bars.items()):
            span = (candles[-1].time - candles[0].time) / 3600.0
            print(f"{symbol:<16}{len(candles):>7}  {_clock(candles[0].time):<12}"
                  f"{_clock(candles[-1].time):<12}{span:>5.1f}h")
        total_bars = sum(len(c) for c in self.store.bars.values())
        print("-" * 55)
        print(f"{'':<16}{total_bars:>7}  over {hours:.1f}h across "
              f"{len(self.store.bars)} markets")
        print()

        per_asset: dict[str, dict[str, int]] = {}
        for trade in self.trades:
            row = per_asset.setdefault(trade["asset"], {"W": 0, "L": 0})
            row["W" if trade["outcome"] == "WIN" else "L"] += 1
        signals = len(self.trades)
        if per_asset:
            print(f"{'market':<16}{'signals':>9}{'WIN':>6}{'LOSS':>6}{'rate':>7}")
            print("-" * 44)
            for asset, row in sorted(per_asset.items(),
                                     key=lambda r: -(r[1]["W"] + r[1]["L"])):
                decided = row["W"] + row["L"]
                rate = row["W"] / decided if decided else 0.0
                print(f"{asset:<16}{decided:>9}{row['W']:>6}{row['L']:>6}{rate:>7.0%}")
            print("-" * 44)

        rate = self.wins / signals if signals else 0.0
        print(f"Signals: {signals}  in {hours:.1f}h"
              + (f"  =  {signals / hours:.1f}/hour" if hours else ""))
        print(f"Outcomes: {self.wins}W / {self.losses}L  =  {rate:.0%}"
              + ("" if signals else "  (nothing fired — the dials are too tight)"))
        if self.boundaries:
            fired = signals / self.boundaries
            print(f"Minutes judged: {self.boundaries}; a signal on {fired:.1%} of them")
        print()

        print(f"Why the rest produced nothing ({self.judgements} judgements, "
              f"{self.candidates} candidate setups):")
        if self.reasons:
            total = sum(self.reasons.values())
            for reason, count in sorted(self.reasons.items(), key=lambda r: -r[1]):
                print(f"  {reason:<34}{count:>7}  {count / total:>6.1%}")
        else:
            print("  (nothing was rejected)")
        if self.scores:
            print("Score distribution of accepted setups: "
                  + ", ".join(f"{k}:{v}" for k, v in sorted(self.scores.items())))
        if self.trends:
            total = sum(self.trends.values())
            print("Market state at judgement: "
                  + ", ".join(f"{k} {v / total:.0%}"
                              for k, v in sorted(self.trends.items(), key=lambda r: -r[1])))
        print()
        print("Caveats: the replay judges *finished* bars (live signals fire 10s")
        print("early); the sample is only as long as the bot has been running; and")
        print(f"the store only keeps the last MAX_BARS={cfg.max_bars} bars per market.")


def _clock(at: float) -> str:
    import datetime
    from telegram.sender import MARKET_TZ
    return datetime.datetime.fromtimestamp(at, tz=MARKET_TZ).strftime("%m-%d %H:%M")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the bot's tick-built candle store through the engine.")
    parser.add_argument("symbols", nargs="*",
                        help="markets to replay (default: everything in the store)")
    parser.add_argument("--hours", type=float, default=0.0,
                        help="only use the most recent N hours of stored bars")
    parser.add_argument("--dir", default=None,
                        help="candle store directory (default: CANDLE_STORE_DIR)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    # The bar times listed below are read next to the times in the signals, so
    # they have to be in the same zone — and it is a setting, not a constant.
    configure_clock(cfg.market_tz_offset_hours)

    directory = args.dir or cfg.candle_store_dir
    store = Store(directory, cfg, hours=args.hours)
    if args.symbols:
        wanted = set(args.symbols)
        store.bars = {s: c for s, c in store.bars.items() if s in wanted}
        store.times = {s: t for s, t in store.times.items() if s in wanted}
        missing = wanted - set(store.bars)
        if missing:
            print(f"not in the store (or too few bars): {', '.join(sorted(missing))}\n",
                  file=sys.stderr)

    print(f"Candle store: {directory}\n")
    Replay(store).run().report()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
