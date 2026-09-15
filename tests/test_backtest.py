"""Tests for the candle-store replay used to tune the selectivity dials.

The backtester is a diagnostic, but its plumbing has to be right or it reports
tuning advice from a broken simulation: entries must land on boundaries, the two
prices must be the two bar closes the live loop would have used, and a gap in the
store must produce no trade rather than a made-up one.
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import backtest
from config import Config
from market.store import CandleStore
from signals.engine import Candle, Signal, SignalConfig

BASE = 1_700_000_040.0
PERIOD = 60


def make_config(**overrides) -> Config:
    values = dict(
        feed="simulated", candle_period=PERIOD, max_bars=1000,
        expiry="1m", expiry_seconds=60, lead_seconds=10, cooldown_seconds=120,
        signal=SignalConfig(bar_seconds=PERIOD, min_score=2, min_components=2))
    values.update(overrides)
    return Config(**values)


def bar(open_at, close, spread=0.001):
    return Candle(float(open_at), close, close + spread, close - spread, close)


class BacktestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_store(self, series: dict[str, list[Candle]], period: int = PERIOD,
                    feed: str = "simulated") -> None:
        # Matches make_config()'s feed: the replay only reads a store that came
        # from the feed it would connect to.
        store = CandleStore(self.dir, period, 1000, feed=feed)
        for symbol, candles in series.items():
            store.save(symbol, candles, force=True)

    def replay(self, **overrides) -> backtest.Replay:
        cfg = make_config(**overrides)
        return backtest.Replay(backtest.Store(self.dir, cfg)).run()

    def straight(self, symbol="AAA_otc", count=10, start=BASE,
                 period: int = PERIOD):
        return {symbol: [bar(start + i * period, 1.0 + i * 0.001)
                         for i in range(count)]}


class TestStoreIndex(BacktestCase):
    def test_reads_symbols_and_boundaries_from_the_store(self):
        self.write_store({**self.straight("AAA_otc", 5), **self.straight("BBB_otc", 3)})
        store = backtest.Store(self.dir, make_config())

        self.assertEqual(sorted(store.bars), ["AAA_otc", "BBB_otc"])
        # Each bar's boundary is its open time plus one period.
        self.assertEqual(store.boundaries()[0], BASE + PERIOD)
        self.assertEqual(store.boundaries()[-1], BASE + 5 * PERIOD)

    def test_ignores_files_that_are_not_candle_series(self):
        self.write_store(self.straight("AAA_otc", 3))
        (self.dir / "stats.json").write_text('{"version": 1, "global": {}}',
                                             encoding="utf-8")
        self.assertEqual(list(backtest.Store(self.dir, make_config()).bars),
                         ["AAA_otc"])

    def test_missing_directory_is_not_an_error(self):
        store = backtest.Store(self.dir / "nope", make_config())
        self.assertEqual(store.bars, {})
        self.assertEqual(store.boundaries(), [])

    def test_a_candle_lookup_returns_the_close_of_that_bar_only(self):
        self.write_store(self.straight("AAA_otc", 5))
        store = backtest.Store(self.dir, make_config())
        self.assertAlmostEqual(store.close_at("AAA_otc", BASE), 1.000)
        self.assertAlmostEqual(store.close_at("AAA_otc", BASE + 2 * PERIOD), 1.002)
        self.assertIsNone(store.close_at("AAA_otc", BASE + 90), "not a real bar")

    def test_bars_are_capped_at_max_bars(self):
        self.write_store(self.straight(count=40))
        cfg = make_config(max_bars=10)
        store = backtest.Store(self.dir, cfg)
        self.assertEqual(len(store.buffer_for("AAA_otc", BASE + 40 * PERIOD)), 10)


class TestReplayPlumbing(BacktestCase):
    """With the engine stubbed out, only the replay's bookkeeping is under test."""

    def setUp(self):
        super().setUp()
        self.signals = []
        patch = unittest.mock.patch.object(backtest, "evaluate", self._fake)
        patch.start()
        self.addCleanup(patch.stop)

    def _fake(self, candles, config):
        signal = Signal(direction="CALL", score=2, votes=["FAKE"],
                        price=candles[-1].close, time=candles[-1].time,
                        confidence=0.5)
        self.signals.append(candles[-1].time)
        return signal

    def test_entries_land_on_boundaries_and_expire_one_bar_later(self):
        self.write_store(self.straight(count=8))
        replay = self.replay()

        self.assertTrue(replay.trades)
        for trade in replay.trades:
            self.assertEqual(trade["entry_at"] % PERIOD, 0)
            # On a straight ramp each bar closes 0.001 above the last, so the two
            # settled prices must be exactly one bar apart.
            self.assertAlmostEqual(trade["exit"], trade["entry"] + 0.001, places=9)
            self.assertAlmostEqual(
                trade["entry"],
                replay.store.close_at(trade["asset"], trade["entry_at"] - PERIOD))

    def test_one_signal_per_boundary_and_the_cooldown_is_respected(self):
        self.write_store(self.straight(count=20))
        replay = self.replay()

        boundaries = [t["entry_at"] for t in replay.trades]
        self.assertEqual(boundaries, sorted(set(boundaries)))
        gaps = [b - a for a, b in zip(boundaries, boundaries[1:])]
        self.assertTrue(all(gap >= 120 for gap in gaps), gaps)

    def test_outcomes_come_from_the_two_bar_closes(self):
        self.write_store({"AAA_otc": [bar(BASE + i * PERIOD, 1.0 + i * 0.01)
                                      for i in range(8)]})
        replay = self.replay()

        self.assertTrue(replay.trades)
        self.assertTrue(all(t["outcome"] == "WIN" for t in replay.trades),
                        "a rising series must win every CALL")
        self.assertEqual(replay.losses, 0)

    def test_a_gap_in_the_store_produces_no_trade(self):
        candles = [bar(BASE + i * PERIOD, 1.0 + i * 0.001) for i in range(6)]
        del candles[3]                      # the exit bar for the first signal
        self.write_store({"AAA_otc": candles})
        replay = self.replay()

        for trade in replay.trades:
            self.assertIsNotNone(replay.store.close_at(trade["asset"],
                                                       trade["entry_at"] - PERIOD))
        self.assertLessEqual(len(replay.trades), 2,
                             "a trade whose exit bar is missing is not invented")

    def test_the_last_bar_cannot_be_judged(self):
        # Only the bars that have a successor can produce a settled trade.
        self.write_store(self.straight(count=3))
        replay = self.replay()
        for trade in replay.trades:
            self.assertLessEqual(trade["entry_at"], BASE + 3 * PERIOD)


class TestFullBarLeadHasNoLookahead(BacktestCase):
    """The replay must see what the live loop sees, at either lead.

    At a 10-second lead the live aggregator hands the engine a *snapshot* of the
    bar still in progress, so replaying that bar's finished close is ten seconds
    of optimism. At a full-bar lead the same shortcut becomes a whole bar of
    hindsight — the replay would judge on the close of the very bar the trade
    opens on, five minutes before the signal. Same call, opposite verdicts, so
    the boundary between them is worth pinning.
    """

    PERIOD = 300
    LEAD = 300
    START = 1_700_000_100.0          # a whole 300s boundary

    def bars(self, count=12):
        series = self.straight(count=count, start=self.START, period=self.PERIOD)
        self.write_store(series, period=self.PERIOD)
        return series["AAA_otc"]

    def store(self, **overrides):
        values = dict(candle_period=self.PERIOD, lead_seconds=self.LEAD,
                      expiry="5m", expiry_seconds=300,
                      signal=SignalConfig(bar_seconds=self.PERIOD, min_score=2,
                                          min_components=2))
        values.update(overrides)
        return backtest.Store(self.dir, make_config(**values))

    def test_a_full_bar_lead_does_not_read_the_entry_bar(self):
        candles = self.bars()
        boundary = int(candles[6].time + self.PERIOD)
        store = self.store()

        buffer = store.buffer_for("AAA_otc", boundary)

        self.assertTrue(buffer)
        self.assertEqual(buffer[-1].time, boundary - 2 * self.PERIOD,
                         "the newest bar judged was the last one to close before "
                         "the signal, not the bar the trade opens on")
        # The withheld bar really is in the store — this is a deliberate blind
        # spot, not a gap in the data.
        self.assertIsNotNone(store.close_at("AAA_otc", boundary - self.PERIOD))

    def test_the_short_lead_still_reads_the_bar_that_just_closed(self):
        candles = self.bars()
        boundary = int(candles[6].time + self.PERIOD)
        store = self.store(lead_seconds=10)

        buffer = store.buffer_for("AAA_otc", boundary)

        self.assertTrue(buffer)
        self.assertEqual(buffer[-1].time, boundary - self.PERIOD,
                         "a short lead keeps the snapshot regime — one bar later")

    def test_the_full_bar_lead_drops_exactly_the_last_bar(self):
        candles = self.bars()
        boundary = int(candles[6].time + self.PERIOD)

        short = self.store(lead_seconds=10).buffer_for("AAA_otc", boundary)
        full = self.store(lead_seconds=self.LEAD).buffer_for("AAA_otc", boundary)

        # Same window, truncated at the end — not slid back along the series.
        # The difference between the two regimes is one bar and nothing else.
        self.assertEqual(len(short), len(full) + 1)
        self.assertEqual([c.time for c in full], [c.time for c in short[:-1]])
        self.assertEqual([c.close for c in full], [c.close for c in short[:-1]])
        self.assertEqual(short[-1].time, boundary - self.PERIOD,
                         "the bar the long lead withholds is the entry bar")

    def test_a_lead_just_under_the_bar_keeps_the_short_regime(self):
        # The predicate is `lead < period`, so 299 is a short lead even though it
        # is nearly a whole bar. Off-by-one here would silently hide a bar.
        candles = self.bars()
        boundary = int(candles[6].time + self.PERIOD)

        buffer = self.store(lead_seconds=299).buffer_for("AAA_otc", boundary)

        self.assertEqual(buffer[-1].time, boundary - self.PERIOD)

    def test_trades_still_settle_on_the_entry_and_exit_bar_closes(self):
        # The decision window moved back a bar; the *settlement* prices must not
        # have moved with it, or the replay would grade trades against the wrong
        # bars entirely.
        self.bars(count=12)
        patch = unittest.mock.patch.object(
            backtest, "evaluate",
            lambda cs, cfg: Signal(direction="CALL", score=2, votes=["FAKE"],
                                   price=cs[-1].close, time=cs[-1].time,
                                   confidence=0.5))
        with patch:
            replay = backtest.Replay(self.store()).run()

        self.assertTrue(replay.trades)
        for trade in replay.trades:
            entry_bar = trade["entry_at"] - self.PERIOD
            self.assertAlmostEqual(trade["entry"],
                                   replay.store.close_at(trade["asset"], entry_bar))
            self.assertAlmostEqual(
                trade["exit"],
                replay.store.close_at(trade["asset"], entry_bar + self.PERIOD))


class TestRealEngineReporting(BacktestCase):
    """The diagnostic output itself: rates, reasons and the empty-store path."""

    def test_a_strongly_trending_series_is_gated_not_guessed(self):
        # 60 bars straight up: RSI and Bollinger call the top, so the PUT half of
        # the vote wins, but the trend gate vetos it. Nothing fires.
        self.write_store({"AAA_otc": [bar(BASE + i * PERIOD, 1.0 + i * 0.002)
                                      for i in range(60)]})
        replay = self.replay()

        self.assertEqual(replay.trades, [], "counter-trend is not traded")
        self.assertGreater(replay.judgements, 0)
        self.assertIn(backtest._GATED, replay.reasons)

    def test_impossible_selectivity_blames_the_score(self):
        self.write_store({"AAA_otc": [bar(BASE + i * PERIOD, 1.0 + (i % 3) * 0.001)
                                      for i in range(60)]})
        replay = self.replay(signal=SignalConfig(bar_seconds=PERIOD, min_score=99,
                                                 min_components=2))

        self.assertEqual(replay.trades, [])
        self.assertGreater(replay.reasons.get(backtest._WEAK, 0), 0)

    def test_warmup_is_reported_as_thin_data(self):
        self.write_store(self.straight(count=4))
        replay = self.replay()
        self.assertGreater(replay.reasons.get(backtest._THIN, 0), 0)

    def test_hours_measured_from_the_stored_bars(self):
        self.write_store(self.straight(count=61))    # 60 minutes of bars
        replay = self.replay()
        self.assertAlmostEqual(replay.hours(), 1.0, places=3)

    def test_empty_store_reports_cleanly(self):
        replay = self.replay()
        self.assertEqual(replay.trades, [])
        self.assertEqual(replay.hours(), 0.0)
        replay.report()      # must not raise

    def test_report_runs_on_a_populated_replay(self):
        self.write_store(self.straight(count=20))
        self.replay().report()


if __name__ == "__main__":
    unittest.main()
