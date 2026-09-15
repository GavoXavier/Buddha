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

    def test_the_ceiling_is_the_cap_the_store_actually_trims_to(self):
        # The same ``max_bars`` that caps the buffer caps the span, so a tool
        # quoting this ceiling is quoting a measured fact rather than a guess.
        self.write_store(self.straight(count=40))
        store = backtest.Store(self.dir, make_config(max_bars=10))

        self.assertAlmostEqual(store.ceiling_hours(), 10 * PERIOD / 3600.0)

    def test_the_span_never_exceeds_the_ceiling(self):
        self.write_store(self.straight(count=40))
        store = backtest.Store(self.dir, make_config(max_bars=10))

        self.assertLessEqual(store.span_hours(), store.ceiling_hours())

    def test_a_deeper_store_can_be_read_in_full(self):
        # What makes the archive replayable. The file holds far more than
        # MAX_BARS — it was written with its own, larger cap — and without this
        # the replay would read its newest MAX_BARS and report a sample no longer
        # than the live store's, silently, since the numbers would look exactly
        # like a working measurement.
        self.write_store(self.straight(count=40))
        cfg = make_config(max_bars=10)

        self.assertEqual(len(backtest.Store(self.dir, cfg).bars["AAA_otc"]), 10)
        deeper = backtest.Store(self.dir, cfg, bars=40)
        self.assertEqual(len(deeper.bars["AAA_otc"]), 40)
        self.assertAlmostEqual(deeper.ceiling_hours(), 40 * PERIOD / 3600.0)

    def test_the_hours_window_still_applies_to_a_deeper_read(self):
        # ``--bars`` raises the ceiling; it does not override ``--hours``. The two
        # are asked for separately and neither should quietly win.
        self.write_store(self.straight(count=40))
        store = backtest.Store(self.dir, make_config(max_bars=10), hours=0.5,
                               bars=40)

        # 31, not 30: the cutoff is inclusive, so the bar exactly half an hour
        # before the newest one is still inside the window.
        self.assertEqual(len(store.bars["AAA_otc"]), 31)


class TestHowMuchOfTheStoreCanWarmTheTrendVeto(BacktestCase):
    """Reachability, which is the one dial effect the settings cannot show.

    The trend veto needs ``TREND_EMA_LEN + TREND_SLOPE_BARS + 1`` bars of one
    unbroken run. A gate that is never warm is not a filter being applied, it is a
    filter that is not there — so the tool that reports the store has to report
    what depth of gate it could have supported.
    """

    # SignalConfig's defaults: 50 + 5 + 1 = 56, which no run here reaches.
    def test_a_run_too_shallow_for_the_gate_scores_nothing(self):
        self.write_store(self.straight(count=20))
        store = backtest.Store(self.dir, make_config())

        self.assertEqual(store.gate_reachability([50]), [(50, 0, 0.0)])

    def test_a_run_warms_the_gate_from_its_k_th_bar_onward(self):
        # 20 bars, a gate needing 16: warm for the last 5 of them.
        self.write_store(self.straight(count=20))
        store = backtest.Store(self.dir, make_config())

        self.assertEqual(store.gate_reachability([10]), [(10, 5, 0.25)])

    def test_a_shallower_gate_is_available_more_often(self):
        self.write_store(self.straight(count=40))
        store = backtest.Store(self.dir, make_config())

        rows = store.gate_reachability([10, 20, 30])
        shares = [share for _length, _warm, share in rows]

        self.assertEqual(shares, sorted(shares, reverse=True),
                         "a gate that needs fewer bars cannot be available less")

    def test_two_shallow_runs_are_not_one_deep_one(self):
        # The point of splitting by contiguity: bars either side of a hole are not
        # 40 bars of history, they are two 20-bar histories, and neither warms a
        # 26-bar gate.
        first = [bar(BASE + i * PERIOD, 1.0) for i in range(20)]
        second = [bar(BASE + (i + 40) * PERIOD, 1.0) for i in range(20)]
        self.write_store({"AAA_otc": first + second})
        store = backtest.Store(self.dir, make_config())

        self.assertEqual(len(store.runs["AAA_otc"]), 2)
        self.assertEqual(store.gate_reachability([20]), [(20, 0, 0.0)])

    def test_an_empty_store_has_a_share_rather_than_a_zero_division(self):
        store = backtest.Store(self.dir / "nope", make_config())

        self.assertEqual(store.gate_reachability([10]), [(10, 0, 0.0)])


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

    def test_a_signal_with_no_exit_bar_is_counted_not_invented(self):
        candles = [bar(BASE + i * PERIOD, 1.0 + i * 0.001) for i in range(6)]
        del candles[3]                      # the exit bar for the first signal
        self.write_store({"AAA_otc": candles})
        replay = self.replay()

        self.assertGreater(replay.unsettled, 0,
                           "the live loop would have had no price to settle on "
                           "either — that is a cost of the store, not a trade")
        self.assertLess(len(replay.trades), replay.candidates,
                        "a candidate whose exit bar is missing is not a trade")

    def test_minutes_with_nothing_to_judge_are_counted(self):
        # BBB_otc stops four bars in; every later boundary has to pass it over.
        self.write_store({**self.straight("AAA_otc", 8),
                          **self.straight("BBB_otc", 4)})
        replay = self.replay()

        self.assertGreater(replay.blind, 0)


class TestAStoreFileIsNotASeries(BacktestCase):
    """A window may not leave its contiguous run, nor end anywhere stale.

    ``CandleStore.save`` unions each session's bars with what is already on disk,
    so a hole cannot delete history — which means a store *file* holds bars on
    both sides of an outage that the live buffer never held together, because the
    aggregator drops the bars before such a hole. A replay that read the file as
    one series would judge windows the bot cannot have, reading each outage as a
    single bar's move; and a market whose feed has stopped would go on being
    judged from its last few bars at every later minute.
    """

    HOLE = 20           # bars missing between the two runs
    BEFORE = 6

    def two_runs(self, after=6):
        first = [bar(BASE + i * PERIOD, 1.0 + i * 0.001)
                 for i in range(self.BEFORE)]
        resume = BASE + (self.BEFORE + self.HOLE) * PERIOD
        second = [bar(resume + i * PERIOD, 2.0 + i * 0.001) for i in range(after)]
        self.write_store({"AAA_otc": first + second})
        return resume

    def test_the_file_is_indexed_as_two_runs(self):
        self.two_runs()
        store = backtest.Store(self.dir, make_config())
        self.assertEqual([len(r) for r in store.runs["AAA_otc"]],
                         [self.BEFORE, 6])
        self.assertEqual(store.deepest_run(), 6,
                         "the deepest history on offer is one run, not the file")

    def test_a_window_never_reaches_back_across_the_hole(self):
        resume = self.two_runs()
        store = backtest.Store(self.dir, make_config())
        boundary = int(resume + 2 * PERIOD)

        buffer = store.buffer_for("AAA_otc", boundary)

        self.assertEqual([c.close for c in buffer], [2.0, 2.001],
                         "only the bars of the run the boundary belongs to")
        self.assertTrue(all(c.time >= resume for c in buffer))

    def test_a_run_too_short_to_judge_produces_nothing(self):
        # The resumed run is one bar deep: there is no window to read from it,
        # even though the file holds twenty-odd bars behind it.
        self.two_runs(after=1)
        store = backtest.Store(self.dir, make_config())
        self.assertEqual(
            store.buffer_for("AAA_otc", int(BASE + (self.BEFORE + self.HOLE + 1)
                                           * PERIOD)), [])

    def test_a_market_that_stopped_is_not_judged_after_its_last_bar(self):
        self.write_store(self.straight(count=6))
        store = backtest.Store(self.dir, make_config())
        last = int(BASE + 5 * PERIOD)

        self.assertTrue(store.buffer_for("AAA_otc", last + PERIOD))
        self.assertEqual(store.buffer_for("AAA_otc", last + 2 * PERIOD), [],
                         "no bar closes at that moment, so there is nothing to "
                         "judge — the quiet market must not be re-read forever")


class TestReplayPlumbing(BacktestCase):
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
