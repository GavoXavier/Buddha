"""Tests for tick->candle aggregation, persistence and the asset universe."""

import json
import tempfile
import unittest
from pathlib import Path

from market.aggregator import CandleSeries, MarketState, contiguous_runs
from market.clock import VirtualClock
from market.store import CandleStore, CandleTiers
from market.universe import (
    AssetMeta, MODE_ALL, MODE_FOREX, MODE_MAJOR, describe_skipped,
    is_currency_pair, is_major_pair, matches_mode, select_assets,
)
from signals.engine import Candle

# Exactly on a minute boundary, so every bar's open time is a whole minute.
BASE = 1_700_000_040.0


class TestCandleSeries(unittest.TestCase):
    def setUp(self):
        self.vc = VirtualClock(start=BASE)
        self.series = CandleSeries("EURUSD_otc", 60, 100, self.vc)

    def test_ticks_in_one_bucket_build_one_bar(self):
        self.series.add_tick(BASE + 1, 1.10)
        self.series.add_tick(BASE + 5, 1.12)
        self.series.add_tick(BASE + 30, 1.09)
        self.series.add_tick(BASE + 59, 1.11)

        self.assertEqual(self.series.bar_count, 0, "bar still forming")
        self.series.finalize(BASE + 60)

        closed = self.series.closed()
        self.assertEqual(len(closed), 1)
        bar = closed[0]
        self.assertEqual(bar.time, BASE, "bars are labelled by their OPEN time")
        self.assertAlmostEqual(bar.open, 1.10)
        self.assertAlmostEqual(bar.high, 1.12)
        self.assertAlmostEqual(bar.low, 1.09)
        self.assertAlmostEqual(bar.close, 1.11)

    def test_next_bucket_closes_previous_bar(self):
        self.series.add_tick(BASE, 1.10)
        self.vc = self.vc  # clock unchanged: a tick may arrive up to one bar ahead
        self.series.add_tick(BASE + 60, 1.20)

        self.assertEqual(self.series.bar_count, 1)
        self.assertAlmostEqual(self.series.closed()[0].close, 1.10)
        forming = self.series.forming()
        self.assertIsNotNone(forming)
        self.assertEqual(forming.time, BASE + 60)

    def test_gap_is_recorded_not_invented(self):
        self.series.add_tick(BASE, 1.0)
        self.vc._now = BASE + 180
        self.series.add_tick(BASE + 180, 1.1)

        self.assertEqual(self.series.gaps, 1)
        self.assertEqual(self.series.last_gap_seconds, 180)
        # The silent buckets must NOT become bars.
        self.assertEqual([c.time for c in self.series.closed()], [BASE])

    def _warm_then_resume(self, missing):
        """20 warm bars, a hole of ``missing`` minutes, then the feed returns."""
        resume = BASE + (20 + missing) * 60
        vc = VirtualClock(start=resume)
        series = CandleSeries("EURUSD_otc", 60, 100, vc)
        for i in range(20):
            series.add_tick(BASE + i * 60, 1.0 + i / 1000)
        series.add_tick(resume, 1.5)          # the feed comes back...
        series.add_tick(resume + 60, 1.6)     # ...and the next bar closes
        return series, resume

    def test_a_long_hole_drops_the_bars_before_it(self):
        # An indicator window that spans a hole reads the whole outage as one
        # bar's move. Live, a 92-minute outage in which EURUSD moved 48 pips
        # pinned RSI and Stochastic to the same extreme and produced a signal
        # every minute off the outage rather than off the market — so the older
        # side of the hole goes, and the buffer warms up again.
        for missing in (12, 92):
            with self.subTest(outage_minutes=missing):
                series, resume = self._warm_then_resume(missing)

                self.assertEqual(series.gaps, 1)
                self.assertEqual(series.bar_count, 1,
                                 "only the bar after the hole survives")
                self.assertEqual(series.closed()[0].time, resume)

    def test_a_stray_missing_bar_keeps_the_buffer(self):
        # One missing minute is noise, not a different series: dropping a warm
        # buffer over it would cost ~20 minutes of signals for nothing.
        series, resume = self._warm_then_resume(1)

        self.assertEqual(series.gaps, 1)
        self.assertEqual(series.bar_count, 21,
                         "the 20 warm bars and the one after the hole")

    def test_restore_keeps_only_the_run_after_the_last_hole(self):
        # A restart must not re-seed a series spanning an outage: the store is
        # written before the process dies, so it holds the hole.
        vc = VirtualClock(start=BASE + 200 * 60)
        series = CandleSeries("EURUSD_otc", 60, 100, vc)
        stale = [Candle(time=BASE + i * 60, open=1.0, high=1.0, low=1.0, close=1.0)
                 for i in range(12)]
        fresh = [Candle(time=BASE + (92 + i) * 60, open=1.1, high=1.1,
                        low=1.1, close=1.1) for i in range(3)]

        series.restore(stale + fresh)

        self.assertEqual(series.bar_count, 3)
        self.assertTrue(all(c.time >= BASE + 92 * 60 for c in series.closed()),
                        "the bars from before the outage were dropped")

    def test_a_hole_is_reported_in_minutes_as_well_as_bars(self):
        # Both messages used to print the *bar* count with the word "minutes",
        # which was near enough at 60s bars and wrong by 5x at 300s: a 10-bar
        # hole read as "a 10-minute hole" when it was fifty. A two-hour outage
        # then looked like a blip for as long as it was being diagnosed.
        vc = VirtualClock(start=BASE + 100 * 300)
        series = CandleSeries("EURUSD_otc", 300, 100, vc)
        for i in range(20):
            series.add_tick(BASE + i * 300, 1.0)

        with self.assertLogs("pocket.market", level="WARNING") as logs:
            series.add_tick(BASE + 30 * 300, 1.5)   # 10 missing 5-minute bars

        self.assertIn("10 bar(s) missing from the feed (50 min at 300s bars)",
                      "\n".join(logs.output))

    def test_the_restore_report_counts_bars_and_minutes(self):
        vc = VirtualClock(start=BASE + 200 * 300)
        series = CandleSeries("EURUSD_otc", 300, 100, vc)
        stale = [Candle(BASE + i * 300, 1, 1, 1, 1) for i in range(12)]
        fresh = [Candle(BASE + (92 + i) * 300, 1, 1, 1, 1) for i in range(3)]

        with self.assertLogs("pocket.market", level="WARNING") as logs:
            series.restore(stale + fresh)

        # 81 buckets apart, so 80 of them never became bars: 400 minutes, not 80.
        self.assertIn("span 80 missing bar(s) (400 min at 300s bars)",
                      "\n".join(logs.output))

    def test_restore_keeps_an_unbroken_series_whole(self):
        vc = VirtualClock(start=BASE + 200 * 60)
        series = CandleSeries("EURUSD_otc", 60, 100, vc)
        bars = [Candle(time=BASE + i * 60, open=1.0, high=1.0, low=1.0, close=1.0)
                for i in range(15)]

        series.restore(bars)

        self.assertEqual(series.bar_count, 15)

    def _restore_then_resume(self, missing, period=60):
        """A restored series, then the first tick ``missing`` bars later.

        This is what a restart looks like from the series' point of view: the
        restore has no idea when the next tick will arrive, so whether the two
        are one series can only be judged when it does. Bars are aligned to the
        period, as the store's are — a bucket is a multiple of the period, and
        an unaligned bar in a test would make the arithmetic say something the
        real series never does.
        """
        start = int(BASE // period * period)
        resume = start + (20 + missing) * period
        vc = VirtualClock(start=resume)
        series = CandleSeries("EURUSD_otc", period, 100, vc)
        series.restore([Candle(start + i * period, 1.0, 1.0, 1.0, 1.0)
                        for i in range(20)])
        series.add_tick(resume, 1.5)              # the session's first tick...
        series.add_tick(resume + period, 1.6)     # ...and the next bar closes
        return series, resume, start

    def test_a_stale_restore_is_dropped_by_the_first_tick(self):
        # A restored series can be arbitrarily old — a market that stopped being
        # subscribed keeps its bars but gets no new ones. Live on 2026-09-15 a
        # series ending at 21:00 took its next tick at 21:45, the two were read
        # as one series, and an RSI window was put across a 45-minute outage:
        # the signal sent off it was noise dressed as a setup.
        for missing in (12, 92):
            with self.subTest(outage_bars=missing):
                series, resume, _ = self._restore_then_resume(missing)

                self.assertEqual(series.gaps, 1)
                self.assertEqual(series.bar_count, 1,
                                 "only the bar after the hole survives")
                self.assertEqual(series.closed()[0].time, resume)

    def test_a_restore_resumed_within_tolerance_keeps_its_history(self):
        # Ordinary restart: the bars stop a bar or two before the tick, which is
        # a dropped packet, not a different series. Dropping the buffer here
        # would cost the whole warm-up after every quick restart.
        series, _, start = self._restore_then_resume(3)

        self.assertEqual(series.bar_count, 21, "the 20 restored bars and the one after")
        self.assertEqual(series.closed()[0].time, start)

    def test_the_stale_restore_is_reported_in_minutes_and_bars(self):
        # Eight missing 5-minute bars is forty minutes, and the twenty bars
        # thrown away with them are worth saying out loud.
        with self.assertLogs("pocket.market", level="WARNING") as logs:
            series, _, _ = self._restore_then_resume(8, period=300)

        self.assertEqual(series.bar_count, 1)
        self.assertIn("8 bar(s) missing from the feed (40 min at 300s bars)",
                      "\n".join(logs.output))
        self.assertIn("dropped 20 bar(s)", "\n".join(logs.output))

    def test_a_first_tick_inside_the_last_restored_bar_is_not_a_gap(self):
        # The restart that lands back in the same bucket as the newest restored
        # bar — the common case — must not be mistaken for an outage.
        vc = VirtualClock(start=BASE + 19 * 60 + 30)
        series = CandleSeries("EURUSD_otc", 60, 100, vc)
        series.restore([Candle(BASE + i * 60, 1.0, 1.0, 1.0, 1.0) for i in range(20)])

        series.add_tick(BASE + 19 * 60 + 30, 1.5)

        self.assertEqual(series.gaps, 0)
        self.assertEqual(series.bar_count, 20, "nothing was thrown away")

    def test_late_tick_for_closed_bar_is_dropped(self):
        self.series.add_tick(BASE, 1.10)
        self.series.finalize(BASE + 60)
        self.series.add_tick(BASE + 30, 9.99)  # arrives after its bar closed

        self.assertEqual(len(self.series.closed()), 1)
        self.assertAlmostEqual(self.series.closed()[0].high, 1.10)
        self.assertEqual(self.series.forming(), None)

    def test_nonsense_ticks_are_ignored(self):
        for ts, price in [(BASE, float("nan")), (BASE, -1.0), (BASE, 0.0),
                          (float("inf"), 1.0), (BASE + 10_000, 1.0)]:
            self.series.add_tick(ts, price)
        self.assertEqual(self.series.bar_count, 0)
        self.assertEqual(self.series.forming(), None)

    def test_finalize_is_idempotent(self):
        self.series.add_tick(BASE, 1.10)
        self.series.finalize(BASE + 60)
        self.series.finalize(BASE + 300)
        self.assertEqual(self.series.bar_count, 1)

    def make_closed_bar(self, open_at, close, high=None, low=None):
        """Feed a whole bar and close it."""
        self.vc._now = open_at
        self.series.add_tick(open_at + 1, close)
        self.series.add_tick(open_at + 30, high if high is not None else close)
        self.series.add_tick(open_at + 30.5, low if low is not None else close)
        self.vc._now = open_at + 59
        self.series.add_tick(open_at + 59, close)
        self.vc._now = open_at + 60
        self.series.finalize(open_at + 60)

    def test_buffer_for_boundary_appends_the_forming_bar(self):
        for i in range(4):
            self.make_closed_bar(BASE + i * 60, 1.0 + i * 0.01)
        # The next bar is in progress.
        self.vc._now = BASE + 240
        self.series.add_tick(BASE + 240 + 1, 1.50)

        # A trade placed at BASE+300 is judged on the bar closing at BASE+300,
        # which is the one still forming right now.
        buf = self.series.buffer_for_boundary(BASE + 300)
        self.assertEqual(len(buf), 5)
        self.assertEqual([c.time for c in buf],
                         [BASE, BASE + 60, BASE + 120, BASE + 180, BASE + 240])
        self.assertAlmostEqual(buf[-1].close, 1.50)

    def test_buffer_for_boundary_uses_real_bar_when_already_closed(self):
        for i in range(5):
            self.make_closed_bar(BASE + i * 60, 1.0 + i * 0.01)

        # Woken after the boundary passed: the bar ending at BASE+300 is closed,
        # so it must be used as-is rather than re-snapshotted as forming.
        buf = self.series.buffer_for_boundary(BASE + 300)
        self.assertEqual(len(buf), 5)
        self.assertEqual(buf[-1].time, BASE + 240)
        self.assertAlmostEqual(buf[-1].close, 1.04)

    def test_price_at_boundary_is_the_close_of_the_bar_ending_there(self):
        self.make_closed_bar(BASE, 1.11)
        self.make_closed_bar(BASE + 60, 1.22)

        self.assertAlmostEqual(self.series.price_at_boundary(BASE + 60), 1.11)
        self.assertAlmostEqual(self.series.price_at_boundary(BASE + 120), 1.22)
        self.assertIsNone(self.series.price_at_boundary(BASE + 180))

    def test_staleness(self):
        self.vc._now = BASE
        self.series.add_tick(BASE, 1.0)
        self.assertFalse(self.series.is_stale(120, self.vc.now()))
        self.vc._now = BASE + 121
        self.assertTrue(self.series.is_stale(120, self.vc.now()))

    def test_stale_when_never_ticked(self):
        self.assertTrue(self.series.is_stale(120, BASE))

    def test_restore_seeds_buffer_and_starts_a_fresh_bucket(self):
        stored = [Candle(BASE + i * 60, 1.0, 1.1, 0.9, 1.05) for i in range(3)]
        self.series.restore(stored)
        self.assertEqual(self.series.bar_count, 3)
        self.assertIsNone(self.series.forming(), "no bucket in progress after restore")

        # The first live tick opens a new bucket that holds only what we saw.
        self.vc._now = BASE + 300
        self.series.add_tick(BASE + 300, 1.30)
        self.series.finalize(BASE + 360)
        self.assertEqual(self.series.bar_count, 4)
        self.assertEqual(self.series.closed()[-1].time, BASE + 300)

    def test_max_bars_is_respected(self):
        series = CandleSeries("X", 60, 3, self.vc)
        for i in range(10):
            self.vc._now = BASE + i * 60
            series.add_tick(BASE + i * 60, 1.0 + i)
            self.vc._now = BASE + i * 60 + 60
            series.finalize(BASE + i * 60 + 60)
        self.assertEqual(series.bar_count, 3)
        self.assertEqual([c.time for c in series.closed()],
                         [BASE + 420, BASE + 480, BASE + 540])


class TestContiguousRuns(unittest.TestCase):
    """One rule for "may these two bars be read as neighbours", three callers.

    The live feed, the restore path and the backtester all have to agree about a
    hole, and they agree by calling this. The store writes additively — a hole
    must not be able to delete history — so a *file* can hold bars on both sides
    of an outage that no live series ever held together, which is exactly where a
    second, subtly different copy of the rule would go unnoticed.
    """

    PERIOD = 60

    def bars(self, times):
        return [Candle(float(t), 1.0, 1.0, 1.0, 1.0) for t in times]

    def runs(self, missing_bars, max_gap_bars=5):
        before = [BASE + i * self.PERIOD for i in range(4)]
        after = [before[-1] + (missing_bars + 1) * self.PERIOD + i * self.PERIOD
                 for i in range(3)]
        return contiguous_runs(self.bars(before + after), self.PERIOD,
                               max_gap_bars)

    def test_an_unbroken_series_is_one_run(self):
        self.assertEqual(len(self.runs(missing_bars=0)), 1)

    def test_a_hole_within_the_tolerance_is_still_one_run(self):
        # A stray missing bar reads as a slightly longer bar, which is noise.
        self.assertEqual(len(self.runs(missing_bars=5)), 1)

    def test_a_hole_wider_than_the_tolerance_starts_a_new_run(self):
        runs = self.runs(missing_bars=6)
        self.assertEqual([len(r) for r in runs], [4, 3])

    def test_the_runs_come_back_oldest_first_whatever_the_order_given(self):
        shuffled = self.bars([BASE, BASE + 3 * self.PERIOD, BASE + self.PERIOD,
                              BASE + 2 * self.PERIOD])
        runs = contiguous_runs(list(reversed(shuffled)), self.PERIOD)
        self.assertEqual([[c.time for c in run] for run in runs],
                         [[BASE + i * self.PERIOD for i in range(4)]],
                         "not sorted, and not reversed, and no empty run after")

    def test_no_bars_is_no_runs(self):
        self.assertEqual(contiguous_runs([], self.PERIOD), [])

    def test_a_gap_is_counted_in_bars_not_seconds(self):
        # The same hole is 6 missing bars at a 60s period and 2 at 300s. Getting
        # this wrong is how a two-hour outage once looked like a blip.
        long_bars = contiguous_runs(self.bars(
            [BASE + i * 300 for i in range(4)]
            + [BASE + (4 + 2) * 300 + i * 300 for i in range(2)]), 300, 5)
        self.assertEqual(len(long_bars), 1, "2 missing 300s bars is inside the "
                                            "tolerance of 5")


class TestMarketState(unittest.TestCase):
    def setUp(self):
        self.vc = VirtualClock(start=BASE)
        self.market = MarketState(60, 100, 120, self.vc)

    def test_healthy_requires_a_bar_and_a_live_feed(self):
        self.market.on_tick("AAA_otc", BASE, 1.0)
        self.assertEqual(self.market.healthy(BASE), [], "no closed bar yet")

        self.vc._now = BASE + 60
        self.market.finalize(BASE + 60)
        self.assertEqual(self.market.healthy(BASE + 60), ["AAA_otc"])

        # Feed goes quiet for longer than stale_after.
        self.assertEqual(self.market.healthy(BASE + 60 + 121), [])

    def test_track_is_created_on_demand(self):
        self.assertEqual(self.market.symbols(), [])
        self.market.on_tick("NEW_otc", BASE, 1.0)
        self.assertEqual(self.market.symbols(), ["NEW_otc"])

    def test_stalest_sorts_by_silence(self):
        self.market.on_tick("OLD_otc", BASE, 1.0)
        self.market.on_tick("NEW_otc", BASE + 50, 1.0)
        self.assertEqual([s for s, _ in self.market.stalest()], ["OLD_otc", "NEW_otc"])


class TestCandleStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.store = CandleStore(self.dir, 60, 100)

    def test_round_trip(self):
        candles = [Candle(BASE + i * 60, 1.0 + i, 1.2 + i, 0.8 + i, 1.1 + i)
                   for i in range(5)]
        self.store.save("EURUSD_otc", candles, force=True)

        loaded = self.store.load("EURUSD_otc")
        self.assertEqual(len(loaded), 5)
        for original, restored in zip(candles, loaded):
            self.assertEqual(original, restored)

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(self.store.load("NOPE_otc"), [])

    def test_period_mismatch_is_discarded(self):
        self.store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        other = CandleStore(self.dir, 300, 100)
        self.assertEqual(other.load("EURUSD_otc"), [],
                         "bars from a different period must never be mixed in")

    def test_a_simulated_store_is_discarded_by_a_live_run(self):
        # The same directory and the same symbols, so nothing but this check
        # stands between a fabricated price path and the live engine's history.
        store = CandleStore(self.dir, 60, 100, feed="simulated")
        store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        live = CandleStore(self.dir, 60, 100, feed="pocket_option")
        self.assertEqual(live.load("EURUSD_otc"), [])

    def test_the_same_feed_round_trips(self):
        store = CandleStore(self.dir, 60, 100, feed="pocket_option")
        store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        again = CandleStore(self.dir, 60, 100, feed="pocket_option")
        self.assertEqual(len(again.load("EURUSD_otc")), 1)

    def test_a_store_that_does_not_say_which_feed_wrote_it_is_not_trusted(self):
        # Files written before origin was recorded, or by a caller that did not
        # name a feed. "Unknown" cannot be shown to be the right feed, so it is
        # treated the same as the wrong one.
        self.store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        self.assertEqual(self.store.feed, "")
        self.assertEqual(CandleStore(self.dir, 60, 100,
                                     feed="pocket_option").load("EURUSD_otc"), [])

    def test_a_named_store_is_not_read_by_a_caller_that_names_nothing(self):
        store = CandleStore(self.dir, 60, 100, feed="pocket_option")
        store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        self.assertEqual(CandleStore(self.dir, 60, 100).load("EURUSD_otc"), [])

    def test_the_feed_is_recorded_in_the_file(self):
        CandleStore(self.dir, 60, 100, feed="pocket_option").save(
            "EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        doc = json.loads((self.dir / "EURUSD_otc.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["feed"], "pocket_option")

    def test_corrupt_file_is_ignored(self):
        (self.dir / "EURUSD_otc.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.load("EURUSD_otc"), [])

    def test_debounce_skips_the_second_write(self):
        one = [Candle(BASE, 1, 1, 1, 1)]
        two = [Candle(BASE, 2, 2, 2, 2)]
        self.store.save("EURUSD_otc", one)
        self.store.save("EURUSD_otc", two)  # within save_interval -> skipped
        self.assertEqual(self.store.load("EURUSD_otc")[0].close, 1)

        self.store.save("EURUSD_otc", two, force=True)
        self.assertEqual(self.store.load("EURUSD_otc")[0].close, 2)

    def test_load_all_skips_symbols_without_data(self):
        self.store.save("A_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        out = self.store.load_all(["A_otc", "B_otc"])
        self.assertEqual(list(out), ["A_otc"])

    def test_a_shorter_series_does_not_erase_the_bars_before_it(self):
        # The aggregator drops every bar before a long hole, so the series it
        # hands the store is legitimately truncated. Overwriting would write
        # that truncation through to disk and turn temporary blindness into
        # permanent loss: measured on 2026-09-15, one teardown flush after a
        # churn-induced gap took AUDUSD from 19 bars to 2 and no later session
        # could recover them.
        self.store.save("EURUSD_otc",
                        [Candle(BASE + i * 60, 1.0, 1.0, 1.0, 1.0) for i in range(20)],
                        force=True)
        self.store.save("EURUSD_otc",
                        [Candle(BASE + (20 + i) * 60, 2.0, 2.0, 2.0, 2.0)
                         for i in range(2)],
                        force=True)

        merged = self.store.load("EURUSD_otc")
        self.assertEqual([c.time for c in merged],
                         [BASE + i * 60 for i in range(22)])
        self.assertEqual(merged[0].close, 1.0, "the older bars kept their values")

    def test_a_re_saved_bar_replaces_the_persisted_one(self):
        self.store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        self.store.save("EURUSD_otc", [Candle(BASE, 9, 9, 9, 9)], force=True)

        loaded = self.store.load("EURUSD_otc")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].close, 9)

    def test_the_merge_still_honours_max_bars(self):
        store = CandleStore(self.dir, 60, 3)
        store.save("X_otc", [Candle(BASE + i * 60, 1, 1, 1, 1) for i in range(3)],
                   force=True)
        store.save("X_otc", [Candle(BASE + (3 + i) * 60, 1, 1, 1, 1)
                             for i in range(3)], force=True)

        self.assertEqual([c.time for c in store.load("X_otc")],
                         [BASE + 180, BASE + 240, BASE + 300])

    def test_a_foreign_store_is_not_merged_into(self):
        # The merge reads the disk through ``load``, so it inherits the same
        # discarding rules: a save must not resurrect bars that the period check
        # has just rejected. The feed check goes further and refuses the write
        # outright — see below — so the period is what shows the merge rule alone.
        CandleStore(self.dir, 300, 100).save(
            "EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        live = CandleStore(self.dir, 60, 100)
        live.save("EURUSD_otc", [Candle(BASE + 60, 2, 2, 2, 2)], force=True)

        self.assertEqual([c.time for c in live.load("EURUSD_otc")], [BASE + 60])

    def test_a_save_will_not_overwrite_another_feeds_history(self):
        # Read-discarding is not enough on its own. A simulated session in the
        # live store's directory reads nothing (the feed check) and then writes
        # over everything, because the merge it does on the way out sees an empty
        # disk. Measured on 2026-09-16: a 50-bar live store became a 3-bar
        # simulated one, and none of the 50 could be read back afterwards. The
        # watched bars are the one input the broker will not re-supply.
        live = CandleStore(self.dir, 60, 100, feed="pocket_option")
        live.save("EURUSD_otc", [Candle(BASE + i * 60, 1, 1, 1, 1)
                                 for i in range(50)], force=True)

        CandleStore(self.dir, 60, 100, feed="simulated").save(
            "EURUSD_otc", [Candle(BASE + 900000, 9, 9, 9, 9)], force=True)

        kept = live.load("EURUSD_otc")
        self.assertEqual(len(kept), 50, "the real history is untouched")
        self.assertEqual(kept[0].close, 1.0, "and it is still the real bars")

    def test_the_refusal_is_per_file_rather_than_per_directory(self):
        # The two feeds share a store directory in practice; what must not be
        # shared is a single market's series. A market the other feed has never
        # written stays writable, so a simulated run beside a live one is only
        # refused the files it would actually destroy.
        CandleStore(self.dir, 60, 100, feed="pocket_option").save(
            "EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        sim = CandleStore(self.dir, 60, 100, feed="simulated")
        sim.save("GBPUSD_otc", [Candle(BASE, 2, 2, 2, 2)], force=True)

        self.assertEqual(len(sim.load("GBPUSD_otc")), 1)

    def test_a_file_that_cannot_be_read_is_still_written_over(self):
        # A corrupt or half-written file is not history: there is nothing behind
        # it to protect, and refusing would leave the market permanently unable
        # to store anything.
        (self.dir / "EURUSD_otc.json").write_text("{not json", encoding="utf-8")
        self.store.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)

        self.assertEqual(len(self.store.load("EURUSD_otc")), 1)

    def test_a_file_of_unknown_origin_is_not_written_over_either(self):
        # Written before the feed field existed. It cannot be shown to be the
        # right feed, so it is not read from — and by the same rule it is not
        # written over, because it may well be real history.
        (self.dir / "EURUSD_otc.json").write_text(
            json.dumps({"version": 1, "period": 60, "asset": "EURUSD_otc",
                        "candles": [[BASE, 1, 1, 1, 1]]}), encoding="utf-8")
        live = CandleStore(self.dir, 60, 100, feed="pocket_option")
        live.save("EURUSD_otc", [Candle(BASE + 60, 2, 2, 2, 2)], force=True)

        self.assertEqual(live.load("EURUSD_otc"), [])

    def test_a_refused_write_is_logged_at_most_once_per_interval(self):
        # Otherwise a simulated session beside a live one logs an error per
        # market per bar, which is how a real warning gets scrolled away.
        live = CandleStore(self.dir, 60, 100, feed="pocket_option")
        live.save("EURUSD_otc", [Candle(BASE, 1, 1, 1, 1)], force=True)
        sim = CandleStore(self.dir, 60, 100, feed="simulated")
        with self.assertLogs("pocket.store", level="ERROR") as caught:
            for i in range(5):
                sim.save("EURUSD_otc", [Candle(BASE + 900000 + i * 60, 9, 9, 9, 9)])

        self.assertEqual(len(caught.records), 1)


class _Series:
    """Enough of ``CandleSeries`` for ``save_all``: what would be written out."""

    def __init__(self, candles):
        self._candles = candles

    def closed(self):
        return list(self._candles)


class TestCandleTiers(unittest.TestCase):
    """The live store, plus a deeper copy that only a replay ever reads.

    ``MAX_BARS`` is two decisions in one setting — the live buffer's memory, and
    the whole sample a selectivity dial can be checked against — and the second
    one is what runs out. These pin that the two tiers can differ in depth
    without the engine ever seeing the deeper one.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.live_dir = Path(self.tmp.name) / "candles"
        self.deep_dir = Path(self.tmp.name) / "candles-archive"
        self.addCleanup(self.tmp.cleanup)

    def tiers(self, live_bars=3, deep_bars=50, feed=""):
        live = CandleStore(self.live_dir, 60, live_bars, feed=feed)
        deep = CandleStore(self.deep_dir, 60, deep_bars, feed=feed)
        return CandleTiers(live, deep), deep

    @staticmethod
    def series(count, start=0, close=1.0):
        return [Candle(BASE + (start + i) * 60, close, close, close, close)
                for i in range(count)]

    def test_the_engine_reads_the_live_tier_only(self):
        # The archive holds bars the live store was never given. If ``load``
        # could reach them the archive would be a strategy change wearing the
        # label of a storage one.
        tiers, deep = self.tiers()
        deep.save("EURUSD_otc", self.series(9), force=True)

        self.assertEqual(tiers.load("EURUSD_otc"), [])
        self.assertEqual(len(deep.load("EURUSD_otc")), 9)

    def test_the_archive_keeps_what_the_live_store_discards(self):
        tiers, deep = self.tiers(live_bars=3, deep_bars=50)
        tiers.save("EURUSD_otc", self.series(10), force=True)

        self.assertEqual(len(tiers.load("EURUSD_otc")), 3, "the live cap still applies")
        self.assertEqual(len(deep.load("EURUSD_otc")), 10)

    def test_one_save_reaches_both_tiers(self):
        tiers, deep = self.tiers()
        tiers.save_all({"EURUSD_otc": _Series(self.series(4))}, force=True)

        self.assertEqual(len(tiers.load("EURUSD_otc")), 3)
        self.assertEqual(len(deep.load("EURUSD_otc")), 4)

    def test_the_live_buffer_and_the_archive_keep_the_same_bars(self):
        # Not merely the same count: a bar the live store kept must be the bar
        # the archive kept, or a replay would be measuring a different market.
        tiers, deep = self.tiers(live_bars=3, deep_bars=50)
        tiers.save("EURUSD_otc", self.series(10), force=True)

        self.assertEqual([c.time for c in tiers.load("EURUSD_otc")],
                         [c.time for c in deep.load("EURUSD_otc")][-3:])

    def test_the_archive_is_written_on_its_own_slower_clock(self):
        # It is a copy, not a buffer waiting to be resumed: rewriting a
        # 20,000-bar file every thirty seconds would be all cost and no gain,
        # because the merge makes each write additive anyway.
        tiers, deep = self.tiers()

        self.assertEqual(deep.save_interval, 900.0)
        self.assertGreater(deep.save_interval, tiers.live.save_interval)

    def test_a_truncated_series_does_not_shrink_the_archive(self):
        # The aggregator drops every bar before a long hole, so a shorter series
        # is legitimate. The archive inherits the live store's union-with-disk,
        # which is what stops a hole in one session deleting history from the
        # copy that exists to remember it.
        tiers, deep = self.tiers(live_bars=500, deep_bars=500)
        tiers.save("EURUSD_otc", self.series(20), force=True)
        tiers.save("EURUSD_otc", self.series(2, start=20), force=True)

        self.assertEqual(len(deep.load("EURUSD_otc")), 22)

    def test_an_unwritable_archive_does_not_end_the_save(self):
        # A copy is a convenience. A mistyped CANDLE_ARCHIVE_DIR must cost the
        # copy and not the session, or archiving would be a way to lose trades.
        blocked = Path(self.tmp.name) / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        tiers = CandleTiers(CandleStore(self.live_dir, 60, 3),
                            CandleStore(blocked, 60, 50))

        tiers.save("EURUSD_otc", self.series(5), force=True)

        self.assertEqual(len(tiers.load("EURUSD_otc")), 3)

    def test_no_archive_is_a_live_store_with_a_wrapper(self):
        tiers = CandleTiers(CandleStore(self.live_dir, 60, 3))
        tiers.save("EURUSD_otc", self.series(5), force=True)

        self.assertIsNone(tiers.archive)
        self.assertEqual(len(tiers.load("EURUSD_otc")), 3)
        self.assertEqual(tiers.archived(["EURUSD_otc"]), {})

    def test_seeding_gives_an_empty_archive_the_history_beside_it(self):
        # Without this an upgraded bot would start an archive that is shallower
        # than the live store next to it — the deeper copy holding less than the
        # original.
        tiers, deep = self.tiers(live_bars=20, deep_bars=50)
        candles = self.series(20)
        # History that predates the archive: the live store has it and the new
        # directory does not, which is what an upgrade looks like.
        tiers.live.save("EURUSD_otc", candles, force=True)
        self.assertEqual(deep.load("EURUSD_otc"), [])

        self.assertTrue(tiers.seed("EURUSD_otc", candles))
        self.assertEqual(len(deep.load("EURUSD_otc")), 20)

    def test_seeding_leaves_an_archive_that_already_has_history_alone(self):
        # The archive's own series is the union of everything ever watched, so
        # it can hold runs the live store's cap has already trimmed away.
        # Seeding over it would be the copy overwriting the original.
        tiers, deep = self.tiers(live_bars=20, deep_bars=50)
        deep.save("EURUSD_otc", self.series(40), force=True)
        candles = self.series(3)

        self.assertFalse(tiers.seed("EURUSD_otc", candles))
        self.assertEqual(len(deep.load("EURUSD_otc")), 40)

    def test_there_is_nothing_to_seed_from_one_bar(self):
        tiers, _ = self.tiers()
        self.assertFalse(tiers.seed("EURUSD_otc", self.series(1)))

    def test_seeding_without_an_archive_is_a_no_op(self):
        tiers = CandleTiers(CandleStore(self.live_dir, 60, 3))
        self.assertFalse(tiers.seed("EURUSD_otc", self.series(9)))

    def test_the_archive_carries_the_live_stores_own_feed_stamp(self):
        # The feed is what stops simulated bars being replayed as live ones, and
        # a second writer is a second chance to lose that stamp.
        tiers, deep = self.tiers(feed="simulated")
        tiers.save("EURUSD_otc", self.series(4), force=True)

        self.assertEqual(deep.feed, "simulated")
        self.assertEqual(CandleStore(self.deep_dir, 60, 50).load("EURUSD_otc"), [])


def meta(symbol, payout=85, is_otc=None, active=True):
    if is_otc is None:
        is_otc = symbol.endswith("_otc")
    return AssetMeta(symbol=symbol, payout=payout, is_otc=is_otc, active=active)


class TestUniverse(unittest.TestCase):
    def test_currency_pair_detection(self):
        self.assertTrue(is_currency_pair("EURUSD_otc"))
        self.assertTrue(is_currency_pair("USDKES"))
        self.assertFalse(is_currency_pair("BTCUSD_otc"))
        self.assertFalse(is_currency_pair("#AAPL_otc"))
        self.assertFalse(is_currency_pair("XAUUSD_otc"))

    def test_major_pair_detection(self):
        self.assertTrue(is_major_pair("EURUSD_otc"))
        self.assertTrue(is_major_pair("GBPJPY_otc"))
        self.assertFalse(is_major_pair("USDKES_otc"), "KES is not a major")

    def test_modes(self):
        self.assertTrue(matches_mode("EURUSD_otc", MODE_MAJOR))
        self.assertTrue(matches_mode("EURUSD_otc", MODE_FOREX))
        self.assertTrue(matches_mode("BTCUSD_otc", MODE_ALL))
        self.assertFalse(matches_mode("BTCUSD_otc", MODE_FOREX))
        self.assertFalse(matches_mode("USDKES_otc", MODE_MAJOR))

    def test_select_assets_filters_and_sorts_by_payout(self):
        metas = {
            "EURUSD_otc": meta("EURUSD_otc", payout=70),
            "GBPUSD_otc": meta("GBPUSD_otc", payout=92),
            "USDKES_otc": meta("USDKES_otc", payout=90),   # not a major
            "BTCUSD_otc": meta("BTCUSD_otc", payout=95),   # not forex
            "AUDUSD_otc": meta("AUDUSD_otc", payout=40),   # below the floor
            "NZDUSD": meta("NZDUSD"),                      # real, not OTC
        }
        picked = select_assets(metas, MODE_MAJOR, min_payout=60, otc_only=True)
        self.assertEqual(picked, ["GBPUSD_otc", "EURUSD_otc"])

    def test_otc_filter_falls_back_when_it_would_leave_nothing(self):
        metas = {"EURUSD": meta("EURUSD", payout=80),
                 "GBPUSD": meta("GBPUSD", payout=75)}
        picked = select_assets(metas, MODE_MAJOR, min_payout=60, otc_only=True)
        self.assertEqual(picked, ["EURUSD", "GBPUSD"],
                         "a silent bot is worse than a weekday-only one")

    def test_inactive_assets_are_dropped(self):
        metas = {"EURUSD_otc": meta("EURUSD_otc", active=False),
                 "GBPUSD_otc": meta("GBPUSD_otc")}
        self.assertEqual(select_assets(metas, MODE_MAJOR, 0), ["GBPUSD_otc"])

    def test_empty_metas_is_empty(self):
        self.assertEqual(select_assets({}, MODE_MAJOR, 0), [])

    def test_describe_skipped_counts_each_reason(self):
        metas = {
            "EURUSD_otc": meta("EURUSD_otc", payout=90),
            "BTCUSD_otc": meta("BTCUSD_otc"),          # wrong kind
            "AUDUSD_otc": meta("AUDUSD_otc", payout=10),  # low payout
            "GBPUSD_otc": meta("GBPUSD_otc", active=False),
            "NZDUSD": meta("NZDUSD"),                  # not otc
        }
        reasons = describe_skipped(metas, MODE_MAJOR, 60, otc_only=True)
        self.assertEqual(reasons["wrong_kind"], 1)
        self.assertEqual(reasons["low_payout"], 1)
        self.assertEqual(reasons["inactive"], 1)
        self.assertEqual(reasons["not_otc"], 1)

    def test_display_formatting(self):
        self.assertEqual(meta("EURUSD_otc").display, "EUR/USD OTC")
        self.assertEqual(meta("EURUSD").display, "EUR/USD")
        self.assertEqual(meta("#AAPL_otc").display, "#AAPL OTC")


if __name__ == "__main__":
    unittest.main()
