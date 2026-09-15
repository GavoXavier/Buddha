"""Tests for tick->candle aggregation, persistence and the asset universe."""

import json
import tempfile
import unittest
from pathlib import Path

from market.aggregator import CandleSeries, MarketState
from market.clock import VirtualClock
from market.store import CandleStore
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

    def test_restore_keeps_an_unbroken_series_whole(self):
        vc = VirtualClock(start=BASE + 200 * 60)
        series = CandleSeries("EURUSD_otc", 60, 100, vc)
        bars = [Candle(time=BASE + i * 60, open=1.0, high=1.0, low=1.0, close=1.0)
                for i in range(15)]

        series.restore(bars)

        self.assertEqual(series.bar_count, 15)

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
