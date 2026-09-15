"""The sweep, and the two things it must never do.

It must not pick a setting, and it must not produce a table out of a store too
short for the table to mean anything. Both are tested here, because both are
failures that look like success from the outside: a confident table is exactly
what a reader wants and exactly what a five-hour store cannot support.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import calibrate
from config import Config
from market.store import CandleStore
from signals.engine import Candle, SignalConfig

BASE = 1_700_000_040.0
PERIOD = 60


def make_config(**overrides) -> Config:
    values = dict(
        feed="simulated", candle_period=PERIOD, max_bars=1000,
        expiry="1m", expiry_seconds=60, lead_seconds=10, cooldown_seconds=120,
        signal=SignalConfig(bar_seconds=PERIOD, min_score=2, min_components=2))
    values.update(overrides)
    return Config(**values)


def bar(open_at, close=1.0):
    return Candle(float(open_at), close, close + 0.001, close - 0.001, close)


class CalibrateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_store(self, count: int, step: float = 0.001) -> None:
        store = CandleStore(self.dir, PERIOD, 1000, feed="simulated")
        candles = [bar(BASE + i * PERIOD, 1.0 + i * step) for i in range(count)]
        store.save("AAA_otc", candles, force=True)

    def captured(self, *args, **kwargs) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = calibrate.sweep(*args, **kwargs)
        return code, buffer.getvalue()


class TestTheVariationsStayOneDialFromShipped(CalibrateCase):
    """A cross product over six dials is unreadable and its best row is luck.

    Varying one dial at a time answers the question actually being asked — what
    does *this* dial buy — and keeps every row comparable to the shipped row.
    """

    def test_the_shipped_settings_are_the_first_row(self):
        base = make_config().signal

        rows = calibrate.variations(base)

        self.assertEqual(rows[0], ("shipped", base))

    def test_every_row_is_the_baseline_with_exactly_one_field_moved(self):
        base = make_config().signal

        rows = calibrate.variations(base)[1:]

        self.assertTrue(rows, "a sweep with nothing to vary is not a sweep")
        for label, signal in rows:
            moved = [f for f in vars(base) if getattr(base, f) != getattr(signal, f)]
            self.assertEqual(len(moved), 1, f"{label} moved {moved}")

    def test_a_value_the_shipped_settings_already_use_is_not_a_row(self):
        # Otherwise the table carries a duplicate of the baseline under another
        # name, and the eye reads it as a second sample.
        base = make_config(signal=SignalConfig(bar_seconds=PERIOD, min_score=1,
                                               min_components=2)).signal

        labels = [label for label, _signal in calibrate.variations(base)]

        self.assertNotIn("MIN_SCORE=1", labels)
        self.assertIn("MIN_SCORE=3", labels)


class TestTheSplit(CalibrateCase):
    def test_the_cut_sits_inside_the_span_at_the_asked_fraction(self):
        self.write_store(count=100)
        store = calibrate.Store(self.dir, make_config())

        cut = calibrate.split_at(store, 0.6)

        first = min(b.time for bars in store.bars.values() for b in bars)
        last = max(b.time for bars in store.bars.values() for b in bars)
        self.assertAlmostEqual(cut, first + (last - first) * 0.6)

    def test_an_empty_store_has_a_cut_rather_than_a_crash(self):
        store = calibrate.Store(self.dir / "nope", make_config())

        self.assertEqual(calibrate.split_at(store, 0.6), 0.0)

    def test_the_two_windows_do_not_overlap_and_keep_the_same_market(self):
        self.write_store(count=100)
        cfg = make_config()
        store = calibrate.Store(self.dir, cfg)
        cut = calibrate.split_at(store, 0.6)

        train = calibrate.Store(self.dir, cfg, window=(0.0, cut))
        test = calibrate.Store(self.dir, cfg, window=(cut, float("inf")))

        self.assertEqual(sorted(train.bars), ["AAA_otc"])
        self.assertEqual(sorted(test.bars), ["AAA_otc"])
        self.assertLess(max(b.time for b in train.bars["AAA_otc"]), cut)
        self.assertGreaterEqual(min(b.time for b in test.bars["AAA_otc"]), cut)


class TestTheVerdictOnlyClaimsWhatTheSampleSupports(CalibrateCase):
    """The rule is ``reconcile.py``'s: the lower bound must clear break-even."""

    def row(self, signals, wins):
        low, high = calibrate.wilson(wins, signals)
        return {"signals": signals, "wins": wins, "low": low, "high": high}

    def test_a_rate_above_break_even_with_a_wide_interval_is_not_a_win(self):
        # 2 of 3 at 54.1% break-even: looks like 67%, decides nothing.
        self.assertEqual(calibrate.verdict_of(self.row(3, 2), 0.541), "no edge")

    def test_a_sample_that_clears_the_lower_bound_beats_it(self):
        self.assertEqual(calibrate.verdict_of(self.row(400, 280), 0.541), "beats")

    def test_a_sample_whose_upper_bound_is_under_is_losing(self):
        self.assertEqual(calibrate.verdict_of(self.row(400, 160), 0.541), "under")

    def test_one_trade_is_never_a_verdict(self):
        self.assertEqual(calibrate.verdict_of(self.row(1, 1), 0.541), "too few")
        self.assertEqual(calibrate.verdict_of(self.row(1, 0), 0.541), "too few")


class TestItRefusesToSweepASampleTooSmallToRead(CalibrateCase):
    """The refusal is the point of the tool, so it is pinned like one."""

    def test_a_small_sample_is_refused_without_a_table(self):
        self.write_store(count=60)          # one hour of one-minute bars
        cfg = make_config()

        code, out = self.captured(cfg, str(self.dir), 0.0, 0.6, 85.0, 30)

        self.assertEqual(code, 1, "a refusal is not a success")
        self.assertNotIn(calibrate.TABLE_HEADER, out,
                         "there must be no table to misread")
        self.assertIn("a column is printed from 30", out)
        self.assertIn("--min-trades", out, "it says how to override itself")

    def test_the_override_prints_the_table(self):
        self.write_store(count=60)
        cfg = make_config()

        code, out = self.captured(cfg, str(self.dir), 0.0, 0.6, 85.0, 0)

        self.assertEqual(code, 0)
        self.assertIn(calibrate.TABLE_HEADER, out)
        self.assertIn("shipped", out)

    def test_an_empty_store_is_reported_rather_than_swept(self):
        cfg = make_config()

        code, out = self.captured(cfg, str(self.dir / "nope"), 0.0, 0.6, 85.0, 0)

        self.assertEqual(code, 1)
        self.assertIn("No bars in the store", out)


class TestTheRefusalSaysWhetherWaitingWouldEvenHelp(CalibrateCase):
    """A store that has rolled over cannot grow, so "wait a few days" can be a lie.

    ``CandleStore.save`` keeps the newest ``max_bars`` and drops the oldest, which
    puts a hard ceiling on the trades a store can hold at a given signal rate. When
    that ceiling is under the gate the tool has to say so, because the alternative —
    telling the operator to wait for a sample that can never accumulate — is advice
    that reads as patience and is actually a dead end.
    """

    def test_a_settled_rate_under_the_gate_cannot_be_waited_out(self):
        # 300 bars is 5h, well past the 0.9h warm-up, so the rate is the engine's
        # rate: 1/h over the 6.7h held-out ceiling is 7 trades, and no amount of
        # uptime reaches 30.
        self.write_store(count=300)
        store = calibrate.Store(self.dir, make_config())

        text = calibrate.describe_pending(store, 2, 2.0, 30, 0.541, 0.6)

        self.assertIn("rolling window", text)
        self.assertIn("cannot close the gap", text)
        self.assertNotIn("more of uptime", text,
                         "waiting is not offered when it cannot work")

    def test_a_short_store_calls_its_rate_a_floor_rather_than_a_forecast(self):
        # 1h of store against a 0.9h warm-up: the zero rate here may be the warm-up
        # still finishing, and saying "waiting cannot help" would be wrong.
        self.write_store(count=60)
        store = calibrate.Store(self.dir, make_config())

        text = calibrate.describe_pending(store, 0, 0.4, 30, 0.541, 0.6)

        self.assertIn("floor and not a forecast", text)
        self.assertNotIn("cannot close the gap", text)

    def test_a_rate_that_can_reach_the_gate_says_roughly_how_long(self):
        self.write_store(count=60)
        store = calibrate.Store(self.dir, make_config())

        text = calibrate.describe_pending(store, 4, 0.4, 30, 0.541, 0.6)

        self.assertIn("clears the gate", text)
        self.assertIn("more of uptime", text)

    def test_the_ceiling_quoted_is_the_stores_own_cap(self):
        # 1000 bars at 60s is ~16.7h — the figure in the text has to be that one,
        # not a guess, or the arithmetic under it is fiction.
        self.write_store(count=60)
        store = calibrate.Store(self.dir, make_config())

        text = calibrate.describe_pending(store, 4, 0.4, 30, 0.541, 0.6)

        self.assertIn("16.7h", text)
        self.assertIn("1000 bars", text)
        self.assertIn("6.7h of that held out", text)

    def test_the_rate_is_measured_on_the_window_the_gate_was_computed_from(self):
        # The held-out window is the unit throughout: the same number of trades in
        # a tenth of the window is ten times the rate, and ten times the reach.
        self.write_store(count=60)
        store = calibrate.Store(self.dir, make_config())

        wide = calibrate.describe_pending(store, 4, 6.0, 30, 0.541, 0.6)
        narrow = calibrate.describe_pending(store, 4, 0.6, 30, 0.541, 0.6)

        self.assertIn("0.67 signals/hour", wide)
        self.assertIn("6.67 signals/hour", narrow)


class TestTheReading(CalibrateCase):
    def test_a_table_of_nothing_is_reported_as_nothing(self):
        rows = [("shipped", {"signals": 0}, {"signals": 0}),
                ("MIN_SCORE=1", {"signals": 1}, {"signals": 1})]

        text = calibrate.reading(rows, 0.541)

        self.assertIn("Nothing here is distinguishable", text)

    def test_a_row_that_clears_the_lower_bound_is_named_and_still_not_a_verdict(self):
        decided, wins = 400, 280
        low, high = calibrate.wilson(wins, decided)
        held = {"signals": decided, "wins": wins, "low": low, "high": high}
        rows = [("USE_TREND=False", {"signals": 0}, held)]

        text = calibrate.reading(rows, 0.541)

        self.assertIn("USE_TREND=False", text)
        self.assertIn("not a verdict", text,
                      "one sweep read once is how a strategy gets fitted to noise")


class TestTheAssumedPayoutIsStated(CalibrateCase):
    def test_the_break_even_is_the_one_the_assumed_payout_gives(self):
        self.write_store(count=60)
        cfg = make_config()

        _code, out = self.captured(cfg, str(self.dir), 0.0, 0.6, 92.0, 0)

        self.assertIn("52.1%", out, "1 / (1 + 0.92)")
        self.assertIn("assumption", out,
                      "the store records no payouts, and the line must say so")


if __name__ == "__main__":
    unittest.main()
