"""Tests for the one-minute cadence: boundary maths, entry/exit timing, results.

These drive the scheduler directly on a ``VirtualClock`` — no sleeping, no feed —
so each minute of the trading loop is exercised at exactly the instant the real
loop wakes, and every entry/exit price can be traced back to a known bar close.

The timing model under test (period 60s, lead 10s):

    T-10s  wake, judge the bar about to close, send the signal for entry T
    T      the trade opens
    T+50s  the bar closing at T is finalised  -> exact entry price
    T+110s the bar closing at T+60 is finalised -> exact exit price, result sent
"""

import asyncio
import logging
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import engine.scheduler as sched_mod
from engine.scheduler import (
    Cadence, MinuteScheduler, OpenTrade, next_boundary, outcome_of,
)
from journal import SignalJournal, load_journal
from market.aggregator import MarketState
from market.clock import VirtualClock
from signals.engine import Readiness, Signal, SignalConfig
from signals.ranking import Candidate, eligible
from stats import StatsTracker
from telegram.control import BotController

BASE = 1_700_000_040.0  # exactly on a minute boundary
PERIOD = 60


def fake_evaluate(direction="CALL", score=2, confidence=0.6):
    """Stand-in for the engine: returns a Signal for any buffer it is given."""
    def _fake(candles, config):
        return Signal(direction=direction, score=score, votes=["FAKE"],
                      price=candles[-1].close, time=candles[-1].time,
                      confidence=confidence)
    return _fake


class FakeSender:
    def __init__(self):
        self.signals = []
        self.confirmations = []
        self.texts = []

    async def send_signal(self, signal, asset, expiry, entry_at, now=None,
                          payout=0, martingale_steps=0, entry_price=None):
        self.signals.append({"asset": asset, "direction": signal.direction,
                             "expiry": expiry, "entry_at": entry_at,
                             "price": entry_price, "payout": payout})
        return {"ok": True}

    async def send_confirmation(self, asset, direction, outcome, wins, losses,
                                win_rate, expiry_at=None, streak="", note=""):
        self.confirmations.append({"asset": asset, "direction": direction,
                                   "outcome": outcome, "expiry_at": expiry_at,
                                   "win_rate": win_rate, "note": note})
        return {"ok": True}

    async def send_text(self, text):
        self.texts.append(text)
        return {"ok": True}


class Harness:
    """A market, a scheduler, and a way to replay bars into it.

    The bar length comes from the cadence, so the same harness drives the
    1-minute cadence and the 5-minute one. Everything expressed in seconds
    (the stale window, the tick offsets) is derived from it rather than
    hardcoded, or a 300s bar would arrive pre-stale and never be judged.
    """

    def __init__(self, direction="CALL", cadence=None, symbols=("AAA_otc",),
                 config=None, payouts=None, min_payout=0, journal=None):
        self.cadence = cadence or Cadence()
        self.period = self.cadence.period
        self.vc = VirtualClock(start=BASE)
        # The stale window must exceed one bar or a healthy market is dropped
        # mid-bar; config enforces the same rule. Identical at period=60.
        self.stale_after = max(120.0, self.period * 2.0)
        self.market = MarketState(self.period, 200, self.stale_after, self.vc)
        self.sender = FakeSender()
        self.controller = BotController()
        self.symbols = list(symbols)
        self._tmp = tempfile.TemporaryDirectory()
        self.stats = StatsTracker(str(Path(self._tmp.name) / "stats.json"), None)
        self.scheduler = MinuteScheduler(
            market=self.market, sender=self.sender, stats=self.stats,
            controller=self.controller, cadence=self.cadence,
            engine_config=config or SignalConfig(), symbols=self.symbols,
            payouts={s: 85 for s in self.symbols} if payouts is None else payouts,
            min_payout=min_payout, clock=self.vc, journal=journal)

    @property
    def tmpdir(self) -> Path:
        """The scratch directory this harness cleans up with itself."""
        return Path(self._tmp.name)

    def feed(self, end, close, high=None, low=None):
        """Replay the bar that *ends* at ``end`` (opens ``end - period``).

        The clock is parked at the bar's open, as a feed replaying live data
        would be, and is left there: the caller decides when the cycle runs.
        """
        open_at = end - self.period
        hi_at = open_at + self.period / 6.0
        self.vc._now = open_at
        for name in self.symbols:
            series = self.market.track(name)
            series.add_tick(open_at, close)                    # sets the open
            series.add_tick(hi_at, high if high else close)
            series.add_tick(hi_at + 0.5, low if low else close)
            series.add_tick(open_at + self.period - 1, close)  # the close

    async def cycle(self, boundary, wake=None):
        """Run one cycle as the real loop does: waking ``lead`` before the entry."""
        lead = self.cadence.lead_seconds if wake is None else wake
        self.vc._now = boundary - lead
        return await self.scheduler.run_cycle(boundary)

    async def play(self, closes, start=None):
        """Replay one bar per period, running the cycle for each bar in turn."""
        start = (BASE - self.period * len(closes)) if start is None else start
        results = []
        for i, close in enumerate(closes):
            boundary = int(start + i * self.period)
            self.feed(boundary, close)
            results.append(await self.cycle(boundary))
        return results

    def cleanup(self):
        self._tmp.cleanup()


class SchedulerCase(unittest.TestCase):
    """Shared fixture: a harness with the engine stubbed out to CALL."""

    direction = "CALL"

    def setUp(self):
        self.h = Harness(direction=self.direction)
        self.addCleanup(self.h.cleanup)
        patch = unittest.mock.patch.object(sched_mod, "evaluate",
                                           fake_evaluate(self.direction))
        patch.start()
        self.addCleanup(patch.stop)

    def play(self, closes, start=None):
        return asyncio.run(self.h.play(closes, start))


class TestBoundaryMaths(unittest.TestCase):
    def test_next_boundary_is_strictly_later(self):
        self.assertEqual(next_boundary(0, 60), 60)
        self.assertEqual(next_boundary(59.9, 60), 60)
        self.assertEqual(next_boundary(60, 60), 120, "that boundary has just passed")
        self.assertEqual(next_boundary(60.1, 60), 120)

    def test_next_boundary_aligns_to_wall_clock(self):
        for now in (BASE, BASE + 0.5, BASE + 59.99):
            boundary = next_boundary(now, 60)
            self.assertEqual(boundary % 60, 0)
            self.assertGreater(boundary, now)

    def test_outcome(self):
        self.assertEqual(outcome_of("CALL", 100.0, 100.1), "WIN")
        self.assertEqual(outcome_of("CALL", 100.0, 99.9), "LOSS")
        self.assertEqual(outcome_of("PUT", 100.0, 99.9), "WIN")
        self.assertEqual(outcome_of("PUT", 100.0, 100.1), "LOSS")
        self.assertEqual(outcome_of("CALL", 100.0, 100.0), "LOSS", "a tie is not a win")
        self.assertEqual(outcome_of("PUT", 100.0, 100.0), "LOSS")


class TestSignalTiming(SchedulerCase):
    def test_entry_is_on_the_boundary_and_expiry_is_one_bar_later(self):
        results = self.play([99.0, 100.0, 101.0])

        self.assertTrue(any(r.chosen for r in results), "one minute should fire")
        self.assertEqual(len(self.h.sender.signals), 1)
        sent = self.h.sender.signals[0]
        entry_at = int(self.h.scheduler.open_trades[0].entry_at)

        self.assertEqual(entry_at % PERIOD, 0, "entry lands on the boundary")
        self.assertEqual(sent["entry_at"], entry_at)
        self.assertEqual(sent["expiry"], "1m")
        self.assertEqual(self.h.scheduler.open_trades[0].expiry_at, entry_at + 60)
        # The price quoted with the signal is the close of the bar it judged.
        self.assertEqual(sent["price"], 100.0)

    def test_entry_price_is_the_close_of_the_bar_ending_at_entry(self):
        self.play([99.0, 100.0, 101.0])

        trade = self.h.scheduler.open_trades[0]
        self.assertEqual(trade.entry_at, BASE - 120)
        # The entry bar closed a minute later, so the price is now exact.
        self.assertEqual(trade.entry_price, 100.0)
        self.assertIsNone(trade.exit_price, "the trade is still open")

    def test_win_is_scored_from_two_bar_closes(self):
        results = self.play([99.0, 100.0, 101.0, 102.0])

        self.assertEqual(len(results[3].resolved), 1)
        trade = results[3].resolved[0]
        self.assertEqual(trade.entry_price, 100.0, "close of the bar ending at entry")
        self.assertEqual(trade.exit_price, 101.0, "close of the bar ending at expiry")
        self.assertNotIn(trade, self.h.scheduler.open_trades, "settled trades are closed out")
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "WIN")
        self.assertEqual(self.h.stats.wins, 1)
        self.assertEqual(self.h.stats.losses, 0)
        self.assertEqual(self.h.stats.win_rate, 1.0)

    def test_loss_is_scored_from_two_bar_closes(self):
        results = self.play([99.0, 100.0, 99.5, 99.0])

        trade = results[3].resolved[0]
        self.assertEqual(trade.entry_price, 100.0)
        self.assertEqual(trade.exit_price, 99.5)
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "LOSS")
        self.assertEqual(self.h.stats.losses, 1)
        self.assertEqual(self.h.stats.win_rate, 0.0)

    def test_flat_close_is_a_loss(self):
        results = self.play([99.0, 100.0, 100.0, 99.0])
        self.assertEqual(results[3].resolved[0].exit_price, 100.0)
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "LOSS")

    def test_result_is_not_reported_before_the_exit_bar_closes(self):
        results = self.play([99.0, 100.0, 101.0])
        self.assertEqual(results[2].resolved, [], "exit bar is still forming")
        self.assertEqual(self.h.sender.confirmations, [])
        self.assertEqual(self.h.stats.total, 0)
        self.assertEqual(len(self.h.scheduler.open_trades), 1)

    def test_at_most_one_signal_per_minute(self):
        results = self.play([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        for result in results:
            self.assertLessEqual(len(self.h.sender.signals), len(results))
        per_boundary = [r.boundary for r in results if r.chosen]
        self.assertEqual(len(per_boundary), len(set(per_boundary)),
                         "never two signals for the same boundary")

    def test_cooldown_holds_a_market_back_for_one_minute(self):
        results = self.play([100.0, 100.5, 101.0, 101.5])
        # Fires on the first minute it can, then sits out the next one.
        self.assertTrue(results[1].chosen)
        self.assertFalse(results[2].chosen)
        self.assertIsNotNone(results[3].chosen, "eligible again after the cooldown")
        self.assertEqual(len(self.h.sender.signals), 2)

    def test_late_wake_still_places_the_trade_on_the_boundary(self):
        self.h.feed(BASE - 60, 100.0)
        self.h.feed(BASE, 101.0)
        # Woken 5s *after* the boundary (a slow cycle): the bar is already closed,
        # so it must be judged as a real bar and the entry still goes on the line.
        result = asyncio.run(self.h.cycle(BASE, wake=-5))
        self.assertIsNotNone(result.chosen)
        self.assertEqual(self.h.sender.signals[0]["entry_at"], BASE)

    def test_expiry_that_is_not_a_whole_number_of_bars_still_resolves(self):
        # A 90s expiry never lands on a bar boundary, so the exit has to fall
        # back to the latest tick instead of hanging the trade forever.
        self.h.cadence.expiry_seconds = 90
        self.h.cadence.cooldown_seconds = 3600     # exactly one trade, to keep it clear
        closes = [100.0, 100.0, 100.0, 100.0, 95.0, 95.0, 95.0, 95.0]
        results = self.play(closes)

        self.assertEqual(self.h.scheduler.open_trades, [],
                         "the trade must not hang waiting for a bar that never comes")
        self.assertEqual(self.h.stats.total, 1)
        self.assertEqual(self.h.sender.confirmations[0]["expiry_at"],
                         self.h.sender.signals[0]["entry_at"] + 90)
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "LOSS")
        self.assertTrue(any(r.resolved for r in results))


class TestPutDirection(SchedulerCase):
    """The same timing, mirrored: a PUT must win when the price falls."""

    direction = "PUT"

    def test_put_wins_when_price_falls(self):
        results = self.play([101.0, 100.0, 99.0, 98.0])

        trade = results[3].resolved[0]
        self.assertEqual(self.h.sender.signals[0]["direction"], "PUT")
        self.assertEqual((trade.entry_price, trade.exit_price), (100.0, 99.0))
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "WIN")

    def test_put_loses_when_price_rises(self):
        results = self.play([99.0, 100.0, 101.0, 102.0])

        trade = results[3].resolved[0]
        self.assertEqual((trade.entry_price, trade.exit_price), (100.0, 101.0))
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "LOSS")


class TestControl(SchedulerCase):
    def test_pause_stops_signals_and_resume_restores_them(self):
        self.h.controller.pause()
        results = self.play([100.0, 101.0, 102.0])
        self.assertTrue(all(r.paused for r in results))
        self.assertEqual(self.h.sender.signals, [])

        self.h.controller.resume()
        results = self.play([103.0, 104.0])
        self.assertFalse(any(r.paused for r in results))
        self.assertTrue(any(r.chosen for r in results))

    def test_circuit_breaker_pauses_after_consecutive_losses(self):
        self.h.cadence.max_consecutive_losses = 2
        self.h.cadence.pause_minutes = 15
        # Every pair of bars closes 5 lower, so every trade loses.
        closes = [100.0 - 5.0 * (i // 2) for i in range(8)]
        results = self.play(closes)

        self.assertEqual(self.h.stats.consecutive_losses(), 2)
        self.assertTrue(results[6].paused, "trading stops after the second loss")
        self.assertTrue(results[7].paused)
        self.assertEqual(len(self.h.sender.signals), 2,
                         "nothing is sent while the breaker is on")
        self.assertIn("pausing", " ".join(self.h.sender.texts))

    def test_circuit_breaker_lifts_on_its_own(self):
        self.h.cadence.max_consecutive_losses = 2
        self.h.cadence.pause_minutes = 15
        closes = [100.0 - 5.0 * (i // 2) for i in range(8)]
        self.play(closes)

        self.h.vc._now = self.h.scheduler._paused_until - 1
        self.assertTrue(self.h.scheduler._blocked(self.h.vc.now()))
        self.h.vc._now = self.h.scheduler._paused_until + 1
        self.assertFalse(self.h.scheduler._blocked(self.h.vc.now()))

    def test_confidence_floor_rejects_a_weak_setup(self):
        self.h.cadence.min_confidence = 0.9
        results = self.play([100.0, 101.0, 102.0])
        chosen = [r for r in results if r.chosen]
        self.assertEqual(chosen, [])
        self.assertTrue(any(r.candidates for r in results),
                        "the setup existed, it was just not good enough")

    def test_stale_market_is_not_judged(self):
        self.play([100.0, 101.0, 102.0])
        self.h.vc._now = BASE + 600          # the feed has gone quiet
        result = asyncio.run(self.h.scheduler.run_cycle(BASE + 660))
        self.assertEqual(result.healthy, 0)
        self.assertIsNone(result.chosen)

    def test_empty_market_is_judged_without_error(self):
        result = asyncio.run(self.h.cycle(BASE))
        self.assertIsNone(result.chosen)
        self.assertEqual(result.healthy, 0)
        self.assertEqual(result.candidates, [])

    def test_expiry_label(self):
        for seconds, label in ((60, "1m"), (300, "5m"), (3600, "1h"), (45, "45s")):
            self.h.cadence.expiry_seconds = seconds
            self.assertEqual(self.h.scheduler._expiry_label(), label)

    def test_status_reports_progress_and_the_next_entry(self):
        self.play([100.0, 101.0])
        # The loop records the boundary it is waiting for; /status reports it.
        self.h.scheduler.next_entry_at = BASE
        text = self.h.scheduler.status_text()
        self.assertIn("Status", text)
        self.assertIn("1/1 live", text)
        self.assertIn("Next entry", text)

    def test_stats_are_recorded_against_the_expiry_timestamp(self):
        self.play([99.0, 100.0, 101.0, 102.0])

        self.assertEqual(self.h.stats.total, 1)
        self.assertEqual(self.h.stats.recent[-1]["at"],
                         self.h.sender.confirmations[0]["expiry_at"])
        self.assertEqual(self.h.stats.recent[-1]["asset"], "AAA_otc")
        self.assertEqual(self.h.stats.asset_tally("AAA_otc").wins, 1)


class TestLoop(unittest.TestCase):
    """The loop itself: it must wake before each boundary, once per boundary."""

    def test_loop_wakes_before_every_boundary_in_order(self):
        h = Harness()
        self.addCleanup(h.cleanup)

        async def scenario():
            wakes = []
            real_run_cycle = h.scheduler.run_cycle

            async def recording_run_cycle(boundary):
                wakes.append((h.vc.now(), boundary))
                return await real_run_cycle(boundary)

            h.scheduler.run_cycle = recording_run_cycle
            task = asyncio.get_running_loop().create_task(h.scheduler.run())
            for i in range(1, 7):
                h.feed(BASE + i * PERIOD, 100.0 + i)
                await h.vc.advance(float(PERIOD), steps=120)
            h.controller.request_stop()
            await h.vc.advance(float(PERIOD), steps=4)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True),
                                   timeout=5.0)
            return wakes

        wakes = asyncio.run(scenario())

        self.assertTrue(wakes, "the loop should have judged at least one minute")
        for now, boundary in wakes:
            self.assertEqual(boundary % PERIOD, 0, "judged on a boundary")
            self.assertLess(now, boundary, "woken before the entry, not after it")
            self.assertLessEqual(boundary - now, h.cadence.lead_seconds + 1)
        boundaries = [b for _, b in wakes]
        self.assertEqual(boundaries, sorted(set(boundaries)),
                         "one judgement per boundary, in order")


class TestFullBarLead(unittest.TestCase):
    """period == lead == 300: signal 16:00, entry 16:05, expiry 16:10.

    The long-lead regime. At a short lead the engine judges a snapshot of the
    bar about to close; here the bar that closes at the entry boundary has not
    started when the signal goes out, so there is nothing to snapshot and the
    decision rests on the bars that have already closed.
    """

    PERIOD = 300
    LEAD = 300
    # A whole 300s boundary (1700000100 / 300 == 5666667). The aggregator
    # buckets bars by ``floor(ts / period)``, so a tick off the boundary lands
    # in a bar whose open time is not the one the cycle goes looking for.
    T0 = 1_700_000_100.0

    def setUp(self):
        cadence = Cadence(period=self.PERIOD, expiry_seconds=self.PERIOD,
                          lead_seconds=self.LEAD)
        self.h = Harness(cadence=cadence)
        self.addCleanup(self.h.cleanup)
        self.judged: list[tuple[int, float]] = []
        patch = unittest.mock.patch.object(sched_mod, "evaluate", self._record)
        patch.start()
        self.addCleanup(patch.stop)

    def _record(self, candles, config):
        """Keep the *shape* of the buffer, not just the signal it produced."""
        self.judged.append((len(candles), candles[-1].time))
        return Signal(direction="CALL", score=2, votes=["FAKE"],
                      price=candles[-1].close, time=candles[-1].time,
                      confidence=0.6)

    def _two_closed_bars(self) -> None:
        """Bars opening T0-600 and T0-300 — both closed by the signal at T0."""
        self.h.feed(self.T0 - self.PERIOD, 1.0)
        self.h.feed(self.T0, 1.0)

    def test_the_engine_judges_the_newest_bar_closed_at_the_signal(self):
        self._two_closed_bars()
        asyncio.run(self.h.cycle(self.T0 + self.PERIOD))

        self.assertEqual(self.judged, [(2, self.T0 - self.PERIOD)],
                         "both closed bars, the newest opening one bar before T0")

    def test_a_tick_in_the_entry_bar_is_not_fed_to_the_engine(self):
        """Why the branch exists — and it is a millisecond race.

        A single tick in the opening instants of the entry bar is enough for
        ``buffer_for_boundary`` to append a zero-second bar: open == high == low
        == close, one price wearing a bar's clothes, landing in the middle of
        every indicator window. Whether it is there when the cycle runs is
        timing, so the distortion would come and go — the worst kind to chase.
        """
        self._two_closed_bars()
        series = self.h.market.track(self.h.symbols[0])
        series.add_tick(self.T0, 1.5)          # the entry bar's very first tick

        boundary = int(self.T0 + self.PERIOD)
        snapshot = series.buffer_for_boundary(boundary)
        self.assertEqual(len(snapshot), 3, "the snapshot path adds the stub")
        self.assertEqual(snapshot[-1].time, self.T0, "and it is zero seconds old")

        asyncio.run(self.h.cycle(boundary))
        self.assertEqual(self.judged, [(2, self.T0 - self.PERIOD)],
                         "the stub is not a bar, so it is not judged")

    def test_a_short_lead_still_judges_the_bar_in_progress(self):
        """The other regime is untouched: a 10s lead keeps its snapshot.

        Guards the branch itself — collapsing both cases onto ``closed()`` would
        silently move the 1-minute cadence's decision back a whole bar.
        """
        self.h.cadence.lead_seconds = 10
        self._two_closed_bars()
        # add_tick refuses a timestamp more than one bar ahead of the clock, so
        # the replay has to move to the bar before the tick arrives.
        self.h.vc._now = self.T0
        self.h.market.track(self.h.symbols[0]).add_tick(self.T0 + 50, 1.9)
        asyncio.run(self.h.cycle(self.T0 + self.PERIOD))

        self.assertEqual(self.judged, [(3, self.T0)],
                         "the bar still in progress at the signal is included")

    def test_entries_advance_one_bar_and_never_overlap(self):
        """The shape that was asked for: one trade live at any moment.

        Note what "one at a time" does *not* mean here. When a signal goes out
        the book legitimately holds two trades — the one currently live and the
        one just signalled for five minutes' time. What must never happen is two
        of them being live at once, so the assertion is on the intervals
        [entry, expiry], not on the length of the book.
        """
        self._two_closed_bars()          # something to judge the first bar on
        seen: list[tuple[float, float]] = []
        live_counts: list[int] = []
        for i in range(1, 7):
            boundary = int(self.T0 + i * self.PERIOD)
            self.h.feed(boundary, 1.0 + i / 1000.0)
            asyncio.run(self.h.cycle(boundary))

            open_trades = self.h.scheduler.open_trades
            newest = max(open_trades, key=lambda t: t.entry_at)
            self.assertEqual(newest.entry_at, boundary,
                             "the trade just signalled enters on this boundary")
            seen.append((newest.entry_at, newest.expiry_at))

            now = self.h.vc.now()
            live = [t for t in open_trades if t.entry_at <= now < t.expiry_at]
            live_counts.append(len(live))
            self.assertLessEqual(len(live), 1, "two trades live at once")

        entries = [entry for entry, _ in seen]
        self.assertEqual(entries, [self.T0 + i * self.PERIOD for i in range(1, 7)],
                         "one entry per bar, each on the boundary")
        self.assertEqual([exp - entry for entry, exp in seen],
                         [self.PERIOD] * 6, "every trade lasts exactly one bar")
        self.assertEqual(entries[1:], [exp for _, exp in seen][:-1],
                         "the next entry is the previous expiry")
        # Nothing precedes the first signal, so the sequence opens with an empty
        # book; after that every signal goes out with exactly one trade live.
        self.assertEqual(live_counts, [0] + [1] * 5,
                         "one trade live at every signal but the first")


class TestASetupJudgedWithAColdTrendEmaIsPassedOver(unittest.TestCase):
    """The trend is a veto, and a veto that is absent is a different engine.

    While the EMA is short there is nothing to veto with, so what gets judged is
    the same engine with its trend filter removed — not the setup .env describes.
    That state is not brief: at a 300s bar it lasts 4h40m of *contiguous* tracking,
    and on 2026-09-16 the candle store held 38 contiguous runs across 21 markets
    with only 3 of them deep enough to warm the gate. So the gate is the
    exception, and the silence it causes has to be legible rather than mysterious.
    """

    ASSET = "AAA_otc"

    def setUp(self):
        self.h = Harness(symbols=(self.ASSET,))
        self.addCleanup(self.h.cleanup)

    def engine(self, missing, trend=None):
        """The engine stubbed out, with a chosen set of cold components."""
        def _fake(candles, config):
            return Signal(direction="CALL", score=2, votes=["FAKE"],
                          price=candles[-1].close, time=candles[-1].time,
                          confidence=0.6, missing=list(missing), trend=trend)
        return unittest.mock.patch.object(sched_mod, "evaluate", _fake)

    def play(self, closes=(99.0, 100.0, 101.0)):
        return asyncio.run(self.h.play(list(closes)))

    def test_a_cold_trend_ema_stops_the_signal(self):
        with self.engine(["trend"]):
            self.play()

        self.assertEqual(self.h.sender.signals, [])
        self.assertEqual(self.h.scheduler.open_trades, [])

    def test_the_same_setup_is_traded_once_the_gate_is_warm(self):
        with self.engine([]):
            self.play()

        self.assertEqual([s["asset"] for s in self.h.sender.signals], [self.ASSET])

    def test_only_the_setup_is_passed_over_not_the_market(self):
        with self.engine(["trend"]):
            self.play()

        self.assertGreater(self.h.market.track(self.ASSET).bar_count, 0,
                           "it keeps ticking and keeps its bars: warming, not "
                           "broken, and the moment the EMA is long enough it is "
                           "judged again")

    def test_the_silence_is_explained_in_the_log(self):
        with self.engine(["trend"]):
            with self.assertLogs("pocket.scheduler", level="INFO") as caught:
                self.play()

        self.assertTrue(
            any("trend EMA is not warm" in line and self.ASSET in line
                for line in caught.output),
            f"why the bot went quiet should be in the log: {caught.output}")

    def test_a_cold_component_that_is_not_a_veto_changes_nothing(self):
        # RSI being short is a scoring problem the engine already weighs; only
        # the veto has to stop the judgement, or every young market is dropped.
        with self.engine(["RSI"]):
            self.play()

        self.assertEqual(len(self.h.sender.signals), 1)

    def test_the_status_says_what_is_being_waited_for(self):
        # A bot that has gone quiet for four hours must not look like a bot that
        # has broken. Bars: n/56 gives the distance; this gives the reason.
        with self.engine(["trend"]):
            self.play()

        self.assertIn("trend EMA", self.h.scheduler.warming_text())
        self.assertIn(self.ASSET, self.h.scheduler.status_text())

    def test_a_warm_cycle_drops_the_waiting_line_again(self):
        # The state is the last cycle's, not a latch: once the EMA is long enough
        # the engine is running as configured and /status must say so.
        with self.engine(["trend"]):
            self.play()
        with self.engine([]):
            self.play()

        self.assertEqual(self.h.scheduler.warming_text(), "")
        self.assertGreater(self.h.scheduler.cold_gate_skips, 0,
                           "the count is cumulative, for the record")

    def test_the_wait_says_how_many_markets_could_end_it(self):
        # "Waiting" on its own reads as a warm-up that is nearly over. This is the
        # half that says whether it is: three bars in, against a gate that needs
        # 56, nothing here can hold the veto — and on the live store it was 3 of
        # 21, which is what made the wait worth questioning rather than sitting
        # through.
        with self.engine(["trend"]):
            self.play()

        text = self.h.scheduler.warming_text()

        self.assertIn("0 of 1 market(s) hold the 56-bar (56m) unbroken run", text)

    def test_the_horizon_reported_is_the_bar_length_it_really_is(self):
        # 4h40m is the figure that explains the live muteness, and it is 56 bars
        # only at a 300s bar. On the 1-minute cadence the same gate is 56m, so a
        # hardcoded horizon would be wrong on one of the two.
        self.h = Harness(symbols=(self.ASSET,),
                         cadence=Cadence(period=300, expiry_seconds=300,
                                         lead_seconds=300))
        self.addCleanup(self.h.cleanup)
        with self.engine(["trend"]):
            self.play()

        self.assertIn("(4h40m)", self.h.scheduler.warming_text())

    def test_the_depth_reported_is_the_configured_gates_own(self):
        # The count is only worth printing if it is about the gate actually
        # configured: a shallower EMA is both a different depth and a different
        # horizon, and both come from the settings rather than from a constant.
        self.h.scheduler.engine_config = SignalConfig(
            use_trend=True, trend_ema_len=1, trend_slope_bars=0)
        with self.engine(["trend"]):
            self.play()

        text = self.h.scheduler.warming_text()

        self.assertIn("1 of 1 market(s) hold the 2-bar (2m) unbroken run", text)

    def test_every_judged_market_is_counted_not_only_the_refused_one(self):
        # Otherwise the line would report the depth of the market that happened
        # to have something to say, which is the one market guaranteed to be the
        # exception rather than the rule.
        self.h = Harness(symbols=(self.ASSET, "BBB_otc"))
        self.addCleanup(self.h.cleanup)
        with self.engine(["trend"]):
            self.play()

        self.assertEqual(self.h.scheduler.gate_markets, 2)
        self.assertIn("0 of 2 market(s)", self.h.scheduler.warming_text())

    def test_the_ready_note_counts_the_markets_it_is_true_of(self):
        # The one message that must not overclaim. Readiness is the leading
        # market's, so a session with one warm market sends "signals active" and
        # then goes on producing nothing — which is what happened on 2026-09-16
        # and is the reason the note now carries the count.
        self.h.scheduler._last_readiness = Readiness(bars=81, bars_needed=56)
        self.h.scheduler.gate_holders = [self.ASSET]
        self.h.scheduler.gate_markets = 21
        self.h.scheduler.gate_needed = 56

        asyncio.run(self.h.scheduler._announce_progress())

        text = " ".join(self.h.sender.texts)
        self.assertIn("1 of 21 market(s)", text)
        self.assertNotIn("Fully warmed up", text)

    def test_with_the_veto_off_there_is_no_census_to_report(self):
        # Every judged market can signal when there is no gate, so a count of
        # "markets past the veto" would be a line about nothing — and a reader
        # would take it as a warning.
        self.h.scheduler.engine_config = SignalConfig(use_trend=False)
        with self.engine([]):
            self.play()

        self.assertIsNone(self.h.scheduler._gate_census())


class TestASignalSaysWhichEngineProducedIt(SchedulerCase):
    """The journal records the gate state and the configuration behind it.

    Without either, "does the trend veto earn its keep?" is unanswerable on live
    data: a signal judged while the EMA was short came from a different strategy
    and afterwards looks exactly like a gated one. The state alone is not enough
    — under ``USE_TREND=0`` the trend is never consulted, so an ungated signal
    and a gated one with a strong reading leave the same trace — which is why the
    dials are written into the same context.
    """

    ASSET = "AAA_otc"

    def setUp(self):
        super().setUp()
        self.path = Path(self.h._tmp.name) / "signals.jsonl"
        self.h.scheduler.journal = SignalJournal(self.path)

    def test_the_context_names_the_cold_gate(self):
        with unittest.mock.patch.object(
                sched_mod, "evaluate",
                lambda candles, config: Signal(
                    direction="CALL", score=2, votes=["FAKE"],
                    price=candles[-1].close, time=candles[-1].time,
                    confidence=0.6, missing=[], trend="up")):
            self.play([99.0, 100.0, 101.0])

        trade = load_journal(self.path).trades[0]
        self.assertEqual(trade.context["trend"], "up")
        self.assertEqual(trade.context["missing"], [])
        self.assertGreater(trade.context["bars"], 0,
                           "how deep the window was is half the story")

    def test_the_context_names_the_configuration_that_judged_it(self):
        with unittest.mock.patch.object(
                sched_mod, "evaluate",
                lambda candles, config: Signal(
                    direction="CALL", score=2, votes=["FAKE"],
                    price=candles[-1].close, time=candles[-1].time,
                    confidence=0.6, missing=[], trend="up")):
            self.play([99.0, 100.0, 101.0])

        scheduler = self.h.scheduler
        trade = load_journal(self.path).trades[0]
        self.assertEqual(trade.config_id,
                         scheduler.engine_config.fingerprint())
        self.assertEqual(trade.context["dials"],
                         scheduler.engine_config.dials())

    def test_a_signal_with_a_cold_gate_is_never_written(self):
        # The skipped setup must leave no trace in the record, or the split
        # cannot answer what the gated engine did — only what it saw.
        with unittest.mock.patch.object(
                sched_mod, "evaluate",
                lambda candles, config: Signal(
                    direction="CALL", score=2, votes=["FAKE"],
                    price=candles[-1].close, time=candles[-1].time,
                    confidence=0.6, missing=["trend"], trend=None)):
            self.play([99.0, 100.0, 101.0])

        self.assertEqual(load_journal(self.path).trades, [])


class TestJournalRecovery(SchedulerCase):
    """A restart must not drop the trade that was open when the last one died.

    Measured on 2026-09-15: a signal sent at 20:15 for the 20:15 entry, the
    process dying at 20:19, and the trade never settled — no result in the
    journal, no outcome in the stats, nothing in any log. The state that would
    have settled it lived only in memory.
    """

    ASSET = "AAA_otc"

    def setUp(self):
        super().setUp()
        self.path = Path(self.h._tmp.name) / "signals.jsonl"
        self.journal = SignalJournal(self.path)
        self.h.scheduler.journal = self.journal

    def journal_a_signal(self, entry_at=BASE + 60, expiry_at=BASE + 120,
                         sent_at=None):
        self.journal.record_signal(
            asset=self.ASSET, direction="CALL", entry_at=entry_at,
            expiry_at=expiry_at, payout=85, score=3, confidence=0.6,
            votes=("RSI", "BB"),
            sent_at=entry_at - 10 if sent_at is None else sent_at)

    def test_a_trade_the_bars_can_still_price_is_adopted_and_settled(self):
        self.journal_a_signal()
        self.h.feed(BASE + 60, 100.0)      # the bar that closed at the entry
        self.h.feed(BASE + 120, 101.0)     # ...and the one that closed at expiry
        self.h.feed(BASE + 180, 102.0)     # which closes only when the next opens
        self.h.vc._now = BASE + 180

        adopted = self.h.scheduler.recover_from_journal()

        self.assertEqual([t.asset for t in adopted], [self.ASSET])
        self.assertTrue(adopted[0].recovered)
        self.assertEqual(adopted[0].entry_price, 100.0,
                         "the price the original session would have used")

        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 240))
        trade = results.resolved[0]
        self.assertEqual((trade.entry_price, trade.exit_price), (100.0, 101.0))
        self.assertEqual(self.h.stats.wins, 1, "the outcome reaches the record")
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], "WIN")
        self.assertIn("after a restart", self.h.sender.confirmations[0]["note"])

        # ...and the journal now has the result line it never got.
        entry = [t for t in load_journal(self.path).trades
                 if t.entry_at == BASE + 60]
        self.assertEqual(len(entry), 1)
        self.assertTrue(entry[0].settled, "the recovered trade is no longer an orphan")
        self.assertEqual(entry[0].our_outcome, "WIN")

    def test_a_trade_whose_bars_are_gone_is_left_unsettled(self):
        # The honest limit: with no bars there is no price, and an outcome
        # invented for it would be indistinguishable from a measured one.
        self.journal_a_signal()
        self.h.vc._now = BASE + 180

        self.assertEqual(self.h.scheduler.recover_from_journal(), [])

        loaded = load_journal(self.path)
        self.assertEqual(len(loaded.unsettled), 1, "still visible as unsettled")
        self.assertEqual(loaded.settled, [])

    def test_a_trade_still_running_is_adopted_before_its_exit_exists(self):
        self.journal_a_signal()
        self.h.feed(BASE + 60, 100.0)
        self.h.feed(BASE + 120, 99.0)       # closes the entry bar
        self.h.vc._now = BASE + 90          # ...while the expiry is still ahead

        adopted = self.h.scheduler.recover_from_journal()

        self.assertEqual(len(adopted), 1)
        self.assertEqual(adopted[0].entry_price, 100.0)
        self.assertIsNone(adopted[0].exit_price, "the exit bar does not exist yet")

        self.h.feed(BASE + 180, 98.0)       # now the exit bar closes too
        self.h.vc._now = BASE + 130
        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 180))

        self.assertEqual(results.resolved[0].exit_price, 99.0)
        self.assertEqual(self.h.stats.losses, 1)

    def test_a_trade_signalled_but_not_yet_entered_is_adopted(self):
        # A restart in the seconds between the message and the entry — the
        # commonest restart there is, since the signal goes out a whole bar
        # before it enters. Its bars are ahead, not missing, and the live feed
        # will produce them.
        self.journal_a_signal(entry_at=BASE + 120, expiry_at=BASE + 180)
        self.h.feed(BASE + 60, 100.0)
        self.h.vc._now = BASE + 90          # the entry is still ahead

        adopted = self.h.scheduler.recover_from_journal()

        self.assertEqual(len(adopted), 1, "a signal ahead of us is not an orphan")
        self.assertIsNone(adopted[0].entry_price, "filled in when the bar closes")

        self.h.feed(BASE + 120, 101.0)      # the entry bar arrives
        self.h.feed(BASE + 180, 102.0)      # ...and the exit bar closes
        self.h.vc._now = BASE + 190
        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 240))

        trade = results.resolved[0]
        self.assertEqual((trade.entry_price, trade.exit_price), (101.0, 102.0))
        self.assertEqual(self.h.stats.wins, 1)

    def test_an_already_settled_signal_is_not_adopted_again(self):
        self.journal_a_signal()
        self.journal.record_result(asset=self.ASSET, entry_at=BASE + 60,
                                   outcome="WIN", entry_price=100.0,
                                   exit_price=101.0)
        self.h.feed(BASE + 60, 100.0)
        self.h.feed(BASE + 120, 101.0)
        self.h.vc._now = BASE + 180

        self.assertEqual(self.h.scheduler.recover_from_journal(), [])
        self.assertEqual(self.h.scheduler.open_trades, [])
        self.assertEqual(self.h.stats.wins, 0, "no second outcome is recorded")

    def test_the_cooldown_survives_the_restart(self):
        # last_signal_at is in memory too. Without it a restart would signal a
        # market the previous process had just signalled.
        self.journal_a_signal(entry_at=BASE + 60, expiry_at=BASE + 120,
                              sent_at=BASE - 15)
        self.h.vc._now = BASE - 15

        self.h.scheduler.recover_from_journal()

        self.assertEqual(self.h.scheduler.last_signal_at[self.ASSET], BASE - 15)
        candidate = Candidate(
            asset=self.ASSET,
            signal=Signal(direction="CALL", score=3, votes=["FAKE"],
                          price=100.0, time=BASE, confidence=0.6))
        self.assertEqual(
            eligible([candidate], self.h.scheduler.last_signal_at, BASE, 120.0), [],
            "the market is still on cooldown")
        self.assertEqual(
            eligible([candidate], self.h.scheduler.last_signal_at, BASE + 200, 120.0),
            [candidate], "...and comes off it on schedule")

    def test_without_a_journal_there_is_nothing_to_recover(self):
        self.h.scheduler.journal = None
        self.assertEqual(self.h.scheduler.recover_from_journal(), [])


class TestThePayoutFloorIsEnforcedAtTheSignal(unittest.TestCase):
    """A market paying too little is passed over, not thrown out.

    The floor was applied once, when the universe was chosen, and the universe
    is sticky by design — so a market picked at 92% could fall to anything and
    still be traded. Observed on 2026-09-15: a CADJPY signal at 38%, which
    needs 72% accuracy to break even, and lost.

    Dropping the market from the universe instead would have been the wrong
    repair: that churn is what truncated the stored series and left signals
    with no bars to settle against. It keeps its place and its bars, and only
    the trade is withheld.
    """

    RICH, POOR = "RICH_otc", "POOR_otc"

    def setUp(self):
        self.h = Harness(symbols=(self.RICH, self.POOR),
                         payouts={self.RICH: 92, self.POOR: 38},
                         min_payout=60)
        self.addCleanup(self.h.cleanup)
        patch = unittest.mock.patch.object(sched_mod, "evaluate",
                                           fake_evaluate("CALL"))
        patch.start()
        self.addCleanup(patch.stop)

    def play(self, closes, start=None):
        return asyncio.run(self.h.play(closes, start))

    def test_a_market_paying_exactly_the_floor_is_traded(self):
        # The floor is "at least", not "more than" — a live NZDUSD signal went
        # out at exactly 60% against MIN_PAYOUT=60. Pinned because the two
        # differ by one character and the wrong one would quietly drop every
        # market sitting on the floor.
        h = Harness(symbols=("EDGE_otc",), payouts={"EDGE_otc": 60}, min_payout=60)
        self.addCleanup(h.cleanup)

        asyncio.run(h.play([99.0, 100.0, 101.0]))

        self.assertEqual([s["asset"] for s in h.sender.signals], ["EDGE_otc"])

    def test_the_market_under_the_floor_is_not_signalled(self):
        results = self.play([99.0, 100.0, 101.0])

        self.assertTrue(any(r.chosen for r in results), "the rich market should fire")
        self.assertEqual([s["asset"] for s in self.h.sender.signals], [self.RICH])
        self.assertEqual([t.asset for t in self.h.scheduler.open_trades], [self.RICH])

    def test_the_signal_carries_the_payout_it_was_judged_on(self):
        self.play([99.0, 100.0, 101.0])

        self.assertEqual(self.h.sender.signals[0]["payout"], 92)

    def test_the_passed_over_market_keeps_its_bars(self):
        self.play([99.0, 100.0, 101.0])

        poor = self.h.market.track(self.POOR).bar_count
        self.assertGreater(poor, 0, "it is still subscribed and still ticking")
        self.assertEqual(poor, self.h.market.track(self.RICH).bar_count,
                         "tracked exactly like a market it would trade")

    def test_the_setup_that_was_passed_over_is_named(self):
        with self.assertLogs("pocket.scheduler", level="INFO") as caught:
            self.play([99.0, 100.0, 101.0])

        self.assertTrue(
            any(f"{self.POOR} 38%" in line and "under 60%" in line
                for line in caught.output),
            f"the reason for the silence should be in the log: {caught.output}")

    def test_a_market_with_nothing_to_say_is_not_named(self):
        # The line is about a trade not taken, so it must not list every cheap
        # market on every bar — only the ones that had a setup to take.
        self.h.scheduler.payouts[self.POOR] = 10
        self.h.feed(int(BASE), 100.0)              # one bar: no setup on either
        said: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: said.append(record.getMessage())
        logger = logging.getLogger("pocket.scheduler")
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)

        candidates = self.h.scheduler._collect([self.POOR], int(BASE))

        self.assertEqual(candidates, [], "one bar is not a setup")
        self.assertEqual([line for line in said if self.POOR in line], [],
                         "nothing was passed over, so nothing is reported")

    def test_an_unknown_payout_is_not_read_as_a_low_one(self):
        # The map is rebuilt per session from what the feed last published; an
        # empty one must not silence every market at once.
        h = Harness(symbols=(self.RICH,), payouts={}, min_payout=60)
        self.addCleanup(h.cleanup)

        asyncio.run(h.play([99.0, 100.0, 101.0]))

        self.assertEqual([s["asset"] for s in h.sender.signals], [self.RICH],
                         "a payout of 0 means unknown, not worthless")

    def test_a_floor_of_zero_disables_the_check(self):
        h = Harness(symbols=(self.POOR,), payouts={self.POOR: 5}, min_payout=0)
        self.addCleanup(h.cleanup)

        asyncio.run(h.play([99.0, 100.0, 101.0]))

        self.assertEqual([s["asset"] for s in h.sender.signals], [self.POOR])

    def test_a_market_alone_under_the_floor_sends_nothing_at_all(self):
        # The decisive case, and the one a two-market test cannot prove: with
        # only the cheap market there is no winner to fall back on, so a signal
        # here would mean the floor is not being applied to the setup at all.
        h = Harness(symbols=(self.POOR,), payouts={self.POOR: 38}, min_payout=60)
        self.addCleanup(h.cleanup)

        with self.assertLogs("pocket.scheduler", level="INFO") as caught:
            asyncio.run(h.play([99.0, 100.0, 101.0]))

        self.assertTrue(any(f"{self.POOR} 38%" in line for line in caught.output),
                        f"the setup existed — the fake engine signals on any "
                        f"bars — and was passed over: {caught.output}")
        self.assertEqual(h.sender.signals, [], "and was still not traded")
        self.assertEqual(h.scheduler.open_trades, [])


class TestATradeWithNoBarsIsAbandonedNotInvented(SchedulerCase):
    """The fallback to the latest tick is bounded, because it can lie.

    Measured live on 2026-09-15: EURNZD_otc entered at 21:05 was adopted at
    23:42 from the restored bars, those bars were dropped two seconds later when
    the live feed resumed through a 135-minute gap, and at 23:45 the trade
    "settled" against the 23:44 price — a loss nobody measured, in the stats and
    in the journal, and one that moved the expected value by 2.5 points.
    """

    ASSET = "AAA_otc"

    def journal_for(self, entry_at, expiry_at):
        journal = SignalJournal(Path(self.h.tmpdir) / "signals.jsonl")
        journal.record_signal(asset=self.ASSET, direction="CALL",
                              entry_at=entry_at, expiry_at=expiry_at, payout=85)
        self.h.scheduler.journal = journal
        return journal

    def open_a_trade(self, entry_at, expiry_at, entry_price=100.0):
        trade = OpenTrade(asset=self.ASSET, direction="CALL", entry_at=entry_at,
                          expiry_at=expiry_at, entry_price=entry_price,
                          recovered=True)
        self.h.scheduler.open_trades.append(trade)
        return trade

    def test_a_bar_that_is_a_little_late_is_still_settled(self):
        # One bar late: the latest tick is a late reading of the expiry, which
        # is the case the fallback exists for.
        self.h.feed(BASE, 100.0)
        self.h.feed(BASE + 120, 101.0)     # the expiry bucket never landed
        self.open_a_trade(entry_at=BASE, expiry_at=BASE + 60)
        self.h.vc._now = BASE + 120

        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 120))

        self.assertEqual(len(results.resolved), 1)
        self.assertEqual(results.resolved[0].exit_price, 101.0)
        self.assertEqual(self.h.stats.wins, 1)

    def test_a_trade_whose_bars_are_hours_gone_is_left_unsettled(self):
        journal = self.journal_for(BASE, BASE + 60)
        self.h.feed(BASE, 100.0)
        self.h.feed(BASE + 120, 99.0)
        self.open_a_trade(entry_at=BASE, expiry_at=BASE + 60)
        self.h.vc._now = BASE + 60 * 4    # three bars past the expiry

        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 60 * 4))

        self.assertEqual(results.resolved, [], "nothing was settled")
        self.assertEqual(self.h.stats.total, 0, "and nothing reached the record")
        self.assertEqual(self.h.stats.consecutive_losses(), 0,
                         "no invented loss, so no streak toward the breaker")
        self.assertEqual(self.h.scheduler.open_trades, [], "not retried forever")
        loaded = load_journal(journal.path)
        self.assertEqual(loaded.settled, [])
        self.assertEqual(len(loaded.unsettled), 1, "still visible as unsettled")

    def test_the_abandoned_trade_is_named_in_the_log(self):
        self.h.feed(BASE, 100.0)
        self.h.feed(BASE + 120, 99.0)
        self.open_a_trade(entry_at=BASE, expiry_at=BASE + 60)
        self.h.vc._now = BASE + 60 * 4

        with self.assertLogs("pocket.scheduler", level="WARNING") as caught:
            asyncio.run(self.h.scheduler.run_cycle(BASE + 60 * 4))

        self.assertTrue(any("left unsettled rather than settled against a price "
                            "from after the expiry" in line
                            for line in caught.output), caught.output)

    def test_an_entry_bar_that_is_gone_does_not_become_a_tie(self):
        # Pricing the trade off the exit bar alone would make entry == exit,
        # and a tie settles as a loss. Silence is the honest answer.
        journal = self.journal_for(BASE + 60, BASE + 120)
        self.h.feed(BASE + 120, 101.0)     # only the exit bucket exists
        self.open_a_trade(entry_at=BASE + 60, expiry_at=BASE + 120,
                          entry_price=None)
        self.h.vc._now = BASE + 60 * 5

        results = asyncio.run(self.h.scheduler.run_cycle(BASE + 60 * 5))

        self.assertEqual(results.resolved, [])
        self.assertEqual(self.h.stats.total, 0)
        self.assertEqual(load_journal(journal.path).settled, [])

    def test_a_fresh_trade_is_never_abandoned(self):
        # The window must not swallow the ordinary case: the exit bar arrives
        # while the trade is still young.
        self.h.feed(BASE, 100.0)
        self.open_a_trade(entry_at=BASE, expiry_at=BASE + 60)
        self.h.vc._now = BASE + 30

        asyncio.run(self.h.scheduler.run_cycle(BASE + 60))

        self.assertEqual(len(self.h.scheduler.open_trades), 1,
                         "still open, waiting for its exit bar")


class TestTheStatusLineStatesWhatTheRecordIsWorth(SchedulerCase):
    """/status shows the bot's state; it has to show the result too.

    A win rate alone cannot be judged when the payout moves — so the status
    carries the per-trade return and its interval, read from the journal at the
    moment the command is typed.
    """

    def journalled(self, outcomes, payout=92):
        """A journal file on disk, as the bot writes it: one signal, one result."""
        path = Path(self.h.tmpdir) / "signals.jsonl"
        journal = SignalJournal(path)
        for i, outcome in enumerate(outcomes):
            entry_at = BASE + i * 60
            journal.record_signal(asset="AAA_otc", direction="CALL",
                                  entry_at=entry_at, expiry_at=entry_at + 60,
                                  payout=payout)
            journal.record_result(asset="AAA_otc", entry_at=entry_at,
                                  outcome=outcome, entry_price=1.0, exit_price=1.1)
        return journal

    def test_the_status_carries_the_return_per_trade(self):
        journal = self.journalled(["WIN", "LOSS", "LOSS"])
        self.h = Harness(journal=journal)
        self.addCleanup(self.h.cleanup)

        text = self.h.scheduler.status_text()

        self.assertIn("Status", text, "the state of the bot is still there")
        self.assertIn("EV", text)
        self.assertIn("n=3", text)
        self.assertIn("-0.360 per trade", text, "(0.92 - 1 - 1) / 3")

    def test_a_win_with_no_payout_on_record_is_named_not_guessed(self):
        # Written the way an older build did, before the payout was journalled.
        path = Path(self.h.tmpdir) / "signals.jsonl"
        journal = SignalJournal(path)
        entry_at = BASE
        journal.record_signal(asset="AAA_otc", direction="CALL", entry_at=entry_at,
                              expiry_at=entry_at + 60)
        journal.record_result(asset="AAA_otc", entry_at=entry_at, outcome="WIN",
                              entry_price=1.0, exit_price=1.1)
        self.h = Harness(journal=journal)
        self.addCleanup(self.h.cleanup)

        text = self.h.scheduler.status_text()

        self.assertIn("unpriced", text)
        self.assertNotIn("per trade", text, "nothing priced, so no average")

    def test_a_signal_that_never_settled_is_named_in_the_status(self):
        # The live case: 4 of 40 signals sit in the journal with a signal line and
        # no result line, because their bars fell out of the store before they could
        # be settled. They will never settle now, so the status is the only place a
        # reader can find out why n is smaller than the number of signals sent.
        journal = self.journalled(["WIN", "LOSS"])
        entry_at = BASE + 2 * 60
        journal.record_signal(asset="BBB_otc", direction="CALL", entry_at=entry_at,
                              expiry_at=entry_at + 60, payout=92)
        self.h = Harness(journal=journal)
        self.addCleanup(self.h.cleanup)

        text = self.h.scheduler.status_text()

        self.assertIn("n=2", text)
        self.assertIn("1 never settled", text)

    def test_a_status_command_survives_a_journal_it_cannot_read(self):
        path = Path(self.h.tmpdir) / "signals.jsonl"
        path.write_text("not json at all\n{}\n", encoding="utf-8")
        self.h = Harness(journal=SignalJournal(path))
        self.addCleanup(self.h.cleanup)

        text = self.h.scheduler.status_text()

        self.assertIn("Status", text)
        self.assertNotIn("EV", text)

    def test_no_journal_means_no_record_line(self):
        self.h = Harness()
        self.addCleanup(self.h.cleanup)

        self.assertNotIn("EV", self.h.scheduler.status_text())


class TestTheStatusSaysWhoseRecordThisIs(SchedulerCase):
    """An EV is a number about a *strategy*, and /status has to name it.

    The failure this guards against is quiet by construction: an EV measured
    under another engine is arithmetically identical to one measured under this
    one, so nothing about the number looks wrong. The record was read for hours
    as if it described the running strategy when it described the ungated one,
    which is why the line carries the answer rather than leaving it to be
    inferred from a settings file.
    """

    def record(self, pairs, **config):
        """A journal as the bot writes it: (config_id, outcome) per settled trade."""
        path = Path(self.h.tmpdir) / "signals.jsonl"
        journal = SignalJournal(path)
        for i, (config_id, outcome) in enumerate(pairs):
            entry_at = BASE + i * 60
            journal.record_signal(
                asset="AAA_otc", direction="CALL", entry_at=entry_at,
                expiry_at=entry_at + 60, payout=92,
                context={"config_id": config_id} if config_id else None)
            journal.record_result(asset="AAA_otc", entry_at=entry_at,
                                  outcome=outcome, entry_price=1.0,
                                  exit_price=1.1)
        self.h = Harness(journal=journal, config=SignalConfig(**config))
        self.addCleanup(self.h.cleanup)
        return self.h.scheduler

    def test_a_record_from_the_running_engine_needs_no_note(self):
        mine = SignalConfig(use_trend=False).fingerprint()
        sched = self.record([(mine, "WIN")], use_trend=False)

        text = sched.expected_value_text()

        self.assertIn("EV", text)
        self.assertNotIn("not this engine's", text)

    def test_a_mixed_record_that_includes_this_engine_is_still_this_engines(self):
        # The average is over more than one strategy and the journal says so,
        # but this engine is among them — which is the question /status asks.
        mine = SignalConfig(use_trend=False).fingerprint()
        sched = self.record([(mine, "WIN"), ("0therc0de", "LOSS")],
                            use_trend=False)

        text = sched.expected_value_text()

        self.assertIn("2 configurations mixed", text)
        self.assertNotIn("not this engine's", text)

    def test_another_engines_record_says_so_and_names_the_running_one(self):
        sched = self.record([("0therc0de", "WIN")], use_trend=False)

        text = sched.expected_value_text()

        self.assertIn("not this engine's", text)
        self.assertIn("use_trend=False", text)

    def test_it_counts_the_settled_trades_it_is_not_describing(self):
        sched = self.record([("0therc0de", "WIN"), ("0therc0de", "LOSS"),
                             ("0therc0de", "WIN")], use_trend=False)

        text = sched.expected_value_text()

        self.assertIn("3 settled trade(s) from 1 other configuration(s)", text)

    def test_a_record_from_before_the_field_existed_is_its_own_answer(self):
        # No config_id is not a missing value: it says which build wrote the
        # line, so it answers the question rather than deferring it.
        sched = self.record([("", "WIN"), ("", "LOSS")], use_trend=False)

        text = sched.expected_value_text()

        self.assertIn("none of its 2 settled trade(s) recorded a configuration",
                      text)

    def test_an_empty_record_gets_no_note(self):
        sched = self.record([])

        self.assertEqual(sched.expected_value_text(), "")


if __name__ == "__main__":
    unittest.main()
