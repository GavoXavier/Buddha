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
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import engine.scheduler as sched_mod
from engine.scheduler import Cadence, MinuteScheduler, next_boundary, outcome_of
from journal import SignalJournal, load_journal
from market.aggregator import MarketState
from market.clock import VirtualClock
from signals.engine import Signal, SignalConfig
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
                             "price": entry_price})
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
                 config=None):
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
            payouts={s: 85 for s in self.symbols}, clock=self.vc)

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


if __name__ == "__main__":
    unittest.main()
