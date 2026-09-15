"""Tests for automated execution as the minute loop actually uses it.

``execution`` has its own tests and the scheduler has its own; this module covers
the seam between them, which is where the mistakes that matter live:

* the order must be for the signal's **own** boundary, not for whenever the
  message was read — otherwise the entry price is not the one the signal was
  judged on and nothing measured afterwards means anything;
* the broker's settlement must become the outcome of record, because it is the
  money, while the bot's own bar label is kept beside it — if the label were
  dropped, label agreement could never be measured again;
* a refund and an order that was never placed must not be counted as losses.

The broker here is a stub, on purpose. What is under test is what the loop does
with an answer — a win, a loss, a refund, no answer, or an order that failed —
and a stub is the only way to produce each of those on demand without a socket.
"""

import asyncio
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import engine.scheduler as sched_mod
from engine.scheduler import Cadence, MinuteScheduler
from execution import LOSS, PUSH, UNPLACED, WIN, Order, Settlement, order_key
from journal import SignalJournal, load_journal
from market.aggregator import MarketState
from market.clock import VirtualClock
from signals.engine import Signal, SignalConfig
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


class StubBroker:
    """A broker that answers whatever the test tells it to.

    Records the orders it was given and the lookups it was asked to resolve, so
    a test can check the loop asked about the trade it actually placed.
    """

    def __init__(self, *, amount: float = 1.0, enabled: bool = True,
                 any_result=None, failure: str = "",
                 summary: str = "auto-trade ON (demo, 1/trade, 60s)") -> None:
        self.amount = amount
        self.enabled = enabled
        self.any_result = any_result
        self.failure = failure
        self._summary = summary
        self.submitted: list[Order] = []
        self.results: dict[str, Settlement] = {}
        self.lookups: list[str] = []

    def settle(self, key: str, outcome: str = WIN, *, profit: float = 0.85) -> None:
        """Arm the answer for one specific trade."""
        self.results[key] = Settlement(outcome=outcome, profit=profit)

    def submit(self, order: Order) -> None:
        self.submitted.append(order)
        if self.failure:
            order.error = self.failure

    def settlement_for(self, asset: str, entry_at: float):
        key = order_key(asset, entry_at)
        self.lookups.append(key)
        return self.results.get(key, self.any_result)

    def summary(self) -> str:
        return self._summary


class Harness:
    """A market, a scheduler wired to a stub broker, and a minute replay.

    ``play`` continues from where the previous call stopped, so a test can let a
    signal fire, decide how the broker answered, and then replay the bars that
    settle it — which is the order these things happen in for real.

    The rhythm, with two bars before the first signal and two more before the
    first resolution (period 60s, lead 10s, cooldown 120s):

        bar 1   nothing to judge yet
        bar 2   signal, entry on this boundary      -> order submitted
        bar 3   the entry bar closes, price exact; still open
        bar 4   the exit bar closes                 -> result sent
    """

    def __init__(self, *, direction: str = "CALL", cadence=None,
                 broker=None, symbol: str = "AAA_otc") -> None:
        self.vc = VirtualClock(start=BASE)
        self.market = MarketState(PERIOD, 200, 120, self.vc)
        self.sender = FakeSender()
        self.controller = BotController()
        self.cadence = cadence or Cadence()
        self.symbol = symbol
        self.symbols = [symbol]
        self.broker = broker
        self.direction = direction
        self._cursor = None
        self._tmp = tempfile.TemporaryDirectory()
        self.journal_path = Path(self._tmp.name) / "signals.jsonl"
        self.stats = StatsTracker(str(Path(self._tmp.name) / "stats.json"), None)
        self.scheduler = MinuteScheduler(
            market=self.market, sender=self.sender, stats=self.stats,
            controller=self.controller, cadence=self.cadence,
            engine_config=SignalConfig(), symbols=self.symbols,
            payouts={symbol: 85}, clock=self.vc, broker=broker,
            journal=SignalJournal(self.journal_path, True))

    def feed(self, end: float, close: float) -> None:
        """Replay the bar that *ends* at ``end`` (open ``end``-60, close ``close``).

        Every tick is at ``close``, so a bar's open and close are the same price
        and the entry/exit prices a test asserts on are exactly the closes it
        passed in.
        """
        open_at = end - PERIOD
        self.vc._now = open_at
        series = self.market.track(self.symbol)
        series.add_tick(open_at, close)
        series.add_tick(open_at + 10, close)
        series.add_tick(open_at + 59, close)

    async def bars(self, closes) -> list:
        """Replay the next bars, continuing from the last call."""
        if self._cursor is None:
            self._cursor = int(BASE - PERIOD * len(closes))
        results = []
        for close in closes:
            boundary = self._cursor
            self.feed(boundary, close)
            self.vc._now = boundary - self.cadence.lead_seconds
            results.append(await self.scheduler.run_cycle(boundary))
            self._cursor += PERIOD
        return results

    def journal_entries(self):
        return load_journal(self.journal_path)

    def cleanup(self) -> None:
        self._tmp.cleanup()


class AutotradeCase(unittest.TestCase):
    """Shared fixture: the engine stubbed to CALL, a stub broker in place."""

    direction = "CALL"

    def setUp(self):
        self.broker = StubBroker()
        self.h = self.harness(self.broker)
        patch = unittest.mock.patch.object(sched_mod, "evaluate",
                                           fake_evaluate(self.direction))
        patch.start()
        self.addCleanup(patch.stop)

    def harness(self, broker) -> Harness:
        h = Harness(direction=self.direction, broker=broker)
        self.addCleanup(h.cleanup)
        return h

    def play(self, closes):
        return asyncio.run(self.h.bars(closes))

    def entries(self):
        """The journal, with the first settled trade's record."""
        loaded = self.h.journal_entries()
        settled = [t for t in loaded.trades if t.settled]
        self.assertTrue(settled, "no trade settled — the replay did not resolve one")
        return loaded


class TestOrderSubmission(AutotradeCase):
    def test_the_order_is_for_the_signal_s_own_boundary(self):
        """Not for whenever the message was read: the measurement depends on it."""
        self.play([99.0, 100.0, 101.0, 102.0])

        sent = self.h.sender.signals[0]
        order = self.h.broker.submitted[0]

        self.assertEqual(order.entry_at, sent["entry_at"])
        self.assertEqual(order.entry_at % PERIOD, 0, "the entry lands on a boundary")
        self.assertEqual(order.asset, self.h.symbol)
        self.assertEqual(order.direction, self.direction)
        self.assertEqual(order.expiry_at, order.entry_at + self.h.cadence.expiry_seconds)

    def test_the_order_carries_the_broker_s_amount_and_the_cadence_s_expiry(self):
        self.play([99.0, 100.0, 101.0, 102.0])

        order = self.h.broker.submitted[0]

        self.assertAlmostEqual(order.amount, self.broker.amount)
        self.assertEqual(order.duration, self.h.cadence.expiry_seconds)

    def test_every_signal_gets_exactly_one_order(self):
        # Two signals over four bars at this cadence, and no duplicates.
        self.play([99.0, 100.0, 101.0, 102.0])

        self.assertEqual(len(self.h.broker.submitted), len(self.h.sender.signals))
        keys = [o.key for o in self.h.broker.submitted]
        self.assertEqual(len(keys), len(set(keys)), "one order per boundary")

    def test_nothing_is_submitted_when_the_bot_only_signals(self):
        # broker=None is the default build: AUTO_TRADE off means no order exists.
        h = self.harness(None)
        asyncio.run(h.bars([99.0, 100.0, 101.0, 102.0]))

        self.assertTrue(h.sender.signals, "the signal must still go out")
        self.assertIsNone(h.scheduler.open_trades[0].order)

    def test_a_refused_broker_leaves_the_signals_alone(self):
        """verify() said the account is not demo: signal, do not trade."""
        broker = StubBroker(enabled=False)
        h = self.harness(broker)
        asyncio.run(h.bars([99.0, 100.0, 101.0, 102.0]))

        self.assertEqual(broker.submitted, [])
        self.assertTrue(h.sender.signals, "signals keep going")
        # With no order there is nothing to confirm, so the bar label stands —
        # and the journal records no broker block rather than an invented one.
        self.assertEqual(h.sender.confirmations[0]["outcome"], "WIN")
        self.assertFalse(h.journal_entries().trades[0].has_broker)


class TestTheBrokerOutranksTheBars(AutotradeCase):
    def test_the_broker_s_win_is_what_is_counted_and_reported(self):
        """Bars say LOSS, the account says WIN. The money happened."""
        # closes[1] is the entry price and closes[2] the exit, so a CALL from
        # 100.0 to 99.0 is a loss on the bars.
        self.play([99.0, 100.0])
        self.broker.settle(self.broker.submitted[0].key, WIN, profit=0.85)
        self.play([99.0, 99.0])

        self.assertEqual(self.h.stats.wins, 1)
        self.assertEqual(self.h.stats.losses, 0)
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], WIN)

    def test_the_journal_keeps_both_answers(self):
        # Our label is the claim; the broker's is the fact. Dropping either one
        # makes label agreement unmeasurable.
        self.play([99.0, 100.0])
        self.broker.settle(self.broker.submitted[0].key, WIN, profit=0.85)
        self.play([99.0, 99.0])

        trade = self.entries().trades[0]

        self.assertEqual(trade.our_outcome, LOSS, "our reading of the two closes")
        self.assertEqual(trade.broker_outcome, WIN, "the account's verdict")
        self.assertEqual(trade.outcome_of_record, WIN, "the money is the record")
        self.assertAlmostEqual(trade.broker_profit, 0.85)

    def test_the_settlement_is_looked_up_by_the_trade_it_belongs_to(self):
        # A settlement asked for under the wrong key would silently never match.
        self.play([99.0, 100.0])
        order = self.broker.submitted[0]
        self.broker.settle(order.key, WIN)
        self.play([99.0, 99.0])

        self.assertEqual(self.broker.lookups, [order.key])
        self.assertEqual(order.key, order_key(self.h.symbol, order.entry_at))

    def test_our_label_is_still_recorded_when_the_broker_never_answers(self):
        """No settlement is not a result: fall back to the bars, and say so."""
        self.play([99.0, 100.0, 99.0, 99.0])

        self.assertEqual(self.h.stats.losses, 1, "the bar label is used")
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], LOSS)
        trade = self.entries().trades[0]
        self.assertFalse(trade.has_broker, "no broker block was invented")
        self.assertEqual(trade.outcome_of_record, LOSS)


class TestWhatMustNotBeCounted(AutotradeCase):
    def test_a_refund_is_reported_as_a_refund_and_not_counted(self):
        self.broker.any_result = Settlement(outcome=PUSH, profit=0.0)
        self.play([99.0, 100.0, 101.0, 102.0])

        self.assertEqual(self.h.stats.total, 0, "neither a win nor a loss")
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], PUSH)
        self.assertEqual(self.entries().broker_outcomes(), {PUSH: 1})

    def test_an_order_that_was_never_placed_is_not_a_loss(self):
        self.broker.failure = "RuntimeError: min_amount is 1"
        self.play([99.0, 100.0, 101.0, 102.0])

        self.assertEqual(self.h.stats.total, 0)
        self.assertEqual(self.h.sender.confirmations[0]["outcome"], UNPLACED)
        trade = self.entries().trades[0]
        self.assertEqual(trade.broker_outcome, UNPLACED)
        self.assertEqual(trade.our_outcome, "WIN", "our label is still recorded")

    def test_a_refund_does_not_dilute_the_win_rate(self):
        """One win and one refund: 100%, not 50% — the refund is not a loss."""
        self.play([99.0, 100.0])
        self.broker.settle(self.broker.submitted[0].key, WIN, profit=0.85)
        self.play([101.0, 101.0])
        second = self.broker.submitted[1]
        self.broker.settle(second.key, PUSH, profit=0.0)
        self.play([101.0, 101.0])

        loaded = self.entries()

        self.assertEqual(self.h.stats.total, 1, "only the decided trade counts")
        self.assertAlmostEqual(self.h.stats.win_rate, 1.0)
        self.assertEqual(loaded.broker_outcomes(), {WIN: 1, PUSH: 1})
        self.assertAlmostEqual(loaded.broker_win_rate(), 1.0)
        # The refunded trade closed flat, which our own label calls a loss — so
        # this is also a trade where the two sides genuinely disagree.
        refunded = [t for t in loaded.trades if t.broker_outcome == PUSH][0]
        self.assertEqual(refunded.our_outcome, LOSS)
        self.assertEqual(loaded.agreements(), (1, 1), "the refund is not comparable")


class TestWhatTheOperatorSees(AutotradeCase):
    def test_the_startup_message_says_whether_the_account_is_being_traded(self):
        # "Is this thing trading my account?" should not need a config file.
        asyncio.run(self.h.scheduler._announce_startup())

        self.assertIn("auto-trade ON", self.h.sender.texts[0])

    def test_the_startup_message_has_no_execution_line_without_a_broker(self):
        h = self.harness(None)
        asyncio.run(h.scheduler._announce_startup())

        self.assertNotIn("auto-trade", h.sender.texts[0])


if __name__ == "__main__":
    unittest.main()
