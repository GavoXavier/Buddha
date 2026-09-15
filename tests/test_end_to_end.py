"""End-to-end run: the whole bot on a virtual clock, offline.

This is the test that says the parts add up. A simulated feed pushes ticks, the
aggregator builds the bars, the engine scores them, the scheduler sends at most
one signal a minute, Telegram-shaped messages come out, and the results are
recorded and persisted — all driven by a ``VirtualClock``, so several hours of
market replay in a couple of seconds.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from config import Config
from data import SimulatedFeed
from data.base import DataFeed
from engine.scheduler import MinuteScheduler, next_boundary
from journal import load_journal
from main import FeedStalled, trading_session, watch_feed
from market.aggregator import MarketState
from market.clock import VirtualClock
from market.store import CandleStore
from signals.engine import SignalConfig
from stats import StatsTracker
from telegram.control import BotController

BASE = 1_700_000_040.0          # exactly on a minute boundary
PERIOD = 60
SYMBOLS = ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDUSD_otc", "USDCAD_otc"]

# The engine settings the bot ships with (see .env): two directional indicators
# in agreement, a 5x confirmation on top. The cadence here is one minute rather
# than the shipped five so a replay covers many trades in a few virtual hours;
# the 5-minute loop is covered in tests/test_scheduler.py.
ENGINE = SignalConfig(
    bar_seconds=PERIOD,
    use_trend=True, trend_ema_len=50, trend_slope_bars=5,
    trend_min_distance_pct=0.05, trend_flat_min_score=3,
    use_mtf=True, mtf_factor=5, mtf_ema_len=10,
    use_atr=True, atr_len=14,
    use_rsi=True, use_macd=True, use_bb=True, use_stoch=True, use_sr=True,
    min_score=2, min_components=2,
)


class RecordingSender:
    """Stands in for Telegram: records what would have been sent."""

    def __init__(self):
        self.signals = []
        self.confirmations = []
        self.texts = []

    async def send_signal(self, signal, asset, expiry, entry_at, now=None,
                          payout=0, martingale_steps=0, entry_price=None):
        self.signals.append({"asset": asset, "direction": signal.direction,
                             "expiry": expiry, "entry_at": entry_at,
                             "price": entry_price, "score": signal.score,
                             "confidence": signal.confidence})
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


class EndToEndCase(unittest.TestCase):
    """Replays one market session for the whole class, then inspects it.

    The replay is deterministic (fixed feed seed, virtual clock) and read-only
    from the tests' point of view, so every test in a class can share the run.
    """

    minutes = 240        # virtual minutes of market to replay

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.out = cls.replay()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def config(cls) -> Config:
        return Config(
            feed="simulated", assets=list(SYMBOLS),
            candle_period=PERIOD, max_bars=500, stale_after=120.0,
            feed_stale_after=90.0,
            candle_store_dir=str(cls.dir / "candles"), persist_candles=True,
            expiry="1m", expiry_seconds=60, lead_seconds=10,
            cooldown_seconds=120, max_consecutive_losses=5, breaker_pause_minutes=15,
            stats_path=str(cls.dir / "stats.json"), legacy_stats_path="",
            signal_journal_path=str(cls.dir / "signals.jsonl"), journal_signals=True,
            signal=ENGINE)

    @classmethod
    def replay(cls) -> dict:
        """Replay the market once and return everything the bot did."""
        cfg = cls.config()
        sender = RecordingSender()
        stats = StatsTracker(str(cls.dir / "stats.json"), None)
        out: dict = {}

        async def scenario():
            clock = VirtualClock(start=BASE)
            feed = SimulatedFeed(SYMBOLS, clock=clock)
            await feed.connect()
            controller = BotController()
            holder: dict = {"scheduler": None}
            session = asyncio.create_task(trading_session(
                cfg, feed, sender, stats, controller, holder, clock))

            await clock.advance(float(cls.minutes * PERIOD),
                                steps=cls.minutes * 2)
            scheduler = holder["scheduler"]
            assert isinstance(scheduler, MinuteScheduler), "session never started"
            out["scheduler"] = scheduler
            out["open_at_stop"] = list(scheduler.open_trades)
            out["signals_sent"] = scheduler.signals_sent

            controller.request_stop()
            await clock.advance(180.0, steps=12)
            await asyncio.wait_for(session, timeout=60.0)

        asyncio.run(scenario())
        out["sender"] = sender
        out["stats"] = stats
        return out

    @property
    def signals(self):
        return self.out["sender"].signals

    @property
    def confirmations(self):
        return self.out["sender"].confirmations

    def bars_on_disk(self) -> list[Path]:
        directory = self.dir / "candles"
        return sorted(directory.glob("*.json")) if directory.exists() else []


class TestSignalCadence(EndToEndCase):
    """The headline requirement: signals on the minute, for one-minute trades."""

    def test_the_run_produces_signals(self):
        self.assertGreaterEqual(len(self.signals), 1,
                                "240 minutes across 5 markets should find a setup")

    def test_every_signal_enters_on_a_minute_boundary(self):
        for signal in self.signals:
            entry_at = signal["entry_at"]
            self.assertEqual(entry_at % PERIOD, 0,
                             f"{signal['asset']} entered at {entry_at}, off boundary")
            self.assertEqual(next_boundary(entry_at - 0.5, PERIOD), entry_at,
                             "the entry is the very next boundary")

    def test_expiry_is_one_minute(self):
        for signal in self.signals:
            self.assertEqual(signal["expiry"], "1m")
        for confirmation in self.confirmations:
            self.assertEqual(confirmation["expiry_at"] % PERIOD, 0)

    def test_at_most_one_signal_per_minute_and_never_the_same_market_twice_over(self):
        boundaries = [s["entry_at"] for s in self.signals]
        self.assertEqual(len(boundaries), len(set(boundaries)),
                         "one signal per minute at most")
        self.assertEqual(boundaries, sorted(boundaries), "signals come out in order")

        # The 120s cooldown means a market cannot fire on consecutive minutes.
        by_asset: dict[str, list[float]] = {}
        for signal in self.signals:
            by_asset.setdefault(signal["asset"], []).append(signal["entry_at"])
        for asset, entries in by_asset.items():
            gaps = [b - a for a, b in zip(entries, entries[1:])]
            self.assertTrue(all(gap >= 120 for gap in gaps),
                            f"{asset} signalled twice inside the cooldown: {gaps}")

    def test_every_signal_is_a_real_setup_with_a_price_and_a_direction(self):
        for signal in self.signals:
            self.assertIn(signal["direction"], ("CALL", "PUT"))
            self.assertIn(signal["asset"], SYMBOLS)
            self.assertGreater(signal["price"], 0)
            self.assertGreaterEqual(signal["score"], 2)
            self.assertGreater(signal["confidence"], 0)


class TestAccounting(EndToEndCase):
    """What the bot reports must match what it sent."""

    def test_every_signal_is_settled_or_still_open(self):
        self.assertEqual(len(self.signals), self.out["signals_sent"])
        self.assertEqual(self.out["stats"].total + len(self.out["open_at_stop"]),
                         self.out["signals_sent"],
                         "each signal is either recorded or still running")

    def test_results_are_reported_as_they_settle(self):
        stats = self.out["stats"]
        self.assertEqual(len(self.confirmations), stats.total)
        for confirmation in self.confirmations:
            self.assertIn(confirmation["outcome"], ("WIN", "LOSS"))
            self.assertGreater(confirmation["expiry_at"], BASE)

    def test_win_rate_line_is_consistent_with_the_tally(self):
        stats = self.out["stats"]
        if stats.total == 0:
            self.skipTest("no trades resolved in this window")
        last = self.confirmations[-1]
        self.assertAlmostEqual(last["win_rate"], stats.win_rate)
        self.assertEqual(last["win_rate"], stats.wins / (stats.wins + stats.losses))

    def test_stats_survive_a_restart(self):
        stats = self.out["stats"]
        # Writes are debounced in real time, so a long replay in virtual time
        # only lands the last one; the supervisor forces a final save on exit.
        stats.save(force=True)

        reloaded = StatsTracker(str(self.dir / "stats.json"), None)
        self.assertEqual(reloaded.total, stats.total)
        self.assertEqual(reloaded.wins, stats.wins)

    def test_open_trades_expire_exactly_one_bar_after_entry(self):
        for trade in self.out["open_at_stop"]:
            self.assertEqual(trade.entry_at % PERIOD, 0)
            self.assertEqual(trade.expiry_at - trade.entry_at, PERIOD)


class TestSignalJournal(EndToEndCase):
    """The per-trade record the broker reconciliation is built on."""

    def journal(self):
        return load_journal(self.dir / "signals.jsonl")

    def test_every_signal_the_bot_sent_is_in_the_journal(self):
        journal = self.journal()

        self.assertEqual(len(journal), self.out["signals_sent"])
        sent = {(s["asset"], s["entry_at"]) for s in self.signals}
        self.assertEqual({(t.asset, t.entry_at) for t in journal}, sent)
        self.assertEqual(journal.bad_lines, 0)
        self.assertEqual(journal.unknown_results, 0)

    def test_journalled_signals_carry_what_reconciliation_needs(self):
        for trade in self.journal():
            self.assertEqual(trade.entry_at % PERIOD, 0, "entry is on a boundary")
            self.assertEqual(trade.expiry_at - trade.entry_at, PERIOD)
            self.assertEqual(trade.payout, 85, "the sim feed pays 85%")
            self.assertIn(trade.direction, ("CALL", "PUT"))

    def test_the_journal_settles_exactly_what_the_stats_recorded(self):
        journal = self.journal()
        stats = self.out["stats"]

        self.assertEqual(len(journal.settled), stats.total)
        self.assertEqual(len(journal.unsettled), len(self.out["open_at_stop"]))
        self.assertEqual(journal.outcomes().get("WIN", 0), stats.wins)
        self.assertEqual(journal.outcomes().get("LOSS", 0), stats.losses)
        self.assertAlmostEqual(journal.our_win_rate(), stats.win_rate)

    def test_a_journalled_outcome_follows_the_prices_it_recorded(self):
        # The label has to be reproducible from the record, or reconciliation
        # would be comparing the broker against a claim it cannot check.
        for trade in self.journal().settled:
            self.assertIsNotNone(trade.our_entry)
            self.assertIsNotNone(trade.our_exit)
            if trade.direction == "CALL":
                expected = "WIN" if trade.our_exit > trade.our_entry else "LOSS"
            else:
                expected = "WIN" if trade.our_exit < trade.our_entry else "LOSS"
            self.assertEqual(trade.our_outcome, expected)


class TestSessionLifecycle(EndToEndCase):
    minutes = 30

    def test_startup_and_warmup_are_announced(self):
        texts = " ".join(self.out["sender"].texts)
        self.assertIn("started", texts)
        self.assertIn("one signal per 1m bar", texts)

    def test_candle_history_is_written_so_a_restart_is_warm(self):
        self.assertTrue(self.bars_on_disk(), "bars should be persisted on shutdown")

        # The store records which feed wrote it, so a reader has to name one.
        stored = CandleStore(str(self.dir / "candles"), PERIOD, 500,
                             feed="simulated").load(SYMBOLS[0])
        self.assertTrue(stored, "the persisted series should reload")
        for candle in stored:
            self.assertEqual(candle.time % PERIOD, 0, "bars are aligned to the minute")
            self.assertGreater(candle.high, 0)
            self.assertLessEqual(candle.low, candle.high)

    def test_the_watchdog_catches_a_feed_that_goes_quiet(self):
        cfg = self.config()

        async def scenario():
            clock = VirtualClock(start=BASE)
            market = MarketState(PERIOD, 100, 120, clock)
            market.on_tick("EURUSD_otc", BASE, 1.0)
            controller = BotController()
            task = asyncio.create_task(
                watch_feed(market, ["EURUSD_otc"], cfg, controller, clock))
            await clock.advance(30.0, steps=60)
            self.assertFalse(task.done(), "a live feed must not trip the watchdog")
            await clock.advance(300.0, steps=300)
            with self.assertRaises(FeedStalled):
                await asyncio.wait_for(task, timeout=10.0)

        asyncio.run(scenario())

    def test_the_watchdog_catches_a_dropped_connection_at_once(self):
        # Tick silence has to be waited out; a dead transport does not, and the
        # difference is most of a flap cycle. On 2026-09-15 engineio gave up on
        # the connection ~8s after the last tick, but the watchdog waited the
        # full silence window and only reacted 95s after it.
        cfg = self.config()

        class DroppingFeed(DataFeed):
            """A feed whose socket has gone, with ticks still recent."""

            def __init__(self):
                self.alive = True

            @property
            def is_connected(self):
                return self.alive

            async def connect(self): ...
            async def close(self): ...
            async def asset_meta(self): return {}
            async def subscribe(self, symbols): ...
            def set_tick_handler(self, handler): ...

        async def scenario():
            clock = VirtualClock(start=BASE)
            market = MarketState(PERIOD, 100, 120, clock)
            controller = BotController()
            feed = DroppingFeed()
            task = asyncio.create_task(
                watch_feed(market, ["EURUSD_otc"], cfg, controller, clock, feed=feed))
            await clock.advance(30.0, steps=60)
            self.assertFalse(task.done())

            feed.alive = False
            await clock.advance(10.0, steps=20)
            with self.assertRaises(FeedStalled):
                await asyncio.wait_for(task, timeout=10.0)

        asyncio.run(scenario())

    def test_the_watchdog_stands_down_on_stop(self):
        cfg = self.config()

        async def scenario():
            clock = VirtualClock(start=BASE)
            market = MarketState(PERIOD, 100, 120, clock)
            controller = BotController()
            controller.request_stop()
            await asyncio.wait_for(
                watch_feed(market, ["EURUSD_otc"], cfg, controller, clock), timeout=10.0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
