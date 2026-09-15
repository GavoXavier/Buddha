"""Tests for the supervisor: the wake lock it holds, and the reconnect ladder.

The lock is the only thing standing between "the bot is running" and "the
machine slept and the socket died with the process still up". It is also the
one piece of the supervisor that touches the host directly, so what it asks for
is worth pinning precisely: the wrong flag keeps the screen lit, and a
silently-refused request leaves the bot asleep at the wheel with nobody told.

The ladder decides how hard the bot tries after a failure. Getting it wrong is
invisible — the bot stays up and appears to be working — so the rule is pinned
here rather than left to the loop.
"""

import sys
import tempfile
import unittest
from pathlib import Path

from config import Config
from main import (
    BACKOFF_MAX, BACKOFF_MIN, HEALTHY_SESSION_SECONDS, WakeLock, _archive_store,
    backoff_step, healthy_session_seconds, load_universe, resolve_universe,
    save_universe,
)
from market.universe import AssetMeta

# Documented in SetThreadExecutionState's remarks; not named here because the
# point of the flag is that we never pass it.
_ES_DISPLAY_REQUIRED = 0x00000002


class FakeExecutionState:
    """Stands in for kernel32.SetThreadExecutionState."""

    def __init__(self, result: int = 1):
        self.calls: list[int] = []
        self.result = result
        self.restype = None
        self.argtypes = None

    def __call__(self, flags: int) -> int:
        self.calls.append(flags)
        return self.result


def with_fake_api() -> tuple[WakeLock, FakeExecutionState]:
    """A WakeLock wired to a fake API, plus the fake to inspect.

    ``_set`` is the single seam between this class and the host, so replacing
    it is the whole substitution — there is no behaviour left unexercised.
    """
    lock = WakeLock()
    fake = FakeExecutionState()
    lock._set = fake
    return lock, fake


class TestWakeLock(unittest.TestCase):
    def test_it_asks_for_the_system_and_not_the_display(self):
        lock, fake = with_fake_api()

        self.assertTrue(lock.acquire())

        self.assertEqual(fake.calls,
                         [WakeLock.ES_CONTINUOUS | WakeLock.ES_SYSTEM_REQUIRED])
        self.assertEqual(fake.calls[0] & _ES_DISPLAY_REQUIRED, 0,
                         "the screen is allowed to sleep; only the system may not")

    def test_the_request_is_continuous_rather_than_one_shot(self):
        # Without ES_CONTINUOUS the lock expires immediately and the machine
        # sleeps on its usual timer, which would look like it worked.
        lock, fake = with_fake_api()
        lock.acquire()

        self.assertEqual(fake.calls[0] & WakeLock.ES_CONTINUOUS, WakeLock.ES_CONTINUOUS)

    def test_a_refused_request_is_reported_not_assumed(self):
        # The API returns 0 on failure. A bot that believes it holds a lock it
        # was refused is worse off than one that says so and gets watched.
        lock, fake = with_fake_api()
        fake.result = 0

        self.assertFalse(lock.acquire())

    def test_release_drops_the_request(self):
        # ES_CONTINUOUS on its own is the documented way to clear it; passing
        # the system-required flag again would re-take the lock.
        lock, fake = with_fake_api()
        lock.acquire()
        lock.release()

        self.assertEqual(fake.calls[-1], WakeLock.ES_CONTINUOUS)
        self.assertEqual(fake.calls[-1] & WakeLock.ES_SYSTEM_REQUIRED, 0)

    def test_without_the_api_it_is_a_no_op(self):
        lock = WakeLock()
        lock._set = None

        self.assertFalse(lock.supported)
        self.assertFalse(lock.acquire(), "no API is not a refusal, but it is not a lock")
        lock.release()          # must not raise

    def test_supported_tracks_whether_the_api_is_there(self):
        lock, _ = with_fake_api()
        self.assertTrue(lock.supported)

    def test_the_flag_values_are_the_documented_ones(self):
        self.assertEqual(WakeLock.ES_CONTINUOUS, 0x80000000)
        self.assertEqual(WakeLock.ES_SYSTEM_REQUIRED, 0x00000001)


class TestWakeLockOnWindows(unittest.TestCase):
    """The ctypes lookup itself — the part that fails silently if miswired."""

    @unittest.skipUnless(sys.platform == "win32", "the API is Windows-only")
    def test_the_real_api_is_found(self):
        self.assertTrue(WakeLock().supported,
                        "kernel32.SetThreadExecutionState was not resolved")

    @unittest.skipUnless(sys.platform == "win32", "the API is Windows-only")
    def test_the_real_api_grants_and_clears_the_lock(self):
        # A wrong restype or argtypes makes the call fail rather than raise, so
        # the only honest check is to actually take the lock and give it back.
        lock = WakeLock()
        self.assertTrue(lock.acquire(), "the host refused the wake lock")
        lock.release()


def meta(symbol, payout=85, active=True):
    return AssetMeta(symbol=symbol, payout=payout,
                     is_otc=symbol.endswith("_otc"), active=active)


class TestUniverseIsSticky(unittest.TestCase):
    """The universe must not churn between reconnects.

    ``select_assets`` reads the live payout list, which the server republishes
    on every connect. Re-deriving the universe per session meant markets came
    and went several times an hour; a market that drops out stops ticking, and
    a market with no ticks accumulates no bars, which is the chain that
    truncated the stored series on 2026-09-15.
    """

    def setUp(self):
        self.cfg = Config()

    def test_the_first_resolve_fills_the_list_and_later_ones_reuse_it(self):
        sticky: list[str] = []
        first = resolve_universe(self.cfg, {
            "EURUSD_otc": meta("EURUSD_otc", payout=92),
            "GBPUSD_otc": meta("GBPUSD_otc", payout=80),
        }, sticky)
        self.assertEqual(first, ["EURUSD_otc", "GBPUSD_otc"])

        # The payouts move on the next connect, as they do live.
        again = resolve_universe(self.cfg, {
            "EURUSD_otc": meta("EURUSD_otc", payout=61),
            "GBPUSD_otc": meta("GBPUSD_otc", payout=99),
            "USDJPY_otc": meta("USDJPY_otc", payout=95),
        }, sticky)

        self.assertEqual(again, first, "a reconnect keeps the same markets")
        self.assertEqual(sticky, first)

    def test_a_market_the_feed_drops_is_let_go(self):
        sticky = ["EURUSD_otc", "GBPUSD_otc"]
        kept = resolve_universe(self.cfg, {"EURUSD_otc": meta("EURUSD_otc")}, sticky)

        self.assertEqual(kept, ["EURUSD_otc"])
        self.assertEqual(sticky, ["EURUSD_otc"], "the list is updated in place")

    def test_an_empty_asset_list_leaves_the_universe_alone(self):
        # asset_meta() returns nothing until the server has answered; that is
        # not the same as the markets having gone away.
        sticky = ["EURUSD_otc"]
        self.assertEqual(resolve_universe(self.cfg, {}, sticky), ["EURUSD_otc"])

    def test_the_universe_is_rederived_if_nothing_survives(self):
        cfg = Config(min_payout=60)
        sticky = ["GONE_otc"]
        kept = resolve_universe(cfg, {
            "EURUSD_otc": meta("EURUSD_otc", payout=90),
            "AUDUSD_otc": meta("AUDUSD_otc", payout=10),
        }, sticky)

        self.assertEqual(kept, ["EURUSD_otc"], "an empty universe is worse than a new one")
        self.assertEqual(sticky, ["EURUSD_otc"])


class TestUniverseSurvivesARestart(unittest.TestCase):
    """The membership is held on disk, because a restart is not a new decision.

    This is the same argument as the sticky list one level out, and it was
    measured rather than reasoned: on 2026-09-16 five of the twenty-one stored
    markets had stopped receiving bars, and they were exactly the markets that
    had produced a signal — a market signals when its series is deep, and depth
    is what a re-resolved universe throws away. A restart is a thing this bot
    does often; four process starts in eighty minutes on 2026-09-15.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "universe.json")

    def test_a_first_run_has_nothing_to_hold(self):
        self.assertEqual(load_universe(self.path), ([], 0))

    def test_an_unreadable_file_is_re_derived_rather_than_refused(self):
        # A cache that cannot be read is not a reason to refuse to start: the
        # bot simply chooses a universe, which is where it would have been
        # without the file at all.
        for junk in ("not json at all", "{}", '{"symbols": "EURUSD_otc"}',
                     "[7, null]"):
            with self.subTest(junk=junk):
                Path(self.path).write_text(junk, encoding="utf-8")
                self.assertEqual(load_universe(self.path), ([], 0))

    def test_the_membership_and_the_size_round_trip(self):
        save_universe(self.path, ["EURUSD_otc", "GBPUSD_otc"], 16)

        self.assertEqual(load_universe(self.path),
                         (["EURUSD_otc", "GBPUSD_otc"], 16))

    def test_a_file_with_no_size_holds_what_it_lists(self):
        # What a hand-written or older file looks like. Reading the size off
        # the list is the same rule the file would have been written with.
        Path(self.path).write_text('["EURUSD_otc", "GBPUSD_otc"]',
                                   encoding="utf-8")

        self.assertEqual(load_universe(self.path),
                         (["EURUSD_otc", "GBPUSD_otc"], 2))

    def test_junk_beside_a_market_is_dropped_rather_than_the_market(self):
        # A half-written or hand-edited file should cost the entries that are
        # not markets, not all of them: the membership is the part that is
        # expensive to re-earn.
        Path(self.path).write_text('["EURUSD_otc", 7, null]', encoding="utf-8")

        self.assertEqual(load_universe(self.path), (["EURUSD_otc"], 1))

    def test_a_file_that_cannot_be_written_does_not_end_the_session(self):
        # The directory does not exist, which is what a mistyped UNIVERSE_PATH
        # looks like. The universe is a convenience, not a dependency.
        save_universe(str(Path(self.tmp.name) / "nope" / "universe.json"),
                      ["EURUSD_otc"], 1)

    def test_a_market_the_feed_lets_go_is_replaced_to_keep_the_size(self):
        # Otherwise the universe slowly shrinks to whatever the feed still
        # offers, and a session starves one departure at a time.
        cfg = Config(min_payout=60)
        sticky = ["EURUSD_otc", "GBPUSD_otc"]
        kept = resolve_universe(cfg, {
            "EURUSD_otc": meta("EURUSD_otc", payout=90),
            "USDJPY_otc": meta("USDJPY_otc", payout=95),
        }, sticky, target=2)

        self.assertEqual(kept, ["EURUSD_otc", "USDJPY_otc"])
        self.assertEqual(sticky, ["EURUSD_otc", "USDJPY_otc"])

    def test_the_replacement_never_outgrows_the_size_it_is_holding(self):
        cfg = Config(min_payout=60)
        sticky = ["EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc"]
        kept = resolve_universe(cfg, {
            "AUDUSD_otc": meta("AUDUSD_otc", payout=90),
            "USDJPY_otc": meta("USDJPY_otc", payout=95),
            "NZDUSD_otc": meta("NZDUSD_otc", payout=95),
            "USDCAD_otc": meta("USDCAD_otc", payout=95),
        }, sticky, target=3)

        self.assertEqual(kept[0], "AUDUSD_otc", "what survived keeps its place")
        self.assertEqual(len(kept), 3)

    def test_a_universe_that_cannot_be_refilled_keeps_what_it_has(self):
        # Every market the feed offers is under the payout floor while the one
        # already held is not. Nothing to replace it with is not a failure —
        # and it must not raise, or a payout dip would end the session.
        cfg = Config(min_payout=60)
        sticky = ["EURUSD_otc", "GBPUSD_otc"]
        kept = resolve_universe(cfg, {
            "EURUSD_otc": meta("EURUSD_otc", payout=90),
            "CHEAP_otc": meta("CHEAP_otc", payout=10),
        }, sticky, target=2)

        self.assertEqual(kept, ["EURUSD_otc"])

    def test_no_target_means_no_refill(self):
        # The pure behaviour the reconnect tests rely on: without a size to
        # hold, the universe is only ever pruned.
        sticky = ["EURUSD_otc", "GBPUSD_otc"]
        kept = resolve_universe(Config(), {
            "EURUSD_otc": meta("EURUSD_otc", payout=90),
            "USDJPY_otc": meta("USDJPY_otc", payout=95),
        }, sticky)

        self.assertEqual(kept, ["EURUSD_otc"])


class TestWhereTheArchiveLives(unittest.TestCase):
    """The archive must land beside whatever store it is copying.

    A fixed default put it in the working directory, so any caller that
    redirected ``candle_store_dir`` — a test into a temp directory, an operator
    to another disk — left the archive behind in the current directory instead.
    That was measured rather than imagined: five files of simulated bars were
    written into the repo's own ``candles-archive/`` by a test run on 2026-09-16.
    """

    def test_an_unnamed_archive_sits_beside_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(candle_store_dir=str(Path(tmp) / "candles"))
            store = _archive_store(cfg)

            self.assertEqual(Path(store.directory), Path(tmp) / "candles-archive")

    def test_archiving_can_be_turned_off(self):
        self.assertIsNone(_archive_store(Config(archive_candles=False)))

    def test_one_directory_for_both_tiers_is_refused(self):
        # Two tiers over one directory is not a deeper archive: every live write
        # would merge the deep series back in and truncate it away again.
        cfg = Config(candle_store_dir="candles", candle_archive_dir="candles")
        self.assertIsNone(_archive_store(cfg))

    def test_the_archive_is_deeper_than_the_store_and_reads_the_same_feed(self):
        cfg = Config(candle_store_dir="candles", max_bars=500, archive_bars=20000,
                     feed="pocket_option")
        store = _archive_store(cfg)

        self.assertEqual(store.max_bars, 20000)
        self.assertEqual(store.feed, "pocket_option", "the feed stamp comes along")
        self.assertEqual(store.period, cfg.candle_period)


class TestBackoffStep(unittest.TestCase):
    """The reconnect ladder, as the rule rather than as the loop."""

    def test_a_healthy_session_starts_the_ladder_again(self):
        failures, delay, backoff = backoff_step(HEALTHY_SESSION_SECONDS, 7, 60.0)

        self.assertEqual(failures, 1, "the count restarts, it does not vanish")
        self.assertEqual(delay, BACKOFF_MIN)
        self.assertEqual(backoff, BACKOFF_MIN * 2)

    def test_a_session_that_barely_connected_does_not_reset_anything(self):
        # This is the flapping case. Observed on 2026-09-15: delays climbed
        # 5-10-20 only while connect() itself failed, then snapped back to 5
        # the moment a connect succeeded and the session died seconds later —
        # so the feed was retried every five seconds indefinitely.
        failures, delay, backoff = backoff_step(3.0, 0, BACKOFF_MIN)

        self.assertEqual((failures, delay, backoff), (1, 5.0, 10.0))

    def test_repeated_short_sessions_climb(self):
        delays = []
        failures, backoff = 0, BACKOFF_MIN
        for _ in range(5):
            failures, delay, backoff = backoff_step(1.0, failures, backoff)
            delays.append(delay)

        self.assertEqual(delays, [5.0, 10.0, 20.0, 40.0, 60.0],
                         "each attempt waits longer than the last")
        self.assertEqual(failures, 5)

    def test_the_delay_is_capped(self):
        failures, backoff = 0, BACKOFF_MIN
        for _ in range(20):
            failures, _, backoff = backoff_step(1.0, failures, backoff)

        self.assertEqual(backoff, BACKOFF_MAX)

    def test_the_healthy_threshold_never_falls_below_the_floor(self):
        self.assertEqual(healthy_session_seconds(60), HEALTHY_SESSION_SECONDS)
        self.assertEqual(healthy_session_seconds(1), HEALTHY_SESSION_SECONDS)

    def test_a_longer_bar_needs_a_longer_session_to_count_as_healthy(self):
        # At a 300s bar the floor is under one bar, so a session could last two
        # minutes and still have produced nothing.
        self.assertEqual(healthy_session_seconds(300), 300.0)

    def test_the_threshold_is_one_bar_and_not_several(self):
        # The feed drops sessions every few minutes by itself. A threshold past
        # that gap pins the ladder at its cap and costs a minute of downtime
        # after every drop (observed live on 2026-09-15), so it must stay short
        # enough for a session that reached a boundary to count.
        self.assertEqual(healthy_session_seconds(300), 300.0)
        self.assertLess(healthy_session_seconds(300), 600.0)

    def test_a_session_shorter_than_the_threshold_keeps_backing_off(self):
        # 3 minutes of a flapping 5-minute feed is not a recovery: the delay
        # must keep growing rather than snapping back to five seconds.
        failures, delay, backoff = backoff_step(181.0, 3, 20.0, healthy=300.0)

        self.assertEqual((failures, delay, backoff), (4, 20.0, 40.0))

    def test_a_long_session_after_a_run_of_failures_resets(self):
        failures, backoff = 0, BACKOFF_MIN
        for _ in range(6):
            failures, _, backoff = backoff_step(1.0, failures, backoff)
        self.assertEqual(backoff, BACKOFF_MAX)

        failures, delay, backoff = backoff_step(3600.0, failures, backoff)

        self.assertEqual((failures, delay, backoff), (1, BACKOFF_MIN,
                                                      BACKOFF_MIN * 2))


if __name__ == "__main__":
    unittest.main()
