"""Tests for the rendered messages.

Nothing here touches the network — these are the strings a person reads and acts
on, and they are the only part of the bot whose output is not a log line. A wrong
entry time costs money when the trade is placed by hand, so the clock, the
countdown and the wording around the price are all pinned here.

``configure_clock`` rebinds a module global, so any test that moves it puts it
back in a cleanup — otherwise it leaks into whatever runs next.
"""

import datetime as dt
import unittest

from signals.engine import Readiness, Signal
from telegram.sender import (
    clock, configure_clock, expiry_seconds, format_confirmation, format_duration,
    format_signal, format_startup, format_status, format_warmup,
    parse_expiry_minutes,
)

# 13:00 UTC renders as 16:00 in the default UTC+3, which is the shape that was
# asked for: signal 16:00, entry 16:05, closes 16:10.
BASE = dt.datetime(2026, 9, 15, 13, 0, 0, tzinfo=dt.timezone.utc).timestamp()
PERIOD = 300


def signal(direction="CALL", score=3, votes=("RSI", "MACD"), price=1.16716,
           confidence=0.62) -> Signal:
    return Signal(direction=direction, score=score, votes=list(votes), price=price,
                  time=BASE, confidence=confidence)


def strip_markup(text: str) -> str:
    return text.replace("<b>", "").replace("</b>", "")


class ClockCase(unittest.TestCase):
    """Anything that moves the clock must move it back."""

    def setUp(self):
        self.addCleanup(configure_clock, 3.0)


class TestClock(ClockCase):
    def test_the_default_zone_is_utc_plus_three(self):
        self.assertEqual(clock(BASE), "16:00:00")

    def test_configuring_the_offset_moves_every_rendered_time(self):
        configure_clock(0.0)
        self.assertEqual(clock(BASE), "13:00:00", "three hours earlier")

        configure_clock(-5.0)
        self.assertEqual(clock(BASE), "08:00:00")

    def test_the_offset_is_accepted_as_text(self):
        # It arrives from .env, where everything is a string.
        configure_clock("0")
        self.assertEqual(clock(BASE), "13:00:00")

    def test_an_already_imported_clock_follows_the_reconfiguration(self):
        # This is the whole reason ``clock`` is a late-binding function rather
        # than a zone constant: scheduler.py binds the *function* at import, so a
        # zone captured as a default or a module attribute would go stale.
        before = clock(BASE)
        configure_clock(0.0)
        self.assertNotEqual(clock(BASE), before)


class TestDurationLabels(unittest.TestCase):
    def test_bar_labels(self):
        for seconds, label in ((60, "1m"), (300, "5m"), (3600, "1h"),
                               (7200, "2h"), (45, "45s"), (0, "0s")):
            with self.subTest(seconds=seconds):
                self.assertEqual(format_duration(seconds), label)

    def test_spans_over_an_hour_combine_hours_and_minutes(self):
        # The trend veto's warm-up is 56 bars — 4h40m at 300s, and 280m written
        # as minutes. Anything over an hour is expressed in both, because these
        # labels now carry warm-up waits and not only bar lengths.
        for seconds, label in ((16800, "4h40m"), (5400, "1h30m"),
                               (3660, "1h01m"), (3601, "1h1s"), (3540, "59m")):
            with self.subTest(seconds=seconds):
                self.assertEqual(format_duration(seconds), label)

    def test_expiry_strings_to_seconds(self):
        for text, seconds in (("1m", 60), ("5m", 300), ("1h", 3600), ("90s", 90),
                              ("2", 120), ("", 60)):
            with self.subTest(expiry=text):
                self.assertEqual(expiry_seconds(text), seconds)

    def test_expiry_strings_to_minutes(self):
        for text, minutes in (("5m", 5), ("1m", 1), ("1h", 60), ("90s", 1),
                              ("", 1), ("30s", 1), ("2", 2)):
            with self.subTest(expiry=text):
                self.assertEqual(parse_expiry_minutes(text), minutes)


class TestSignalMessage(ClockCase):
    """The message is a set of instructions: when to enter, when it closes."""

    def render(self, now=None, **kwargs):
        return format_signal(signal(**kwargs.pop("signal_kwargs", {})), "EURUSD_otc",
                             "5m", BASE + PERIOD, now=now, **kwargs)

    def test_the_three_moments_read_as_a_schedule(self):
        text = self.render(now=BASE, entry_price=1.16716)

        self.assertIn("16:05:00", text, "the entry time")
        self.assertIn("closes 16:10:00", text, "the expiry as a clock time")
        self.assertIn("Expiration 5 minutes", text)

    def test_a_five_minute_wait_does_not_read_as_seconds(self):
        text = self.render(now=BASE)

        self.assertIn("(in 5m)", text)
        self.assertNotIn("300s", text, "the reader should not do the division")

    def test_a_partial_wait_keeps_the_remainder(self):
        self.assertIn("(in 4m 30s)", self.render(now=BASE + 30))

    def test_an_imminent_entry_counts_in_seconds(self):
        self.assertIn("(in 45s)", self.render(now=BASE + PERIOD - 45))

    def test_the_price_line_says_it_is_stale(self):
        # At a full-bar lead the price in the message is five minutes old by the
        # time the trade opens. Saying "Price" and nothing else invites the reader
        # to place at a number that has moved.
        text = self.render(now=BASE, entry_price=1.16716)

        self.assertIn("💵 Price 1.16716 (at signal — you enter at 16:05:00)", text)

    def test_at_the_boundary_there_is_no_wait_to_qualify(self):
        text = self.render(now=BASE + PERIOD, entry_price=1.16716)

        self.assertIn("✅ Entry now (16:05:00)", text)
        self.assertIn("💵 Price 1.16716", text)
        self.assertNotIn("at signal", text, "there is nothing to warn about")

    def test_a_past_entry_does_not_count_downward(self):
        text = self.render(now=BASE + 10 * PERIOD)

        self.assertIn("✅ Entry now", text)
        self.assertNotIn("(in -", text)

    def test_no_price_means_no_price_line(self):
        self.assertNotIn("💵", self.render(now=BASE))

    def test_direction_is_spelled_out_not_coded(self):
        self.assertIn("🟢 <b>BUY</b>", self.render(now=BASE, signal_kwargs={
            "direction": "CALL"}))
        self.assertIn("🔴 <b>SELL</b>", self.render(now=BASE, signal_kwargs={
            "direction": "PUT"}))

    def test_the_detail_line_carries_the_evidence(self):
        text = self.render(now=BASE, payout=85)

        self.assertIn("score 3", text)
        self.assertIn("62% conf", text)
        self.assertIn("payout 85%", text)
        self.assertIn("RSI, MACD", text)

    def test_the_asset_is_humanised(self):
        text = format_signal(signal(), "EURUSD_otc", "5m", BASE + PERIOD, now=BASE)

        self.assertIn("EUR/USD OTC", text)
        self.assertNotIn("_otc", text)

    def test_martingale_steps_are_a_clock_schedule_too(self):
        text = self.render(now=BASE, martingale_steps=2)

        self.assertIn("MARTINGALE AT 16:10:00", text)
        self.assertIn("MARTINGALE AT 16:15:00", text)


class TestConfirmationMessage(ClockCase):
    def render(self, outcome="WIN", direction="CALL", **kwargs):
        return format_confirmation("EURUSD_otc", direction, outcome, 7, 3, 0.7,
                                   expiry_at=BASE + PERIOD, **kwargs)

    def test_a_win_reports_the_tally_and_the_clock(self):
        text = self.render()

        self.assertIn("✅ <b>WIN</b>", text)
        self.assertIn("Expired 16:05:00", text)
        self.assertIn("Win rate: 70% (7W / 3L)", text)

    def test_a_loss_is_marked_as_one(self):
        self.assertIn("❌ <b>LOSS</b>", self.render(outcome="LOSS"))

    def test_a_refund_is_not_rendered_as_a_loss(self):
        # A refund is neither: counting it as a loss would understate the bot to
        # the person deciding whether to keep reading the signals.
        text = self.render(outcome="PUSH")

        self.assertIn("↩️ <b>REFUNDED</b>", text)
        self.assertNotIn("LOSS", text)
        self.assertIn("(not counted in the win rate)", text)

    def test_an_order_that_never_landed_says_so(self):
        text = self.render(outcome="UNPLACED")

        self.assertIn("⚠️ <b>NOT PLACED</b>", text)
        self.assertNotIn("LOSS", text)

    def test_a_streak_line_is_optional(self):
        self.assertIn("🔥 3 wins in a row", self.render(streak="3 wins in a row"))
        self.assertNotIn("🔥", self.render())


class TestStartupAndStatus(ClockCase):
    def test_startup_names_the_cadence_in_minutes(self):
        text = format_startup(assets=20, period=PERIOD, expiry="5m", mode="major",
                              min_payout=60, bars_restored=0)

        self.assertIn("Cadence: one signal per 5m bar · expiry 5m", text)
        self.assertNotIn("300s", text)

    def test_startup_states_how_it_will_execute(self):
        text = format_startup(20, PERIOD, "5m", "major", 60, 0,
                              execution="signal-only — no orders will be placed")

        self.assertIn("signal-only", text)

    def test_startup_mentions_restored_bars_only_when_there_are_some(self):
        self.assertIn("Restored 54 bars", format_startup(20, PERIOD, "5m", "major",
                                                        60, 54))
        self.assertNotIn("Restored", format_startup(20, PERIOD, "5m", "major", 60, 0))

    def test_status_counts_down_to_the_next_entry(self):
        text = format_status(running=True, assets=20, healthy=18, bars=30,
                             bars_needed=56, next_entry_at=BASE + PERIOD, now=BASE)

        self.assertIn("Markets: 18/20 live", text)
        self.assertIn("Bars: 30/56", text)
        self.assertIn("Next entry: 16:05:00 (in 5m)", text)

    def test_status_labels_the_bar_count_as_one_markets(self):
        # It is the maximum across markets, not the session's progress: read
        # unlabelled, 81/56 over a session where 19 of 21 markets could not be
        # judged at all says "fully warmed".
        text = format_status(running=True, assets=21, healthy=16, bars=81,
                             bars_needed=56, next_entry_at=None, now=BASE)

        self.assertIn("Bars: 81/56 (most advanced market)", text)

    def test_status_without_a_next_entry_omits_the_line(self):
        text = format_status(running=False, assets=20, healthy=0, bars=0,
                             bars_needed=56, next_entry_at=None, now=BASE)

        self.assertIn("Bot: paused", text)
        self.assertNotIn("Next entry", text)


class TestWarmupMessage(unittest.TestCase):
    def test_warmup_reports_the_wait(self):
        readiness = Readiness(bars=18, bars_needed=56, missing=["trend", "MTF"],
                              available=["RSI", "MACD"])
        text = format_warmup(readiness, eta_minutes=190, available="RSI, MACD")

        self.assertIn("Bars: 18/56", text)
        self.assertIn("Waiting on: trend, MTF", text)
        self.assertIn("Full readiness in ~190 min", text)

    def test_a_warm_buffer_says_signals_are_live(self):
        readiness = Readiness(bars=56, bars_needed=56, missing=[], available=[])
        text = format_warmup(readiness)

        self.assertIn("Fully warmed up — signals active.", text)

    def test_a_warm_buffer_does_not_speak_for_a_cold_session(self):
        # The same message, one bar count higher, over the session the live bot
        # actually had on 2026-09-16: the leading market held 81 bars while 19 of
        # the other 20 held 15-53. "Signals active" was true of a market and
        # false of the bot, and it reads as "the bot is working".
        readiness = Readiness(bars=81, bars_needed=56, missing=[], available=[])
        text = format_warmup(readiness, eta_minutes=0, gate_census=(1, 21))

        self.assertIn("The trend veto is warm on 1 of 21 market(s)", text)
        self.assertIn("only from the markets past the veto", text)
        self.assertNotIn("Fully warmed up", text)

    def test_a_session_that_is_whole_still_says_so(self):
        # The claim is kept for the state that earns it, or the note becomes a
        # caveat nobody reads.
        readiness = Readiness(bars=81, bars_needed=56, missing=[], available=[])
        text = format_warmup(readiness, eta_minutes=0, gate_census=(21, 21))

        self.assertIn("Fully warmed up — signals active.", text)
        self.assertNotIn("veto", text)

    def test_a_partly_warm_session_mid_warmup_reports_both(self):
        # Both halves are needed while it is still filling: the distance to
        # readiness, and how much of the session that readiness would be true of.
        readiness = Readiness(bars=18, bars_needed=56, missing=["trend"],
                              available=["RSI"])
        text = format_warmup(readiness, eta_minutes=190, gate_census=(2, 16))

        self.assertIn("Full readiness in ~190 min", text)
        self.assertIn("2 of 16 market(s)", text)

    def test_no_census_means_no_count_line(self):
        # With no veto configured there is nothing to count, and an absent
        # argument must not be read as "zero markets are warm".
        readiness = Readiness(bars=18, bars_needed=56, missing=["RSI"],
                              available=[])
        text = format_warmup(readiness, eta_minutes=190)

        self.assertNotIn("veto", text)
        self.assertNotIn("market(s)", text)


class TestMarkupSafety(ClockCase):
    """Messages go out with parse_mode=HTML, so a stray bracket is a broken send."""

    def test_rendered_messages_use_only_the_allowed_tags(self):
        messages = [
            format_signal(signal(), "EURUSD_otc", "5m", BASE + PERIOD, now=BASE,
                          payout=85, entry_price=1.16716, martingale_steps=2),
            format_confirmation("EURUSD_otc", "CALL", "WIN", 7, 3, 0.7,
                                expiry_at=BASE + PERIOD, streak="3 in a row"),
            format_startup(20, PERIOD, "5m", "major", 60, 54, execution="signal-only"),
            format_status(running=True, assets=20, healthy=18, bars=30,
                          bars_needed=56, next_entry_at=BASE + PERIOD, now=BASE),
            format_warmup(Readiness(bars=1, bars_needed=56, missing=["x"],
                                    available=[]), eta_minutes=5),
        ]
        for text in messages:
            with self.subTest(message=text.splitlines()[0]):
                for char in "<>&":
                    self.assertNotIn(char, strip_markup(text),
                                     f"unescaped {char!r} would break the send")


if __name__ == "__main__":
    unittest.main()
