"""Tests for configuration: the defaults that define the product, and the
fail-fast validation that stops a typo from becoming a silent bot.

The environment is cleared for every case, so these assert what the *code*
defaults to rather than whatever happens to be in the developer's ``.env``.
"""

import os
import unittest
import unittest.mock

from config import ConfigError, load_config


class ConfigCase(unittest.TestCase):
    """Runs ``load_config`` against a scrubbed environment."""

    def load(self, **env):
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            return load_config()

    def assert_rejected(self, message_fragment, **env):
        with self.assertRaises(ConfigError) as ctx:
            self.load(**env)
        self.assertIn(message_fragment, str(ctx.exception))
        return str(ctx.exception)


class TestDefaults(ConfigCase):
    """No .env at all: the bot must still start, in the intended configuration."""

    def setUp(self):
        self.cfg = self.load()

    def test_runs_offline_by_default(self):
        self.assertEqual(self.cfg.feed, "simulated")
        self.assertFalse(self.cfg.telegram_enabled)

    def test_one_minute_cadence(self):
        self.assertEqual(self.cfg.candle_period, 60)
        self.assertEqual(self.cfg.expiry, "1m")
        self.assertEqual(self.cfg.expiry_seconds, 60)
        self.assertEqual(self.cfg.lead_seconds, 10, "signalled before the boundary")
        self.assertEqual(self.cfg.cooldown_seconds, 120)

    def test_universe(self):
        self.assertEqual(self.cfg.asset_mode, "major")
        self.assertTrue(self.cfg.otc_only)
        self.assertEqual(self.cfg.min_payout, 60)
        self.assertEqual(self.cfg.assets, [])

    def test_warmup_and_persistence(self):
        self.assertTrue(self.cfg.persist_candles)
        self.assertEqual(self.cfg.candle_store_dir, "candles")
        self.assertEqual(self.cfg.max_bars, 500)
        self.assertGreater(self.cfg.stale_after, self.cfg.candle_period,
                           "a healthy market must not be dropped mid-bar")

    def test_engine_defaults_suit_one_minute_bars(self):
        signal = self.cfg.signal
        self.assertEqual(signal.bar_seconds, 60)
        self.assertEqual(signal.min_components, 2, "two indicators must agree")
        self.assertEqual(signal.min_score, 2)
        self.assertEqual(signal.trend_flat_min_score, 3,
                         "a flat market demands more evidence, it is not vetoed")
        self.assertTrue(signal.use_trend)
        self.assertTrue(signal.use_mtf)
        self.assertEqual(signal.mtf_factor, 5, "5 x 1m bars = a 5m confirmation")
        self.assertEqual(signal.mtf_ema_len, 10)
        self.assertTrue(signal.use_sr)
        self.assertTrue(signal.use_atr)

    def test_breaker_is_on_by_default(self):
        self.assertEqual(self.cfg.max_consecutive_losses, 5)
        self.assertEqual(self.cfg.breaker_pause_minutes, 15)
        self.assertEqual(self.cfg.martingale_steps, 0, "martingale is opt-in")


class TestDerived(ConfigCase):
    def test_cadence_carries_the_timing_and_risk_settings(self):
        cadence = self.load(CANDLE_PERIOD="60", EXPIRY="1m", SIGNAL_LEAD_SECONDS="8",
                            COOLDOWN_SECONDS="90", MAX_CONSECUTIVE_LOSSES="3",
                            BREAKER_PAUSE_MINUTES="20", MARTINGALE_STEPS="2").cadence()
        self.assertEqual(cadence.period, 60)
        self.assertEqual(cadence.expiry_seconds, 60)
        self.assertEqual(cadence.lead_seconds, 8)
        self.assertEqual(cadence.cooldown_seconds, 90)
        self.assertEqual(cadence.max_consecutive_losses, 3)
        self.assertEqual(cadence.pause_minutes, 20)
        self.assertEqual(cadence.martingale_steps, 2)

    def test_cadence_carries_the_confidence_floor(self):
        self.assertEqual(self.load(MIN_CONFIDENCE="0.4").cadence().min_confidence, 0.4)

    def test_telegram_enabled_needs_both_fields(self):
        self.assertFalse(self.load().telegram_enabled)
        self.assertTrue(self.load(TELEGRAM_BOT_TOKEN="test-token",
                                  TELEGRAM_CHAT_ID="12345").telegram_enabled)


class TestExpiryParsing(ConfigCase):
    def test_durations(self):
        for text, seconds in (("1m", 60), ("5m", 300), ("1h", 3600), ("90s", 90),
                              ("2", 120)):
            self.assertEqual(self.load(EXPIRY=text).expiry_seconds, seconds, text)

    def test_empty_expiry_falls_back_to_one_minute(self):
        self.assertEqual(self.load(EXPIRY="").expiry_seconds, 60)

    def test_unparseable_expiry_is_rejected(self):
        self.assert_rejected("EXPIRY", EXPIRY="soon")


class TestUniverse(ConfigCase):
    def test_explicit_asset_list_is_split_and_trimmed(self):
        cfg = self.load(ASSETS=" EURUSD_otc , GBPJPY_otc ,,")
        self.assertEqual(cfg.assets, ["EURUSD_otc", "GBPJPY_otc"])

    def test_payout_floor(self):
        self.assertEqual(self.load(MIN_PAYOUT="75").min_payout, 75)

    def test_models_are_case_insensitive(self):
        self.assertEqual(self.load(ASSET_MODE="FOREX").asset_mode, "forex")

    def test_unknown_mode_is_rejected(self):
        self.assert_rejected("ASSET_MODE", ASSET_MODE="crypto")


class TestValidation(ConfigCase):
    """Every rejection here would otherwise be a bot that silently does nothing."""

    def test_unknown_feed(self):
        self.assert_rejected("FEED", FEED="csv")

    def test_absurd_candle_period(self):
        self.assert_rejected("CANDLE_PERIOD", CANDLE_PERIOD="3")
        self.assert_rejected("CANDLE_PERIOD", CANDLE_PERIOD="7200")

    def test_lead_longer_than_the_bar(self):
        message = self.assert_rejected("SIGNAL_LEAD_SECONDS", SIGNAL_LEAD_SECONDS="61")
        self.assertIn("before its entry", message)

    def test_a_lead_of_one_whole_bar_is_allowed(self):
        # The long-lead cadence: signal 16:00, entry 16:05. One bar is the
        # limit, and it is a real limit rather than a preference. The stale
        # window has to come up with the bar or the market is dropped mid-bar.
        for period, lead, stale in (("60", "60", "120"), ("300", "300", "420")):
            with self.subTest(period=period):
                cfg = self.load(CANDLE_PERIOD=period, SIGNAL_LEAD_SECONDS=lead,
                                STALE_AFTER=stale)
                self.assertEqual(cfg.lead_seconds, int(lead))
                self.assertEqual(cfg.candle_period, int(period))

    def test_a_lead_of_one_whole_bar_needs_a_stale_window_to_match(self):
        # A 5-minute bar with the 1-minute stale window is still refused: a
        # market would be dropped mid-bar. The two settings move together.
        self.assert_rejected("STALE_AFTER", CANDLE_PERIOD="300",
                             SIGNAL_LEAD_SECONDS="300", STALE_AFTER="120")

    def test_non_positive_lead(self):
        self.assert_rejected("SIGNAL_LEAD_SECONDS", SIGNAL_LEAD_SECONDS="0")

    def test_stale_window_shorter_than_the_bar(self):
        self.assert_rejected("STALE_AFTER", STALE_AFTER="60")

    def test_buffer_too_small_to_warm_up(self):
        self.assert_rejected("MAX_BARS", MAX_BARS="50")

    def test_payout_out_of_range(self):
        self.assert_rejected("MIN_PAYOUT", MIN_PAYOUT="150")

    def test_negative_martingale(self):
        self.assert_rejected("MARTINGALE_STEPS", MARTINGALE_STEPS="-1")

    def test_min_components_cannot_be_zero(self):
        self.assert_rejected("MIN_COMPONENTS", MIN_COMPONENTS="0")

    def test_min_confidence_out_of_range(self):
        self.assert_rejected("MIN_CONFIDENCE", MIN_CONFIDENCE="1.5")

    def test_pocket_feed_requires_credentials(self):
        self.assert_rejected("POCKET_SSID", FEED="pocket_option")

    def test_pocket_feed_accepts_credentials(self):
        cfg = self.load(FEED="pocket_option", POCKET_SSID="test-ssid", POCKET_UID="42")
        self.assertEqual(cfg.feed, "pocket_option")
        self.assertEqual(cfg.pocket_uid, 42)

    def test_token_without_a_chat_id(self):
        self.assert_rejected("TELEGRAM_CHAT_ID", TELEGRAM_BOT_TOKEN="test-token")


class TestBadNumbers(ConfigCase):
    def test_non_numeric_integers(self):
        self.assert_rejected("CANDLE_PERIOD", CANDLE_PERIOD="sixty")

    def test_non_numeric_floats(self):
        self.assert_rejected("MIN_CONFIDENCE", MIN_CONFIDENCE="high")

    def test_blank_values_use_the_default(self):
        cfg = self.load(CANDLE_PERIOD="  ", MIN_CONFIDENCE="")
        self.assertEqual(cfg.candle_period, 60)
        self.assertEqual(cfg.signal.min_confidence, 0.0)


class TestBooleans(ConfigCase):
    def test_truthy_spellings(self):
        for text in ("1", "true", "TRUE", "yes", "on"):
            self.assertTrue(self.load(USE_TREND=text).signal.use_trend, text)

    def test_falsy_spellings(self):
        for text in ("0", "false", "no", "off", "anything else"):
            self.assertFalse(self.load(USE_TREND=text).signal.use_trend, text)

    def test_defaults_when_blank(self):
        cfg = self.load(USE_MTF="", USE_SR="")
        self.assertTrue(cfg.signal.use_mtf)
        self.assertTrue(cfg.signal.use_sr)


class TestIndicatorOverrides(ConfigCase):
    def test_every_indicator_dial_is_wired(self):
        cfg = self.load(RSI_LEN="7", MACD_FAST="6", MACD_SLOW="13", MACD_SIGNAL="4",
                        BB_LEN="10", BB_MULT="1.5", STOCH_K="9", ATR_LEN="7",
                        TREND_EMA_LEN="20", TREND_FLAT_MIN_SCORE="5", EVENT_WEIGHT="3",
                        MIN_SCORE="4", SR_PROXIMITY_PCT="0.5")
        signal = cfg.signal
        self.assertEqual(signal.rsi_len, 7)
        self.assertEqual((signal.macd_fast, signal.macd_slow, signal.macd_signal),
                         (6, 13, 4))
        self.assertEqual((signal.bb_len, signal.bb_mult), (10, 1.5))
        self.assertEqual(signal.stoch_k, 9)
        self.assertEqual(signal.atr_len, 7)
        self.assertEqual(signal.trend_ema_len, 20)
        self.assertEqual(signal.trend_flat_min_score, 5)
        self.assertEqual(signal.event_weight, 3)
        self.assertEqual(signal.min_score, 4)
        self.assertAlmostEqual(signal.sr_proximity_pct, 0.5)

    def test_indicators_can_be_switched_off(self):
        signal = self.load(USE_RSI="0", USE_MACD="0", USE_BB="0", USE_STOCH="0",
                           USE_MTF="0", USE_ATR="0").signal
        self.assertFalse(any([signal.use_rsi, signal.use_macd, signal.use_bb,
                              signal.use_stoch, signal.use_mtf, signal.use_atr]))


class TestAutoTrade(ConfigCase):
    """The first of the three refusals that keep orders off a funded account.

    It is tested as behaviour because it is the one that runs before a socket is
    opened or the session string is touched, so it is the one that has to hold on
    its own — a ``.env`` that says demo while the session string came from a live
    login is exactly the mistake being fenced off.
    """

    # A live feed, so the only thing under test is the auto-trade check.
    LIVE = dict(FEED="pocket_option", POCKET_SSID="test-ssid", POCKET_UID="42")

    def test_off_by_default(self):
        cfg = self.load()

        self.assertFalse(cfg.auto_trade, "the bot signals unless told otherwise")
        self.assertAlmostEqual(cfg.trade_amount, 1.0)

    def test_asked_for_on_a_simulated_feed_is_refused(self):
        # There is nothing to trade, and a bot that thinks it is trading a
        # simulated market is worse than one that refuses to start.
        self.assert_rejected("FEED=pocket_option", AUTO_TRADE="1")

    def test_a_funded_account_is_refused(self):
        self.assert_rejected("will not do that on a funded account",
                             **self.LIVE, AUTO_TRADE="1", POCKET_IS_DEMO="0")

    def test_a_demo_account_is_accepted(self):
        cfg = self.load(**self.LIVE, POCKET_IS_DEMO="1", AUTO_TRADE="1",
                        TRADE_AMOUNT="2.5")

        self.assertTrue(cfg.auto_trade)
        self.assertAlmostEqual(cfg.trade_amount, 2.5)

    def test_a_non_positive_amount_is_refused(self):
        for amount in ("0", "-1"):
            with self.subTest(amount=amount):
                self.assert_rejected("must be positive", **self.LIVE,
                                     POCKET_IS_DEMO="1", AUTO_TRADE="1",
                                     TRADE_AMOUNT=amount)

    def test_a_funded_account_may_still_be_read_for_signals(self):
        # The fence is on placing orders, not on connecting: watching the real
        # market with AUTO_TRADE off is the normal way to run this.
        cfg = self.load(**self.LIVE, POCKET_IS_DEMO="0", AUTO_TRADE="0")

        self.assertFalse(cfg.auto_trade)
        self.assertFalse(cfg.pocket_is_demo)

    def test_the_amount_dial_is_read_whether_or_not_it_is_in_use(self):
        cfg = self.load(TRADE_AMOUNT="5")

        self.assertAlmostEqual(cfg.trade_amount, 5.0)


if __name__ == "__main__":
    unittest.main()
