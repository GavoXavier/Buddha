"""Unit tests for the signal engine and indicators (stdlib unittest, no deps)."""

import math
import unittest

from signals import indicators
from signals.engine import Candle, SignalConfig, evaluate


def make_candles(closes, highs=None, lows=None, start_time=1_700_000_000.0, step=60.0):
    """Build Candle objects from close prices (highs/lows default to close)."""
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    out = []
    for i, c in enumerate(closes):
        out.append(Candle(
            time=start_time + i * step,
            open=c,
            high=highs[i],
            low=lows[i],
            close=c,
        ))
    return out


class TestIndicators(unittest.TestCase):
    def test_sma(self):
        self.assertEqual(indicators.sma_series([1, 2, 3, 4, 5], 3),
                         [None, None, 2.0, 3.0, 4.0])

    def test_ema(self):
        # seed = SMA(1,2,3) = 2.0; k = 0.5
        self.assertEqual(indicators.ema_series([1, 2, 3, 4, 5], 3),
                         [None, None, 2.0, 3.0, 4.0])

    def test_rsi_all_gains_is_100(self):
        closes = list(range(1, 16))  # 15 strictly rising closes
        rsi = indicators.rsi_series(closes, 14)
        self.assertAlmostEqual(rsi[-1], 100.0)

    def test_rsi_all_losses_is_0(self):
        closes = list(range(15, 0, -1))  # 15 strictly falling closes
        rsi = indicators.rsi_series(closes, 14)
        self.assertAlmostEqual(rsi[-1], 0.0)

    def test_bollinger(self):
        basis, upper, lower = indicators.bollinger_series([1, 2, 3, 4, 5], 5, 2.0)
        self.assertAlmostEqual(basis[-1], 3.0)
        self.assertAlmostEqual(upper[-1], 3.0 + 2.0 * math.sqrt(2.0))
        self.assertAlmostEqual(lower[-1], 3.0 - 2.0 * math.sqrt(2.0))

    def test_stochastic_flat_is_50(self):
        flat = [10.0] * 20
        k, d = indicators.stochastic_series(flat, flat, flat, 14, 3, 3)
        self.assertAlmostEqual(k[-1], 50.0)
        self.assertAlmostEqual(d[-1], 50.0)


class TestEngine(unittest.TestCase):
    def _macd_only_config(self, **overrides):
        cfg = SignalConfig(
            use_trend=False,
            use_rsi=False,
            use_macd=True,
            use_bb=False,
            use_stoch=False,
            use_sr=False,
            use_mtf=False,
            use_atr=False,
            min_score=1,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _decline_then_rise(self, n_down=60, n_up=30, start=100.0, bottom=70.0):
        """A decline followed by a rise — produces a MACD bullish cross."""
        down = [start - (start - bottom) * (i / (n_down - 1)) for i in range(n_down)]
        up = [bottom + (start - bottom) * (i / (n_up - 1)) for i in range(1, n_up + 1)]
        return down + up

    def _rise_then_decline(self, n_up=60, n_down=30, start=70.0, top=100.0):
        """A rise followed by a decline — produces a MACD bearish cross."""
        up = [start + (top - start) * (i / (n_up - 1)) for i in range(n_up)]
        down = [top - (top - start) * (i / (n_down - 1)) for i in range(1, n_down + 1)]
        return up + down

    def test_macd_bullish_cross_fires_call(self):
        closes = self._decline_then_rise()
        cfg = self._macd_only_config()

        # Find the bar where MACD actually crosses up, then feed up to it.
        macd_line, signal_line = indicators.macd_series(closes, 12, 26, 9)
        cross_idx = None
        for i in range(1, len(closes)):
            if (macd_line[i - 1] is not None and macd_line[i] is not None
                    and signal_line[i - 1] is not None and signal_line[i] is not None
                    and macd_line[i - 1] <= signal_line[i - 1]
                    and macd_line[i] > signal_line[i]):
                cross_idx = i
                break

        self.assertIsNotNone(cross_idx, "expected a MACD bullish cross in the series")

        sig = evaluate(make_candles(closes[: cross_idx + 1]), cfg)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, "CALL")
        self.assertIn("MACD", sig.votes)

    def test_macd_bearish_cross_fires_put(self):
        closes = self._rise_then_decline()
        cfg = self._macd_only_config()

        macd_line, signal_line = indicators.macd_series(closes, 12, 26, 9)
        cross_idx = None
        for i in range(1, len(closes)):
            if (macd_line[i - 1] is not None and macd_line[i] is not None
                    and signal_line[i - 1] is not None and signal_line[i] is not None
                    and macd_line[i - 1] >= signal_line[i - 1]
                    and macd_line[i] < signal_line[i]):
                cross_idx = i
                break

        self.assertIsNotNone(cross_idx, "expected a MACD bearish cross in the series")

        sig = evaluate(make_candles(closes[: cross_idx + 1]), cfg)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, "PUT")
        self.assertIn("MACD", sig.votes)

    def test_trend_filter_blocks_counter_trend(self):
        # Strong downtrend: a bullish MACD cross should be suppressed.
        closes = self._decline_then_rise()
        cfg = self._macd_only_config(use_trend=True, trend_ema_len=20)

        macd_line, signal_line = indicators.macd_series(closes, 12, 26, 9)
        cross_idx = None
        for i in range(1, len(closes)):
            if (macd_line[i - 1] is not None and macd_line[i] is not None
                    and signal_line[i - 1] is not None and signal_line[i] is not None
                    and macd_line[i - 1] <= signal_line[i - 1]
                    and macd_line[i] > signal_line[i]):
                cross_idx = i
                break
        self.assertIsNotNone(cross_idx)

        # The cross happens during the recovery; price is still below the
        # longer EMA, so the trend filter should block the CALL.
        sig = evaluate(make_candles(closes[: cross_idx + 1]), cfg)
        self.assertIsNone(sig)

    def test_insufficient_data_returns_none(self):
        cfg = self._macd_only_config()
        self.assertIsNone(evaluate(make_candles([1.0, 2.0]), cfg))


if __name__ == "__main__":
    unittest.main()
