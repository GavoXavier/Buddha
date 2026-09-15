"""Pure-Python technical indicators.

Each function returns a list aligned with the input (same length), using
``None`` for the warm-up period where the indicator is not yet defined.
No external dependencies (numpy/pandas) so the engine runs anywhere and is
trivially unit-testable.

Values are computed to match TradingView's Pine Script semantics where it
matters (e.g. population standard deviation for Bollinger Bands, Wilder's
smoothing for RSI).
"""

from __future__ import annotations

from typing import Optional, Sequence


def sma_series(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Simple moving average. ``None`` for the first ``period - 1`` bars."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0:
        return out
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(v is None for v in window):
            continue
        out[i] = sum(window) / period  # type: ignore[arg-type]
    return out


def ema_series(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Exponential moving average, seeded with an SMA of the first ``period``
    valid values. Leading ``None`` values are preserved."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0:
        return out

    # Collect the contiguous run of valid values (indicators like MACD produce
    # leading Nones then a continuous tail).
    valid = [v for v in values if v is not None]
    if len(valid) < period:
        return out

    k = 2.0 / (period + 1)
    ema_vals: list[Optional[float]] = [None] * len(valid)
    ema_vals[period - 1] = sum(valid[:period]) / period
    for i in range(period, len(valid)):
        ema_vals[i] = valid[i] * k + ema_vals[i - 1] * (1 - k)  # type: ignore[operator]

    # Map the EMA values back onto the original (possibly None-padded) list.
    idx = 0
    for i in range(n):
        if values[i] is not None:
            out[i] = ema_vals[idx]
            idx += 1
    return out


def rsi_series(closes: Sequence[float], period: int = 14) -> list[Optional[float]]:
    """Relative Strength Index using Wilder's smoothing."""
    n = len(closes)
    out: list[Optional[float]] = [None] * n
    if n <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains += max(diff, 0.0)
        losses += max(-diff, 0.0)

    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, n):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)

    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd_series(
    closes: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    """MACD line and signal line (EMA of the MACD line)."""
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)

    n = len(closes)
    macd_line: list[Optional[float]] = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]  # type: ignore[operator]

    signal_line = ema_series(macd_line, signal)
    return macd_line, signal_line


def bollinger_series(
    closes: Sequence[float],
    period: int = 20,
    mult: float = 2.0,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Bollinger Bands: (basis, upper, lower). Uses population stddev to match
    Pine Script's ``ta.stdev``."""
    n = len(closes)
    basis = sma_series(closes, period)
    upper: list[Optional[float]] = [None] * n
    lower: list[Optional[float]] = [None] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        sd = variance ** 0.5
        upper[i] = mean + mult * sd
        lower[i] = mean - mult * sd

    return basis, upper, lower


def atr_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[Optional[float]]:
    """Average True Range (Wilder's smoothing).

    ``None`` until ``period`` true ranges exist, so the first defined value is
    at index ``period``. Used to reject dead (no-range) and spiking minutes.
    """
    n = len(closes)
    out: list[Optional[float]] = [None] * n
    if period <= 0 or n <= period:
        return out

    # true_ranges[i - 1] is the true range of bar i (needs bar i-1's close).
    true_ranges: list[float] = []
    for i in range(1, n):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    atr = sum(true_ranges[:period]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + true_ranges[i - 1]) / period
        out[i] = atr
    return out


def stochastic_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    """Stochastic oscillator: (slow %K, %D). %K is smoothed with an SMA."""
    n = len(closes)
    raw_k: list[Optional[float]] = [None] * n

    for i in range(k_period - 1, n):
        hh = max(highs[i - k_period + 1 : i + 1])
        ll = min(lows[i - k_period + 1 : i + 1])
        if hh == ll:
            raw_k[i] = 50.0
        else:
            raw_k[i] = 100.0 * (closes[i] - ll) / (hh - ll)

    k = sma_series(raw_k, smooth)
    d = sma_series(k, d_period)
    return k, d
