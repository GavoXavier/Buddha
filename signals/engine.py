"""Confluence-scoring signal engine.

Each enabled indicator contributes a weighted vote on the current bar:

* an **event** vote (a crossover on this bar) — precise, worth ``event_weight``
* a **state** vote (RSI oversold, MACD histogram negative, ...) — always
  present, worth ``state_weight``

Events outrank states, and a bar casts at most one vote per indicator (so an
event never double-counts with the state it just produced). The direction with
the higher weighted score wins; a tie is ambiguous and produces no signal.

On top of the votes sit **gates** — conditions that reject a direction outright
rather than scoring it: the trend EMA (with optional slope and distance
strength), a higher-timeframe EMA, an ATR volatility band, and the
support/resistance proximity filter.

**Graceful degradation.** A gate or vote whose indicator does not yet have
enough bars is *skipped*, not fatal: at startup the fast indicators (RSI,
Bollinger, Stochastic) come online after ~20 bars, and MACD / trend / MTF join
as the buffer grows. ``readiness()`` reports exactly what is live and how many
bars are still needed, so the bot can say "warming up: 12/60 bars" instead of
silently staying quiet. ``min_components`` keeps this honest: at least that
many *directional* indicators (RSI / MACD / BB / Stoch — the gates do not
count) must have real data before any signal can fire.

The engine is a pure function of the candle buffer — no state, so it is
trivially unit-testable. Cooldowns and cadence live in the caller.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import indicators


@dataclass
class Candle:
    time: float  # epoch seconds of the bar's OPEN
    open: float
    high: float
    low: float
    close: float


@dataclass
class SignalConfig:
    # Bar size in seconds — needed to build higher timeframes from this series.
    bar_seconds: int = 60

    # --- Trend filter (EMA on the traded timeframe) ---
    use_trend: bool = True
    trend_ema_len: int = 50
    # The EMA must slope in the trend direction over this many bars, and price
    # must be at least this % away from it, or the market reads as "flat"
    # (ranging). Set either to 0 to disable that half of the check.
    trend_slope_bars: int = 5
    trend_min_distance_pct: float = 0.05
    # A trending market vetoes the counter-trend direction outright. A *flat*
    # one is ambiguous rather than hostile, so instead of vetoing both sides it
    # demands a stronger reading: this many points of agreement. Raise it to
    # trade ranges more cautiously, set it high (e.g. 9) to sit them out
    # entirely, or 0 to ignore the flat reading and trade them freely.
    #
    # This matters more than it looks: FX majors are correlated, so when the
    # dollar goes quiet *every* pair reads flat at once and an outright veto
    # leaves the bot with little to do. Demanding a stronger score keeps it in
    # the market for the setups that are clear even without a trend.
    trend_flat_min_score: int = 3

    # --- Higher-timeframe confirmation (built from this series) ---
    use_mtf: bool = True
    mtf_factor: int = 5          # 5 x 1m bars = 5m bars
    mtf_ema_len: int = 10

    # --- Volatility band (rejects dead and spiking minutes) ---
    use_atr: bool = True
    atr_len: int = 14
    atr_min_pct: float = 0.0     # 0 disables the floor
    atr_max_pct: float = 0.0     # 0 disables the ceiling

    # --- RSI ---
    use_rsi: bool = True
    rsi_len: int = 14
    rsi_ob: float = 70.0
    rsi_os: float = 30.0

    # --- MACD ---
    use_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- Bollinger Bands ---
    use_bb: bool = True
    bb_len: int = 20
    bb_mult: float = 2.0

    # --- Stochastic ---
    use_stoch: bool = True
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # --- Support/resistance pullback filter ---
    use_sr: bool = True
    sr_swing_k: int = 2
    sr_proximity_pct: float = 0.1

    # --- Scoring ---
    event_weight: int = 2
    state_weight: int = 1
    min_score: int = 2
    # Vote components that must have real data before a signal may fire.
    min_components: int = 1
    min_confidence: float = 0.0

    def dials(self) -> dict:
        """The settings that differ from this dataclass's defaults.

        Only what differs, because that is what a reader of a journalled signal
        needs: the shipped values are in the file, so a line that says
        ``{"use_trend": False}`` is reporting "the trend veto was off" and not
        forty identical rows. Two configurations with the same deltas are the same
        strategy, which is exactly what ``fingerprint`` relies on.
        """
        base = SignalConfig()
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(base, name)}

    def fingerprint(self) -> str:
        """A short, stable name for this configuration, for grouping a record.

        Deliberately not ``hash()``: its seed is randomised per process, so the
        same configuration would get a different name in every run and a record
        grouped by it would be split for no reason. This one is a digest of the
        canonical JSON of ``dials()``, so it depends only on the values.

        It is a grouping key, not a display string — the deltas themselves are
        recorded beside it, so nothing ever has to invert this.
        """
        canon = json.dumps(self.dials(), sort_keys=True, default=str)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


@dataclass
class Signal:
    direction: str  # "CALL" or "PUT"
    score: int
    votes: list[str] = field(default_factory=list)
    price: float = 0.0
    time: float = 0.0
    # How clean the setup is, 0..1 — used to rank assets against each other.
    confidence: float = 0.0
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    atr_pct: Optional[float] = None
    trend: Optional[str] = None
    mtf: Optional[str] = None


@dataclass
class Readiness:
    """What the engine can currently see, for warm-up reporting."""

    bars: int
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    bars_needed: int = 0

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def progress(self) -> float:
        if self.bars_needed <= 0:
            return 1.0
        return min(1.0, self.bars / self.bars_needed)


def _crossed_up(prev_a: Optional[float], cur_a: Optional[float],
                prev_b: Optional[float], cur_b: Optional[float]) -> bool:
    """True if series A crossed from <= B to > B on the current bar."""
    if None in (prev_a, cur_a, prev_b, cur_b):
        return False
    return prev_a <= prev_b and cur_a > cur_b  # type: ignore[operator]


def _crossed_down(prev_a: Optional[float], cur_a: Optional[float],
                  prev_b: Optional[float], cur_b: Optional[float]) -> bool:
    """True if series A crossed from >= B to < B on the current bar."""
    if None in (prev_a, cur_a, prev_b, cur_b):
        return False
    return prev_a >= prev_b and cur_a < cur_b  # type: ignore[operator]


def _recent_swing_low(lows: Sequence[float], k: int) -> Optional[float]:
    """Most recent swing low (local min over ``k`` bars each side), excluding
    the current bar so we only react to levels that have already formed."""
    n = len(lows)
    for i in range(n - 1 - k, k - 1, -1):
        if lows[i] == min(lows[i - k:i + k + 1]):
            return lows[i]
    return None


def _recent_swing_high(highs: Sequence[float], k: int) -> Optional[float]:
    """Most recent swing high (local max over ``k`` bars each side), excluding
    the current bar so we only react to levels that have already formed."""
    n = len(highs)
    for i in range(n - 1 - k, k - 1, -1):
        if highs[i] == max(highs[i - k:i + k + 1]):
            return highs[i]
    return None


def resample(candles: Sequence[Candle], factor: int, bar_seconds: int) -> list[Candle]:
    """Aggregate ``candles`` into higher-timeframe bars of ``factor`` bars each.

    Only *complete* groups are returned, so a half-formed higher-timeframe bar
    never leaks into an indicator. Bars are grouped by their open time, so gaps
    in the source series do not shift the alignment.
    """
    if factor <= 1 or bar_seconds <= 0:
        return list(candles)
    span = factor * bar_seconds
    groups: dict[int, list[Candle]] = {}
    for c in candles:
        groups.setdefault(int(c.time // span), []).append(c)

    out: list[Candle] = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) < factor:
            continue  # incomplete (or gap-damaged) higher-timeframe bar
        out.append(Candle(
            time=key * span,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
        ))
    return out


def component_bars(config: SignalConfig) -> dict[str, int]:
    """Bars each enabled component needs before it can vote or gate."""
    need: dict[str, int] = {}
    if config.use_rsi:
        need["RSI"] = config.rsi_len + 2
    if config.use_macd:
        need["MACD"] = config.macd_slow + config.macd_signal + 1
    if config.use_bb:
        need["BB"] = config.bb_len + 1
    if config.use_stoch:
        need["Stoch"] = config.stoch_k + config.stoch_smooth + config.stoch_d - 2
    if config.use_trend:
        need["trend"] = config.trend_ema_len + max(config.trend_slope_bars, 0) + 1
    if config.use_mtf:
        need["MTF"] = config.mtf_ema_len * config.mtf_factor + config.mtf_factor
    if config.use_atr:
        need["ATR"] = config.atr_len + 2
    return need


# The components that actually cast votes on direction. The gates (trend, MTF,
# ATR) are counted separately: "two indicators agree" must mean two *directional*
# readings, not one reading plus a volatility measurement.
VOTE_COMPONENTS = ("RSI", "MACD", "BB", "Stoch")


def vote_components(analysis: "_Analysis") -> list[str]:
    return [c for c in analysis.available if c in VOTE_COMPONENTS]


@dataclass
class _Analysis:
    """Everything the engine can read off the current bar."""

    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    bull: int = 0
    bear: int = 0
    bull_votes: list[str] = field(default_factory=list)
    bear_votes: list[str] = field(default_factory=list)
    trend: Optional[str] = None      # "up" | "down" | "flat" | None (not ready)
    mtf: Optional[str] = None        # "up" | "down" | None (not ready)
    atr_pct: Optional[float] = None


def _analyze(candles: Sequence[Candle], config: SignalConfig) -> _Analysis:
    """Score the latest bar. Never raises on thin data — reports what is missing."""
    a = _Analysis()
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    def vote(name: str, direction: str, weight: int) -> None:
        if direction == "CALL":
            a.bull += weight
            a.bull_votes.append(name)
        else:
            a.bear += weight
            a.bear_votes.append(name)

    # --- RSI ---------------------------------------------------------------
    if config.use_rsi:
        rsi = indicators.rsi_series(closes, config.rsi_len)
        if rsi[-1] is None or rsi[-2] is None:
            a.missing.append("RSI")
        else:
            a.available.append("RSI")
            if _crossed_up(rsi[-2], rsi[-1], config.rsi_os, config.rsi_os):
                vote("RSI", "CALL", config.event_weight)
            elif _crossed_down(rsi[-2], rsi[-1], config.rsi_ob, config.rsi_ob):
                vote("RSI", "PUT", config.event_weight)
            elif rsi[-1] < config.rsi_os:
                vote("RSI", "CALL", config.state_weight)
            elif rsi[-1] > config.rsi_ob:
                vote("RSI", "PUT", config.state_weight)

    # --- MACD --------------------------------------------------------------
    if config.use_macd:
        macd_line, signal_line = indicators.macd_series(
            closes, config.macd_fast, config.macd_slow, config.macd_signal
        )
        if macd_line[-1] is None or signal_line[-1] is None:
            a.missing.append("MACD")
        else:
            a.available.append("MACD")
            if _crossed_up(macd_line[-2], macd_line[-1], signal_line[-2], signal_line[-1]):
                vote("MACD", "CALL", config.event_weight)
            elif _crossed_down(macd_line[-2], macd_line[-1], signal_line[-2], signal_line[-1]):
                vote("MACD", "PUT", config.event_weight)
            elif macd_line[-1] > signal_line[-1]:
                vote("MACD", "CALL", config.state_weight)
            elif macd_line[-1] < signal_line[-1]:
                vote("MACD", "PUT", config.state_weight)

    # --- Bollinger Bands ---------------------------------------------------
    if config.use_bb:
        _, bb_upper, bb_lower = indicators.bollinger_series(
            closes, config.bb_len, config.bb_mult
        )
        if bb_upper[-1] is None or bb_lower[-1] is None or bb_lower[-2] is None:
            a.missing.append("BB")
        else:
            a.available.append("BB")
            if _crossed_up(closes[-2], closes[-1], bb_lower[-2], bb_lower[-1]):
                vote("BB", "CALL", config.event_weight)
            elif _crossed_down(closes[-2], closes[-1], bb_upper[-2], bb_upper[-1]):
                vote("BB", "PUT", config.event_weight)
            elif closes[-1] <= bb_lower[-1]:
                vote("BB", "CALL", config.state_weight)
            elif closes[-1] >= bb_upper[-1]:
                vote("BB", "PUT", config.state_weight)

    # --- Stochastic --------------------------------------------------------
    if config.use_stoch:
        stoch_k, stoch_d = indicators.stochastic_series(
            highs, lows, closes, config.stoch_k, config.stoch_d, config.stoch_smooth
        )
        if stoch_k[-1] is None or stoch_d[-1] is None:
            a.missing.append("Stoch")
        else:
            a.available.append("Stoch")
            if (_crossed_up(stoch_k[-2], stoch_k[-1], stoch_d[-2], stoch_d[-1])
                    and stoch_k[-1] < 20):
                vote("Stoch", "CALL", config.event_weight)
            elif (_crossed_down(stoch_k[-2], stoch_k[-1], stoch_d[-2], stoch_d[-1])
                    and stoch_k[-1] > 80):
                vote("Stoch", "PUT", config.event_weight)
            elif stoch_k[-1] < 20:
                vote("Stoch", "CALL", config.state_weight)
            elif stoch_k[-1] > 80:
                vote("Stoch", "PUT", config.state_weight)

    # --- Trend gate --------------------------------------------------------
    if config.use_trend:
        trend_ema = indicators.ema_series(closes, config.trend_ema_len)
        ema_now = trend_ema[-1]
        if ema_now is None:
            a.missing.append("trend")
        else:
            up = closes[-1] > ema_now
            down = closes[-1] < ema_now
            if config.trend_slope_bars > 0:
                ema_prev = (trend_ema[-1 - config.trend_slope_bars]
                            if len(trend_ema) > config.trend_slope_bars else None)
                if ema_prev is None:
                    up = down = False
                else:
                    up = up and ema_now > ema_prev
                    down = down and ema_now < ema_prev
            if config.trend_min_distance_pct > 0:
                dist = (closes[-1] - ema_now) / ema_now
                min_dist = config.trend_min_distance_pct / 100.0
                up = up and dist > min_dist
                down = down and dist < -min_dist
            a.available.append("trend")
            a.trend = "up" if up else "down" if down else "flat"

    # --- Higher-timeframe gate --------------------------------------------
    if config.use_mtf:
        htf = resample(candles, config.mtf_factor, config.bar_seconds)
        if len(htf) < config.mtf_ema_len:
            a.missing.append("MTF")
        else:
            htf_ema = indicators.ema_series([c.close for c in htf], config.mtf_ema_len)
            if htf_ema[-1] is None:
                a.missing.append("MTF")
            else:
                a.available.append("MTF")
                a.mtf = "up" if htf[-1].close > htf_ema[-1] else "down"

    # --- Volatility gate --------------------------------------------------
    if config.use_atr:
        atr = indicators.atr_series(highs, lows, closes, config.atr_len)
        if atr[-1] is None or closes[-1] <= 0:
            a.missing.append("ATR")
        else:
            a.available.append("ATR")
            a.atr_pct = atr[-1] / closes[-1] * 100.0

    return a


def readiness(candles: Sequence[Candle], config: SignalConfig) -> Readiness:
    """Report which components have enough data on this buffer."""
    need = component_bars(config)
    r = Readiness(bars=len(candles), bars_needed=max(need.values(), default=0))
    if len(candles) < 2:
        r.missing = sorted(need)
        return r
    analysis = _analyze(candles, config)
    r.available = analysis.available
    r.missing = analysis.missing
    return r


def _confidence(analysis: _Analysis, score: int, opponent: int,
                config: SignalConfig) -> float:
    """Rank a setup 0..1 so the best of many assets can be chosen each minute.

    Agreement carries most of the weight; higher-timeframe agreement and a live
    volatility reading refine it. Opposing votes discount the result, so a
    clean 2-of-2 beats a noisy 3-of-5.
    """
    components = len(vote_components(analysis))
    if components <= 0:
        return 0.0
    ceiling = components * config.event_weight
    agreement = max(0.0, (score - 0.5 * opponent) / ceiling)
    mtf_term = 0.5 if analysis.mtf is None else (1.0 if analysis.mtf else 0.0)
    vol_term = 0.5 if analysis.atr_pct is None else 1.0
    return max(0.0, min(1.0, 0.65 * agreement + 0.25 * mtf_term + 0.10 * vol_term))


def evaluate(candles: Sequence[Candle], config: SignalConfig) -> Optional[Signal]:
    """Evaluate the latest bar of ``candles`` and return a Signal, or None.

    The buffer must contain closed bars only, except that the caller may append
    a snapshot of the forming bar as the last element (that is how a signal is
    produced a few seconds before the bar closes, so it can be acted on).
    """
    if len(candles) < 2:
        return None

    a = _analyze(candles, config)
    last = candles[-1]

    # Not enough independent evidence to judge anything.
    if len(vote_components(a)) < config.min_components:
        return None
    # A tie has no directional edge — skip rather than guess.
    if a.bull == a.bear:
        return None

    if a.bull > a.bear:
        direction, score, opponent = "CALL", a.bull, a.bear
        votes = a.bull_votes
    else:
        direction, score, opponent = "PUT", a.bear, a.bull
        votes = a.bear_votes

    if score < config.min_score:
        return None

    # --- Gates: reject a direction outright ---
    against_trend = ((direction == "CALL" and a.trend == "down")
                     or (direction == "PUT" and a.trend == "up"))
    if against_trend:
        return None
    # No trend is not the same as the wrong trend: demand better evidence
    # rather than refusing to trade at all.
    if a.trend == "flat" and score < config.trend_flat_min_score:
        return None

    if direction == "CALL":
        if a.mtf == "down":
            return None
    else:
        if a.mtf == "up":
            return None

    if a.atr_pct is not None:
        if config.atr_min_pct > 0 and a.atr_pct < config.atr_min_pct:
            return None
        if config.atr_max_pct > 0 and a.atr_pct > config.atr_max_pct:
            return None

    if config.use_sr:
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        if direction == "CALL":
            support = _recent_swing_low(lows, config.sr_swing_k)
            if support is None or (last.close - support) / support > config.sr_proximity_pct / 100.0:
                return None
        else:
            resistance = _recent_swing_high(highs, config.sr_swing_k)
            if resistance is None or (resistance - last.close) / resistance > config.sr_proximity_pct / 100.0:
                return None

    confidence = _confidence(a, score, opponent, config)
    if confidence < config.min_confidence:
        return None

    return Signal(
        direction=direction,
        score=score,
        votes=votes,
        price=last.close,
        time=last.time,
        confidence=confidence,
        available=list(a.available),
        missing=list(a.missing),
        atr_pct=a.atr_pct,
        trend=a.trend,
        mtf=a.mtf,
    )
