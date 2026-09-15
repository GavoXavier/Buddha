"""Configuration: environment -> app config + engine config.

Every value has a default, so ``FEED=simulated python main.py`` runs with no
``.env`` at all. The defaults describe the shortest useful cadence: **one signal
per minute on the major OTC forex pairs, for a one-minute expiry.** The bar
length is the dial that reconfigures it — ``CANDLE_PERIOD=300`` with
``EXPIRY=5m`` and ``SIGNAL_LEAD_SECONDS=300`` gives a signal at 16:00 for an
entry at 16:05 and an expiry at 16:10, one trade open at a time.

``load_config`` deliberately validates rather than trusting: a typo in the
expiry or a lead longer than the bar would otherwise show up as a bot that
silently sends nothing, which is the hardest kind of failure to diagnose at
2am. Anything that cannot work raises ``ConfigError`` at startup instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from engine.scheduler import Cadence
from market.universe import MODE_MAJOR, VALID_MODES
from signals.engine import SignalConfig
from telegram.sender import expiry_seconds

load_dotenv()


class ConfigError(Exception):
    """Raised for a configuration that cannot work (never for a missing one)."""


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _int(key: str, default: int) -> int:
    v = _get(key).strip()
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ConfigError(f"{key}={v!r} is not a whole number") from None


def _float(key: str, default: float) -> float:
    v = _get(key).strip()
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"{key}={v!r} is not a number") from None


def _bool(key: str, default: bool) -> bool:
    v = _get(key).strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- feed ---
    feed: str = "simulated"
    pocket_ssid: str = ""
    pocket_uid: int = 0
    pocket_is_demo: bool = True
    pocket_region: str = ""

    # --- universe ---
    assets: list[str] = field(default_factory=list)   # explicit override
    asset_mode: str = MODE_MAJOR
    otc_only: bool = True
    min_payout: int = 60

    # --- candles / timing ---
    candle_period: int = 60
    max_bars: int = 500
    stale_after: float = 120.0
    feed_stale_after: float = 90.0
    # A hole in the feed longer than this many bars makes the bars either side
    # of it two different series, and the older side is dropped rather than left
    # for an indicator window to read across. See market/aggregator.py.
    max_gap_bars: int = 5
    candle_store_dir: str = "candles"
    persist_candles: bool = True
    # A second copy of the same bars, kept far deeper than the live store, in its
    # own directory and read only by the replay tool. MAX_BARS is a memory
    # decision about the live buffer and *also* the ceiling on how much history a
    # backtest can ever have — the two are not the same question, and at 0.2
    # signals/hour the sample the dials need is three times what that ceiling
    # holds. Archiving decouples them: the engine keeps reading a small warm
    # buffer, and the history it has watched accumulates anyway. Nothing here
    # reaches the engine, so it cannot change a signal.
    archive_candles: bool = True
    # Empty means "beside CANDLE_STORE_DIR" (``candles`` -> ``candles-archive``),
    # resolved in ``load_config``. The archive is the same store read deeper, so
    # its location has to follow the store's rather than be a second absolute
    # path that can drift from it — and a caller that redirects the store (a test
    # into a temp directory, say) must not leave the archive writing into the
    # working tree, which is exactly what a fixed default did.
    candle_archive_dir: str = ""
    archive_bars: int = 20000

    # --- telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- cadence ---
    expiry: str = "1m"
    expiry_seconds: int = 60
    cooldown_seconds: int = 120
    lead_seconds: int = 10
    martingale_steps: int = 0
    max_consecutive_losses: int = 5
    breaker_pause_minutes: int = 15

    # --- automated execution (demo only) ---
    # Off by default: with it off the bot only signals, and nothing in
    # execution.py is ever constructed.
    auto_trade: bool = False
    trade_amount: float = 1.0

    # --- reporting ---
    # The clock every rendered time is printed in — messages, stats and the
    # journal. Fixed at UTC+3 for years because it was a constant; it is a
    # setting because a trade you place by hand lives or dies on the entry time
    # in the message matching the clock in front of you.
    market_tz_offset_hours: float = 3.0
    stats_path: str = "stats.json"
    legacy_stats_path: str = "winrate.json"
    # Per-trade record, for reconciling the bot's labels against the broker's.
    signal_journal_path: str = "signals.jsonl"
    journal_signals: bool = True
    # The markets this bot trades, held across restarts rather than re-derived
    # from the broker's live payout list on every start. Delete it to choose a
    # universe afresh — see ``main.resolve_universe`` for what re-choosing costs.
    universe_path: str = "universe.json"
    log_level: str = "INFO"

    signal: SignalConfig = field(default_factory=SignalConfig)

    # -- derived -------------------------------------------------------------
    def cadence(self) -> Cadence:
        return Cadence(
            period=self.candle_period,
            expiry_seconds=self.expiry_seconds,
            lead_seconds=self.lead_seconds,
            cooldown_seconds=self.cooldown_seconds,
            martingale_steps=self.martingale_steps,
            max_consecutive_losses=self.max_consecutive_losses,
            pause_minutes=self.breaker_pause_minutes,
            min_confidence=self.signal.min_confidence,
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def _signal_config() -> SignalConfig:
    """Engine settings. Defaults chosen for 1-minute bars on FX majors.

    Every window is counted in *bars*, never in seconds, so the same defaults
    on 5-minute bars describe five times the horizon: a 50-bar trend EMA is 50
    minutes at ``CANDLE_PERIOD=60`` and 4h10m at 300. They are not re-tuned for
    that — the dials were calibrated on 1-minute bars and would need a
    ``backtest.py`` run against 5-minute history to be re-measured.

    The MTF confirmation is resampled from this same series rather than
    fetched, so nothing depends on the broker's history.
    """
    return SignalConfig(
        bar_seconds=_int("CANDLE_PERIOD", 60),
        use_trend=_bool("USE_TREND", True),
        trend_ema_len=_int("TREND_EMA_LEN", 50),
        trend_slope_bars=_int("TREND_SLOPE_BARS", 5),
        trend_min_distance_pct=_float("TREND_MIN_DISTANCE_PCT", 0.05),
        trend_flat_min_score=_int("TREND_FLAT_MIN_SCORE", 3),
        use_mtf=_bool("USE_MTF", True),
        mtf_factor=_int("MTF_FACTOR", 5),
        mtf_ema_len=_int("MTF_EMA_LEN", 10),
        use_atr=_bool("USE_ATR", True),
        atr_len=_int("ATR_LEN", 14),
        atr_min_pct=_float("ATR_MIN_PCT", 0.0),
        atr_max_pct=_float("ATR_MAX_PCT", 0.0),
        use_rsi=_bool("USE_RSI", True),
        rsi_len=_int("RSI_LEN", 14),
        rsi_ob=_float("RSI_OB", 70.0),
        rsi_os=_float("RSI_OS", 30.0),
        use_macd=_bool("USE_MACD", True),
        macd_fast=_int("MACD_FAST", 12),
        macd_slow=_int("MACD_SLOW", 26),
        macd_signal=_int("MACD_SIGNAL", 9),
        use_bb=_bool("USE_BB", True),
        bb_len=_int("BB_LEN", 20),
        bb_mult=_float("BB_MULT", 2.0),
        use_stoch=_bool("USE_STOCH", True),
        stoch_k=_int("STOCH_K", 14),
        stoch_d=_int("STOCH_D", 3),
        stoch_smooth=_int("STOCH_SMOOTH", 3),
        use_sr=_bool("USE_SR", True),
        sr_swing_k=_int("SR_SWING_K", 2),
        sr_proximity_pct=_float("SR_PROXIMITY_PCT", 0.1),
        event_weight=_int("EVENT_WEIGHT", 2),
        state_weight=_int("STATE_WEIGHT", 1),
        min_score=_int("MIN_SCORE", 2),
        # Two directional indicators must agree: one alone is a coin flip.
        min_components=_int("MIN_COMPONENTS", 2),
        min_confidence=_float("MIN_CONFIDENCE", 0.0),
    )


def _validate(cfg: Config) -> None:
    if cfg.feed not in ("simulated", "pocket_option"):
        raise ConfigError(f"FEED={cfg.feed!r} — expected 'simulated' or 'pocket_option'")
    if cfg.asset_mode not in VALID_MODES:
        raise ConfigError(
            f"ASSET_MODE={cfg.asset_mode!r} — expected one of {', '.join(VALID_MODES)}")

    if cfg.candle_period < 5 or cfg.candle_period > 3600:
        raise ConfigError(f"CANDLE_PERIOD={cfg.candle_period} — expected 5..3600 seconds")
    if cfg.expiry_seconds <= 0:
        raise ConfigError(f"EXPIRY={cfg.expiry!r} — could not read a duration from it")
    if cfg.lead_seconds <= 0:
        raise ConfigError(f"SIGNAL_LEAD_SECONDS={cfg.lead_seconds} — must be positive")
    # A lead of exactly one bar is the long-lead cadence: signal 16:00, entry
    # 16:05. Longer than one bar is refused, and not for tidiness — the run
    # loop catches up with a single ``if``, so a lead further back than the
    # previous boundary leaves ``fire_at`` permanently in the past, sleeps no
    # seconds at all, and spins the same entry boundary out over and over.
    if cfg.lead_seconds > cfg.candle_period:
        raise ConfigError(
            f"SIGNAL_LEAD_SECONDS={cfg.lead_seconds} is longer than "
            f"CANDLE_PERIOD={cfg.candle_period} — a signal is sent at most one "
            "bar before its entry, because the bars it judges are the ones that "
            "have closed by the time it is sent")
    if cfg.stale_after <= cfg.candle_period:
        raise ConfigError(
            f"STALE_AFTER={cfg.stale_after:g} must exceed CANDLE_PERIOD="
            f"{cfg.candle_period} or healthy markets would be dropped mid-bar")
    if cfg.max_bars < 100:
        raise ConfigError(f"MAX_BARS={cfg.max_bars} is too small to hold a warm buffer")
    if cfg.archive_candles and cfg.archive_bars < cfg.max_bars:
        # An archive shallower than the live store would be a copy that holds
        # less than the original — the one arrangement with no reason to exist.
        raise ConfigError(
            f"ARCHIVE_BARS={cfg.archive_bars} is shallower than MAX_BARS="
            f"{cfg.max_bars}; the archive is the deeper copy and must not be")
    if cfg.max_gap_bars < 0:
        raise ConfigError("MAX_GAP_BARS cannot be negative")
    if cfg.min_payout < 0 or cfg.min_payout > 100:
        raise ConfigError(f"MIN_PAYOUT={cfg.min_payout} — expected a percentage 0..100")
    if cfg.martingale_steps < 0:
        raise ConfigError("MARTINGALE_STEPS cannot be negative")
    if cfg.signal.min_components < 1:
        raise ConfigError("MIN_COMPONENTS must be at least 1")
    if not 0.0 <= cfg.signal.min_confidence <= 1.0:
        raise ConfigError("MIN_CONFIDENCE must be between 0 and 1")
    if cfg.feed == "pocket_option":
        if not cfg.pocket_ssid:
            raise ConfigError("FEED=pocket_option needs POCKET_SSID in .env")
        if not cfg.pocket_uid:
            raise ConfigError("FEED=pocket_option needs POCKET_UID in .env")
    if cfg.telegram_bot_token and not cfg.telegram_chat_id:
        raise ConfigError("TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is not")
    if cfg.auto_trade:
        # The first of the three refusals that keep automated orders off a
        # funded account — this one fires before a socket is opened or the
        # session string is used, so a live login cannot be traded by accident.
        if cfg.feed != "pocket_option":
            raise ConfigError(
                f"AUTO_TRADE=1 needs FEED=pocket_option — there is nothing to "
                f"trade on FEED={cfg.feed!r}")
        if not cfg.pocket_is_demo:
            raise ConfigError(
                "AUTO_TRADE=1 is refused with POCKET_IS_DEMO=0: the bot places "
                "orders on demand with no human in the loop, and this build "
                "will not do that on a funded account. Set POCKET_IS_DEMO=1.")
        if cfg.trade_amount <= 0:
            raise ConfigError(f"TRADE_AMOUNT={cfg.trade_amount:g} must be positive")


def load_config() -> Config:
    assets = [a.strip() for a in _get("ASSETS").split(",") if a.strip()]
    expiry = _get("EXPIRY", "1m").strip() or "1m"
    try:
        seconds = expiry_seconds(expiry)
    except ValueError:
        raise ConfigError(
            f"EXPIRY={expiry!r} — expected a duration like '1m', '5m', '1h' or '90s'"
        ) from None

    cfg = Config(
        feed=_get("FEED", "simulated").strip().lower(),
        pocket_ssid=_get("POCKET_SSID"),
        pocket_uid=_int("POCKET_UID", 0),
        pocket_is_demo=_bool("POCKET_IS_DEMO", True),
        pocket_region=_get("POCKET_REGION").strip().lower(),
        assets=assets,
        asset_mode=_get("ASSET_MODE", MODE_MAJOR).strip().lower(),
        otc_only=_bool("OTC_ONLY", True),
        min_payout=_int("MIN_PAYOUT", 60),
        candle_period=_int("CANDLE_PERIOD", 60),
        max_bars=_int("MAX_BARS", 500),
        stale_after=_float("STALE_AFTER", 120.0),
        feed_stale_after=_float("FEED_STALE_AFTER", 90.0),
        max_gap_bars=_int("MAX_GAP_BARS", 5),
        candle_store_dir=_get("CANDLE_STORE_DIR", "candles"),
        persist_candles=_bool("PERSIST_CANDLES", True),
        archive_candles=_bool("ARCHIVE_CANDLES", True),
        candle_archive_dir=_get("CANDLE_ARCHIVE_DIR").strip(),
        archive_bars=_int("ARCHIVE_BARS", 20000),
        telegram_bot_token=_get("TELEGRAM_BOT_TOKEN").strip(),
        telegram_chat_id=_get("TELEGRAM_CHAT_ID").strip(),
        expiry=expiry,
        expiry_seconds=seconds,
        cooldown_seconds=_int("COOLDOWN_SECONDS", 120),
        lead_seconds=_int("SIGNAL_LEAD_SECONDS", 10),
        martingale_steps=_int("MARTINGALE_STEPS", 0),
        max_consecutive_losses=_int("MAX_CONSECUTIVE_LOSSES", 5),
        breaker_pause_minutes=_int("BREAKER_PAUSE_MINUTES", 15),
        auto_trade=_bool("AUTO_TRADE", False),
        trade_amount=_float("TRADE_AMOUNT", 1.0),
        market_tz_offset_hours=_float("MARKET_TZ_OFFSET_HOURS", 3.0),
        stats_path=_get("STATS_PATH", "stats.json"),
        legacy_stats_path=_get("LEGACY_STATS_PATH", "winrate.json"),
        signal_journal_path=_get("SIGNAL_JOURNAL_PATH", "signals.jsonl"),
        universe_path=_get("UNIVERSE_PATH", "universe.json"),
        journal_signals=_bool("JOURNAL_SIGNALS", True),
        log_level=_get("LOG_LEVEL", "INFO").strip().upper(),
        signal=_signal_config(),
    )
    _validate(cfg)
    return cfg


def derived_archive_dir(store_dir: str) -> str:
    """Where the archive lives when it is not named: beside the store it copies.

    ``candles`` -> ``candles-archive``, dropping only a trailing separator, so
    ``data/candles/`` gives ``data/candles-archive/``. Derived here rather than
    defaulted in the dataclass because the answer depends on ``CANDLE_STORE_DIR``,
    and it is derived at the point of use rather than stored so that a ``Config``
    built by hand (a test's) resolves the same way a loaded one does — the
    alternative wrote test data into the working tree.
    """
    trimmed = str(store_dir).rstrip("/\\")
    return f"{trimmed}-archive" if trimmed else "candles-archive"
