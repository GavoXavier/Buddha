"""Market data plumbing: tick-built candles, the asset universe, persistence.

The important idea in this package is that candles are built from *ticks this
process received* (``aggregator``), never from broker history — for OTC assets
those are unrelated price paths. ``store`` persists what was watched so a
restart resumes warm, ``universe`` decides which markets are tradeable, and
``clock`` makes the whole minute cadence testable.
"""

from .aggregator import CandleSeries, MarketState
from .clock import Clock, RealClock, VirtualClock
from .store import CandleStore
from .universe import (
    MODE_ALL, MODE_FOREX, MODE_MAJOR, AssetMeta, describe_skipped, format_universe,
    is_currency_pair, is_major_pair, matches_mode, select_assets,
)

__all__ = [
    "CandleSeries", "MarketState", "Clock", "RealClock", "VirtualClock",
    "CandleStore", "AssetMeta", "MODE_MAJOR", "MODE_FOREX", "MODE_ALL",
    "is_currency_pair", "is_major_pair", "matches_mode", "select_assets",
    "describe_skipped", "format_universe",
]
