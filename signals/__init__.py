"""Signal generation: technical indicators, the confluence engine, ranking."""

from .engine import (
    Candle, Readiness, Signal, SignalConfig, component_bars, evaluate, readiness,
    resample,
)
from .ranking import Candidate, describe, eligible, rank, select_best

__all__ = [
    "Candle", "Readiness", "Signal", "SignalConfig", "component_bars", "evaluate",
    "readiness", "resample", "Candidate", "describe", "eligible", "rank",
    "select_best",
]
