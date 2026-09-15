"""Market data feeds: abstraction + Pocket Option (live) + simulated (offline)."""

from .base import DataFeed, TickHandler
from .simulated import SimulatedFeed

__all__ = ["DataFeed", "TickHandler", "SimulatedFeed"]
