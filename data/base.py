"""DataFeed abstraction.

A feed does three things:

* publishes the live asset list with broker metadata (payout, digits, ...)
* subscribes to a set of symbols and pushes every price update to one tick
  handler — ``handler(symbol, timestamp, price)``
* optionally exposes broker history, which is **diagnostic only**: for OTC
  assets the history endpoint and the tick stream are unrelated price paths,
  so candles for the engine are aggregated from ticks (see ``market``).

Implementations: Pocket Option (live) and SimulatedFeed (offline).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

from market.universe import AssetMeta
from signals.engine import Candle

# (symbol, epoch seconds, price)
TickHandler = Callable[[str, float, float], None]


class DataFeed(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection / authenticate."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection."""

    @abstractmethod
    async def asset_meta(self) -> dict[str, AssetMeta]:
        """Live metadata for every tradeable symbol, keyed by symbol."""

    @abstractmethod
    async def subscribe(self, symbols: Sequence[str]) -> None:
        """Start streaming the given symbols to the tick handler."""

    @abstractmethod
    def set_tick_handler(self, handler: TickHandler) -> None:
        """Register the single callback that receives every price update."""

    async def get_candles(self, symbol: str, period: int, count: int) -> list[Candle]:
        """Broker history — diagnostic only, never the basis for signals."""
        raise NotImplementedError

    @property
    def last_tick_at(self) -> float:
        """Epoch seconds of the most recent tick (0 if none yet)."""
        return getattr(self, "_last_tick_at", 0.0)

    @property
    def is_connected(self) -> bool:
        """Whether the transport is still up.

        The supervisor watches this as well as tick silence, because a dead
        socket is knowable at once while tick silence has to be waited out: on
        2026-09-15 engineio gave up on the connection ~8 seconds after the last
        tick, but the watchdog only reacted 95 seconds after it, so most of
        every flap cycle was spent in a session that could not have worked.
        Feeds that cannot drop (the simulator) are always connected.
        """
        return True
