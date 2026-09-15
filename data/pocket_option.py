"""Pocket Option data feed (live).

Built on the ``pocket-option`` async SDK (v0.4.0). Auth uses your SSID + UID,
grabbed from the browser DevTools (see README).

The SDK's flow:
  1. ``default_init`` wires up auth-on-connect + a candle storage.
  2. ``connect(url)`` opens the Socket.IO websocket; on connect the SDK sends
     the ``auth`` event automatically.
  3. ``wait_for_authorization`` blocks until the server confirms auth.

All price updates arrive on ``updateStream`` (``update_close_value``) and are
handed to a single tick handler, which aggregates them into candles.

``get_candles`` is kept for diagnostics and is deliberately *not* used to warm
up indicators: measured against this account, the history endpoint and the live
tick stream disagree about price level by 6-18 pips over the same minutes for
OTC symbols (and drift apart), so history is not the series that trades. See
the README.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional, Sequence

from market.universe import AssetMeta
from signals.engine import Candle

from .base import DataFeed, TickHandler

log = logging.getLogger("pocket.feed")


class PocketOptionFeed(DataFeed):
    def __init__(self, session: str, uid: int, is_demo: bool = True,
                 region: Optional[str] = None, is_fast_history: bool = True,
                 on_client: Optional[Callable[[object], None]] = None) -> None:
        self.session = session
        self.uid = uid
        self.is_demo = is_demo
        self.region = region  # name of a Regions member, e.g. "europa"
        self.is_fast_history = is_fast_history
        # Called once the client exists and its plumbing is initialised, but
        # before the socket connects: the server pushes deal history during auth,
        # so anything that wants to receive it has to be subscribed by then.
        self._on_client = on_client
        self._client = None
        self._http_session = None
        self._handler: Optional[TickHandler] = None
        self._subscribed: set[str] = set()
        self._last_tick_at: float = 0.0
        self._ticks_seen: int = 0
        # Pending history requests, keyed by asset symbol, so concurrent
        # get_candles() calls each receive their own response.
        self._history_waiters: dict[str, list[asyncio.Future]] = {}
        # Cap concurrent history requests: firing 100+ at once can overwhelm
        # the server and break the Socket.IO connection.
        self._history_semaphore = asyncio.Semaphore(10)

    # -- lifecycle -----------------------------------------------------------
    def set_tick_handler(self, handler: TickHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        import aiohttp
        from pocket_option import PocketOptionClient
        from pocket_option.contrib.default_init import default_init
        from pocket_option.models import AuthorizationData

        # Use aiohttp's threaded resolver: the default aiodns (c-ares) can fail
        # to contact DNS servers on some Windows setups.
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        self._http_session = aiohttp.ClientSession(connector=connector)

        # The SDK's own reconnect is switched off, because this supervisor is
        # the reconnector. Left on, it fights for the same seat: when the socket
        # dropped, the SDK would silently reconnect the *old* client — a fresh
        # session on the same SSID — while the watchdog was already tearing the
        # session down, so the server briefly held two sessions for one account
        # and the one just built was the one it refused. Observed as
        # "ConnectionError: One or more namespaces failed to connect" on the
        # reconnect that followed a flap (2026-09-15 21:47:20).
        self._client = PocketOptionClient(logger=False, reconnection=False,
                                          http_session=self._http_session)
        self._client.on.load_history_period_fast(self._on_history)
        self._client.on.update_close_value(self._on_ticks)
        # Clear any stale waiters left over from a previous (broken) connection.
        self._history_waiters.clear()
        self._subscribed.clear()

        auth = AuthorizationData(
            session=self.session,
            is_demo=1 if self.is_demo else 0,
            uid=self.uid,
            platform=1,
            is_fast_history=self.is_fast_history,
            is_optimized=True,
        )
        default_init(self._client, authorization=auth, sub_assets=[], sub_period=60)

        if self._on_client is not None:
            self._on_client(self._client)

        await self._client.connect(self._region().value)
        await self._client.wait_for_authorization(timeout=10)
        log.info("connected to %s", self._region().value)

    @property
    def client(self):
        """The live ``PocketOptionClient``, or None when not connected."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Whether the Socket.IO transport is still up.

        Checked on both layers because they can disagree: the Engine.IO socket
        dies first and the namespaces above it are only torn down afterwards.

        The two background loops are checked as well, because a socket can be
        open and dead at once. On 2026-09-15 the write loop gave up at 21:46:01
        after thirty seconds with nothing to send, and ``state`` still read
        "connected" forty-five seconds later: the server had gone quiet, the
        socket had not yet closed, and nothing could be sent over it — no
        order, no pong to the ping that would have kept it open. Waiting for
        the socket to notice costs the rest of that window, so a finished loop
        ends the session instead. Engine.IO clears both attributes when it
        resets, and only ever finishes them on a connection that is already
        over, so a missing task is not evidence of anything.

        Anything unexpected in the SDK's shape reads as "connected" — this is a
        tripwire alongside the tick watchdog, not a replacement for it, so
        failing open leaves the existing protection in place rather than
        reconnecting a session that was working.
        """
        client = self._client
        if client is None:
            return False
        sio = getattr(client, "sio", None)
        if sio is None:
            return True
        eio = getattr(sio, "eio", None)
        if eio is not None:
            if getattr(eio, "state", "connected") != "connected":
                return False
            for name in ("write_loop_task", "read_loop_task"):
                task = getattr(eio, name, None)
                if task is not None and task.done():
                    return False
        return bool(getattr(sio, "connected", True))

    def _region(self):
        from pocket_option.constants import Regions

        if not self.region:
            return Regions.DEMO if self.is_demo else Regions.EUROPA
        try:
            return Regions[self.region.upper()]
        except KeyError:
            names = ", ".join(r.name.lower() for r in Regions)
            raise ValueError(
                f"POCKET_REGION={self.region!r} is not a region — expected one of {names}"
            ) from None

    async def close(self) -> None:
        """Let go of the transport, whatever the transport does about it.

        A failed disconnect does not change what happens next — the client is
        dropped either way, so the process can still exit — which is why it is
        swallowed. Logged at debug rather than warning for the same reason, and
        logged at all so that "the socket took three seconds to let go" is
        answerable from the log instead of only from a stack trace that never
        appears.
        """
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:
                log.debug("disconnect raised on close, dropping the client anyway: %r",
                          exc)
            self._client = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as exc:
                log.debug("closing the HTTP session raised, dropping it anyway: %r",
                          exc)
            self._http_session = None
        self._subscribed.clear()

    # -- assets --------------------------------------------------------------
    async def asset_meta(self) -> dict[str, AssetMeta]:
        """Live metadata for every symbol, including current payout."""
        if self._client is None:
            return {}
        items = []
        for _ in range(50):  # up to ~5s: updateAssets lands shortly after auth
            items = await self._client.assets.get_assets()
            if items:
                break
            await asyncio.sleep(0.1)

        metas: dict[str, AssetMeta] = {}
        for item in items:
            symbol = item.asset.value
            metas[symbol] = AssetMeta(
                symbol=symbol,
                payout=int(item.payout),
                digits=int(item.digits),
                is_otc=bool(item.is_otc),
                active=bool(item.active),
            )
        return metas

    # -- streaming -----------------------------------------------------------
    async def subscribe(self, symbols: Sequence[str]) -> None:
        from pocket_option.models import Asset

        if self._client is None:
            raise RuntimeError("feed not connected")
        for symbol in symbols:
            if symbol in self._subscribed:
                continue
            await self._client.emit.subscribe_to_asset(Asset(symbol))
            self._subscribed.add(symbol)
        log.info("subscribed to %d symbols", len(self._subscribed))

    def _on_ticks(self, items) -> None:
        """Route every price update to the tick handler."""
        handler = self._handler
        self._ticks_seen += len(items)
        for item in items:
            ts = float(item.timestamp)
            if ts > self._last_tick_at:
                self._last_tick_at = ts
            if handler is not None:
                handler(item.asset.value, ts, float(item.value))

    @property
    def ticks_seen(self) -> int:
        return self._ticks_seen

    # -- history (diagnostic only) ------------------------------------------
    def _on_history(self, data) -> None:
        """Route a loadHistoryPeriodFast response to the matching waiter."""
        asset = data.asset.value
        waiters = self._history_waiters.pop(asset, [])
        for fut in waiters:
            if not fut.done():
                fut.set_result(data)

    async def get_candles(self, asset: str, period: int, count: int) -> list[Candle]:
        """Broker history. Diagnostic only — NOT the traded price path."""
        from pocket_option.models import Asset, LoadHistoryPeriodRequest

        if self._client is None:
            raise RuntimeError("feed not connected")

        asset_enum = Asset(asset)
        data = None
        for attempt in range(3):
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self._history_waiters.setdefault(asset, []).append(fut)
            try:
                async with self._history_semaphore:
                    await self._client.emit.load_history_period(
                        LoadHistoryPeriodRequest(
                            asset=asset_enum,
                            index=None,
                            time=time.time(),
                            # offset is a time span in seconds, not a candle count.
                            offset=count * period,
                            period=period,
                        )
                    )
                    data = await asyncio.wait_for(fut, timeout=10)
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
            finally:
                waiters = self._history_waiters.get(asset, [])
                if fut in waiters:
                    waiters.remove(fut)
                if not waiters:
                    self._history_waiters.pop(asset, None)

        candles: list[Candle] = []
        for it in data.data:
            candles.append(Candle(
                time=float(it.time),
                open=float(it.open),
                high=float(it.high),
                low=float(it.low),
                close=float(it.close),
            ))
        candles.sort(key=lambda c: c.time)
        return candles
