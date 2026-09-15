"""Demo-only automated execution.

The bot could always *say* what it thought happened; it could never check. This
module places the order so that the broker answers instead: the trade opens on
the boundary the signal named, and the settlement the broker pays out becomes
the recorded outcome. Whatever the tick bars say is kept alongside it, as the
bot's own claim, so the two can be compared trade by trade.

**It will not trade a funded account.** Three refusals, independently:

1. ``config.load_config`` rejects ``AUTO_TRADE=1`` without ``POCKET_IS_DEMO=1``,
   before a socket is opened or a session string is used.
2. ``verify`` re-reads ``authorization_data.is_demo`` — the account the server
   says this session landed on, not the one we asked for — and disables itself
   if it is not demo. A session string pasted from a live login cannot get past
   this by being mislabelled in ``.env``.
3. Every order carries ``is_demo=1`` explicitly.

The order path is deliberately inert until ``AUTO_TRADE=1``: with the default
configuration this module is never constructed, and the scheduler behaves
exactly as it did before.

Why a background task per order rather than an await in the loop: ``open_deal``
waits up to 30s for the server to confirm, and the settlement takes a further
minute. Blocking the minute loop on either would cost signals. So each order
runs on its own task — sleep to the boundary, place, settle — and the scheduler
reads the result when it next settles that trade.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from market.clock import Clock, RealClock
from telegram.sender import format_duration

log = logging.getLogger("pocket.execution")

WIN, LOSS, PUSH, UNKNOWN = "WIN", "LOSS", "PUSH", "UNKNOWN"
# An order the broker never accepted. Not an outcome of a trade — there was no
# trade — but it has to be distinguishable from "not attempted", so it is
# recorded in the journal as its own thing and excluded from every rate.
UNPLACED = "UNPLACED"

# The SDK's own limits, restated so a bad amount is caught before a socket call.
MIN_AMOUNT = 1.0
MAX_AMOUNT = 50_000.0
MIN_DURATION = 5
MAX_DURATION = 43_200


def outcome_of_profit(profit: Optional[float]) -> str:
    """The broker's verdict, from the money it actually moved.

    A refund — profit exactly zero — is neither a win nor a loss, and must not
    be counted as one. ``StatsTracker`` counts anything that is not a WIN as a
    loss, so a refund has to be kept away from it explicitly.
    """
    if profit is None:
        return UNKNOWN
    if profit > 0:
        return WIN
    if profit < 0:
        return LOSS
    return PUSH


def order_key(asset: str, entry_at: float) -> str:
    """Join key between a signal, its order and its settlement.

    Same shape as ``journal.signal_id``: market plus the second of entry.
    """
    return f"{asset}@{int(entry_at)}"


@dataclass
class Order:
    """One order the bot asked the broker to place."""

    asset: str
    direction: str
    entry_at: float
    expiry_at: float
    amount: float
    duration: int
    placed_at: Optional[float] = None
    request_id: Optional[int] = None
    open_price: Optional[float] = None
    error: str = ""
    # The SDK's own Deal object, kept so the settlement can be looked up.
    deal: Any = None

    @property
    def key(self) -> str:
        return order_key(self.asset, self.entry_at)

    @property
    def deal_id(self) -> str:
        return str(getattr(self.deal, "id", "") or "")

    @property
    def placed(self) -> bool:
        return not self.error and self.deal is not None

    def describe(self) -> str:
        if self.error:
            return f"{self.asset} {self.direction}: NOT placed ({self.error})"
        return (f"{self.asset} {self.direction} {self.amount:g} "
                f"@ {self.open_price} (deal {self.deal_id[:8]})")


@dataclass
class Settlement:
    """What the broker paid out on one order."""

    outcome: str
    deal_id: str = ""
    open_at: Optional[float] = None
    close_at: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    profit: Optional[float] = None
    payout: Optional[float] = None
    amount: Optional[float] = None

    def as_dict(self) -> dict:
        """The record written into the journal. ``None`` where unknown."""
        return {
            "outcome": self.outcome,
            "deal_id": self.deal_id,
            "open_at": self.open_at,
            "close_at": self.close_at,
            "entry": self.open_price,
            "exit": self.close_price,
            "profit": self.profit,
            "payout": self.payout,
        }


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> Optional[bool]:
    """The SDK's own flag, which is an IntBool rather than a bool."""
    if value is None:
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def has_settled(deal: Any) -> bool:
    """Whether the broker has settled this deal — not merely scheduled it.

    ``Deal.closed`` is ``close_timestamp is not None``, and the server sets that
    **when the deal opens**: it is the second the option is *due* to expire, not
    a record that it did. Two more fields read like a result without being one —
    ``close_price`` arrives as ``0`` (the exit price is not known yet) and
    ``profit`` as the payout the trade *would* pay if it won.

    Trusting ``closed`` therefore reports every order as an instant win of
    exactly its potential profit. Measured live on the demo account: four orders
    in a row "settled" in the same millisecond they were placed, each with
    ``profit == amount * percent_profit / 100`` and ``close_price == 0``.

    A settled deal has a real exit price, and that test needs no clock — which
    matters, because the broker's timestamps are on a different timezone from
    ours (a constant offset of about two hours), so comparing them to ``now``
    is not safe either.
    """
    if deal is None or not getattr(deal, "closed", False):
        return False
    price = _number(getattr(deal, "close_price", None))
    return price is not None and price > 0


def settlement_of(deal: Any) -> Optional[Settlement]:
    """Read a *settled* SDK ``Deal`` as a ``Settlement``.

    Returns None for a deal the broker has not settled, so that a deal's own
    projection of its payout can never be recorded as its result. See
    ``has_settled``.
    """
    if not has_settled(deal):
        return None
    profit = _number(getattr(deal, "profit", None))
    return Settlement(
        outcome=outcome_of_profit(profit),
        deal_id=str(getattr(deal, "id", "") or ""),
        open_at=_number(getattr(deal, "open_timestamp", None)),
        close_at=_number(getattr(deal, "close_timestamp", None)),
        open_price=_number(getattr(deal, "open_price", None)),
        close_price=_number(getattr(deal, "close_price", None)),
        profit=profit,
        payout=_number(getattr(deal, "percent_profit", None)),
        amount=_number(getattr(deal, "amount", None)),
    )


class DemoBroker:
    """Places orders on the demo account and collects what the broker paid.

    Ordering matters: ``attach`` has to run *before* the socket connects,
    because the server pushes the deal history during authorisation and a
    listener attached afterwards misses it.
    """

    def __init__(self, *, amount: float = MIN_AMOUNT, duration: int = 60,
                 clock: Optional[Clock] = None,
                 settle_grace: float = 60.0,
                 max_concurrent: int = 3) -> None:
        self.amount = float(amount)
        self.duration = int(duration)
        self.clock = clock or RealClock()
        self.settle_grace = settle_grace
        self.max_concurrent = max_concurrent

        self.client: Any = None
        self.storage: Any = None
        self.enabled = False
        self.refusal = "not attached to a session"
        self.account_is_demo: Optional[bool] = None
        self.balance: Optional[float] = None
        self.balance_is_demo: Optional[bool] = None

        self.orders: dict[str, Order] = {}
        self.settlements: dict[str, Settlement] = {}
        self.failures: list[str] = []
        self._tasks: dict[str, asyncio.Task] = {}

    # -- wiring --------------------------------------------------------------
    def attach(self, client: Any) -> None:
        """Create the deals storage on a client that is about to connect."""
        from pocket_option.contrib.deals import MemoryDealsStorage

        self.client = client
        self.storage = MemoryDealsStorage(client)
        self.refusal = "the session has not authorised yet"
        self._subscribe_balance(client)

    async def verify(self) -> tuple[bool, str]:
        """Confirm from the *server* that this session is a demo account.

        Called after ``wait_for_authorization``. ``authorization_data.is_demo``
        is the broker's statement about the account the session landed on, which
        is the only version of that fact worth trusting.
        """
        if self.storage is None or self.client is None:
            self.refusal = "no session to check"
            return False, self.refusal

        data = getattr(self.client, "authorization_data", None)
        is_demo = getattr(data, "is_demo", None)
        if is_demo is None:
            self.refusal = ("the session did not report whether the account is "
                            "demo, so automated orders are refused")
            return False, self.refusal

        self.account_is_demo = bool(int(is_demo))
        if not self.account_is_demo:
            self.refusal = ("the account this session authorised to is NOT demo — "
                            "automated orders are disabled")
            self.enabled = False
            log.error("%s", self.refusal)
            return False, self.refusal

        self.enabled = True
        self.refusal = ""
        await self._read_balance()
        log.info("auto-trading enabled on the demo account (amount %g, %ds)",
                 self.amount, self.duration)
        return True, ""

    def _subscribe_balance(self, client: Any) -> None:
        """Listen for the balance, which arrives as an event and not a return value."""
        register = getattr(getattr(client, "on", None), "balance_success_update", None)
        if register is None:
            return

        def on_balance(event: Any) -> None:
            self.balance = _number(getattr(event, "balance", None))
            self.balance_is_demo = _truthy(getattr(event, "is_demo", None))

        try:
            register(on_balance)
        except Exception as exc:
            log.debug("could not subscribe to balance updates: %s", exc)

    async def _read_balance(self) -> None:
        """Ask for a balance update, and do not wait for the answer.

        Purely informative — the startup line reads better with a figure on it —
        so it must never delay the fence or hold up a session. The figure lands
        on ``balance`` whenever the event arrives, and ``summary`` simply omits
        it until then.
        """
        ask = getattr(self.client, "update_balance", None)
        if ask is None:
            return
        try:
            await ask()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("could not request the balance: %s", exc)

    def disable(self, reason: str) -> None:
        """Stop placing orders, keeping the process alive."""
        self.enabled = False
        self.refusal = reason
        log.error("auto-trading disabled: %s", reason)

    # -- placing -------------------------------------------------------------
    def validate(self, order: Optional[Order] = None) -> Optional[str]:
        """Why this order cannot be placed, or None when it can.

        Checks the values that will actually be sent — taken from the order when
        it is given, so the pre-flight check and the request cannot disagree.
        """
        if not self.enabled:
            return self.refusal or "auto-trading is not enabled"
        amount = self.amount if order is None else (order.amount or self.amount)
        duration = self.duration if order is None else (order.duration or self.duration)
        if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
            return f"amount {amount:g} is outside {MIN_AMOUNT:g}..{MAX_AMOUNT:g}"
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            return f"duration {duration}s is outside {MIN_DURATION}..{MAX_DURATION}s"
        return None

    def submit(self, order: Order) -> None:
        """Schedule one order: sleep to its entry second, place, wait to settle.

        Returns immediately — the caller is the minute loop, which must not wait
        on a socket round-trip. The result is read later via ``settlement_for``.
        """
        if order.key in self._tasks:
            log.debug("an order for %s is already in flight", order.key)
            return
        # Freeze what will be sent onto the order, so the record of the request
        # and the request itself cannot drift apart.
        order.amount = order.amount or self.amount
        order.duration = order.duration or self.duration
        self.orders[order.key] = order
        self._tasks[order.key] = asyncio.create_task(
            self._run(order), name=f"order:{order.key}")

    async def _run(self, order: Order) -> None:
        """Place the order on its boundary, then wait for the broker's answer."""
        try:
            problem = self.validate(order)
            if problem:
                order.error = problem
                self.failures.append(f"{order.key}: {problem}")
                return

            await self.clock.sleep_until(order.entry_at)
            order.placed_at = self.clock.now()

            try:
                order.deal = await self._open(order)
            except Exception as exc:
                order.error = f"{type(exc).__name__}: {exc}"
                self.failures.append(f"{order.key}: {order.error}")
                log.error("order failed for %s: %s", order.key, order.error)
                return

            order.open_price = _number(getattr(order.deal, "open_price", None))
            order.request_id = getattr(order.deal, "request_id", None)
            log.info("ORDER PLACED %s", order.describe())

            settlement = await self._settle(order)
            if settlement is not None:
                self.settlements[order.key] = settlement
                log.info("SETTLED %s %s profit=%s payout=%s",
                         order.asset, settlement.outcome, settlement.profit,
                         settlement.payout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                # one bad order must not kill the bot
            order.error = f"{type(exc).__name__}: {exc}"
            self.failures.append(f"{order.key}: {order.error}")
            log.exception("unexpected failure placing %s", order.key)

    async def _open(self, order: Order) -> Any:
        """Translate our terms into the SDK's and send the order."""
        from pocket_option.models import Asset, DealAction

        # ``Asset._missing_`` fabricates a member for *any* string, so
        # ``Asset("NOPE")`` quietly succeeds and ``except ValueError`` around it
        # is dead code — a typo'd symbol would travel to the server as a market
        # that does not exist. Membership is the only check that means anything.
        if order.asset not in Asset.__members__:
            raise ValueError(f"{order.asset} is not an asset the broker knows")

        return await self.storage.open_deal(
            asset=Asset(order.asset),
            amount=order.amount or self.amount,
            action=DealAction(order.direction.lower()),
            time=int(order.duration or self.duration),
            # Asked for explicitly on every order, not only in the config.
            is_demo=1,
            check_limits=True,
        )

    async def _settle(self, order: Order) -> Optional[Settlement]:
        """Wait for the deal to settle and read what it paid.

        The storage is checked first. ``check_deal_result`` registers a *fresh*
        event for the deal id and waits on it, so calling it after the close
        message has already arrived waits out the whole timeout and then raises
        — the settled deal is sitting in the storage the entire time.

        "In storage" is not the same as "settled", though: the deal is there
        from the moment it opens, wearing a scheduled close time and a projected
        payout. ``settlement_of`` refuses those, so this only short-circuits on a
        real result.
        """
        deal = order.deal
        if deal is None:
            return None

        stored = await self._stored_deal(deal)
        if has_settled(stored):
            return settlement_of(stored)

        try:
            closed = await self.storage.check_deal_result(
                deal=stored or deal,
                wait_time=int((order.duration or self.duration) + self.settle_grace))
        except Exception as exc:
            # The close can land before the listener is registered, in which
            # case this waits out the whole timeout and raises while the settled
            # deal sits in storage — so look once more before giving up.
            late = settlement_of(await self._stored_deal(deal))
            if late is not None:
                return late
            log.warning("no settlement arrived for %s: %s: %s",
                        order.key, type(exc).__name__, exc)
            return None

        settlement = settlement_of(closed)
        if settlement is None:
            # Not a result: the deal is still open, or closed without an exit
            # price. Either way the bot's own bar-close label stands.
            log.warning("the broker has not settled %s — no exit price yet",
                        order.key)
        return settlement

    async def _stored_deal(self, deal: Any) -> Any:
        try:
            return await self.storage.get_deal(deal_id=deal.id)
        except Exception as exc:
            log.debug("could not re-read deal %s: %s", getattr(deal, "id", "?"), exc)
            return None

    # -- reading results -----------------------------------------------------
    def settlement_for(self, asset: str, entry_at: float) -> Optional[Settlement]:
        """The broker's settlement, or None while it is still in flight."""
        return self.settlements.get(order_key(asset, entry_at))

    def order_for(self, asset: str, entry_at: float) -> Optional[Order]:
        return self.orders.get(order_key(asset, entry_at))

    def in_flight(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    async def shutdown(self) -> None:
        """Cancel anything still running, so no order outlives the session."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("execution stopped (%d order(s) abandoned, %d failure(s))",
                 len(tasks), len(self.failures))

    def summary(self) -> str:
        """One line for the startup/status message."""
        if not self.enabled:
            return f"auto-trade OFF ({self.refusal})" if self.refusal else "auto-trade OFF"
        money = f", balance {self.balance:g}" if self.balance is not None else ""
        return (f"auto-trade ON (demo, {self.amount:g}/trade, "
                f"{format_duration(self.duration)}{money})")
