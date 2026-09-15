"""Tests for demo-only automated execution.

Two things are being defended. First, the fence: this module places orders
without a human, so the refusals that keep it off a funded account — the config
check, the server's own ``is_demo`` answer, and the flag on every order — are
tested as behaviour, not read as intentions. Second, the settlement read: the
SDK's ``check_deal_result`` registers a *fresh* event for the deal id and waits
on it, so calling it after the close has already arrived burns the whole timeout
and then raises while the closed deal sits in storage. The storage is read
first, and that ordering is what the tests below pin down.

No socket, no account, no SDK client: the storage is faked, so these run in
milliseconds and cannot place anything.
"""

import asyncio
import datetime as dt
import tempfile
import types
import unittest
import uuid
from decimal import Decimal
from pathlib import Path

from execution import (
    LOSS, PUSH, UNKNOWN, UNPLACED, WIN, DemoBroker, Order, Settlement,
    has_settled, order_key, outcome_of_profit, settlement_of,
)
from market.clock import VirtualClock

BASE = 1_700_000_040.0        # on a minute boundary
PERIOD = 60


# ---------------------------------------------------------------------------
# stand-ins for the SDK
# ---------------------------------------------------------------------------
class FakeDeal:
    """The parts of ``pocket_option.models.Deal`` this module reads."""

    def __init__(self, *, deal_id="deal-1", closed=False, profit=None,
                 open_price=1.0842, close_price=1.0850, request_id=77):
        self.id = deal_id
        self._closed = closed
        self.profit = profit
        self.open_price = open_price
        self.close_price = close_price if closed else None
        self.percent_profit = 85.0
        self.amount = 1
        self.open_timestamp = BASE
        self.close_timestamp = BASE + PERIOD if closed else None
        self.request_id = request_id

    @property
    def closed(self):
        return self._closed


class FakeStorage:
    """Stands in for ``MemoryDealsStorage``.

    ``stored_closed`` is the whole point: it decides whether the closed deal is
    already in storage when the settlement is read, which is the case the real
    SDK handles badly.
    """

    def __init__(self, *, stored_closed=True, settle_profit=0.85,
                 fail_open=None, fail_check=None, deal_id="deal-1"):
        self.opened: list[dict] = []
        self.check_calls = 0
        self.stored_closed = stored_closed
        self.fail_open = fail_open
        self.fail_check = fail_check
        self.deal_id = deal_id
        self.open_copy = FakeDeal(deal_id=deal_id, closed=False)
        self.final = FakeDeal(deal_id=deal_id, closed=True, profit=settle_profit)

    async def open_deal(self, **kwargs):
        self.opened.append(kwargs)
        if self.fail_open is not None:
            raise self.fail_open
        return self.open_copy

    async def get_deal(self, *, deal_id=None, request_id=None):
        return self.final if self.stored_closed else self.open_copy

    async def check_deal_result(self, *, deal=None, wait_time=600):
        self.check_calls += 1
        if self.fail_check is not None:
            raise self.fail_check
        return self.final


def scheduled_deal(deal_id="deal-1", *, amount=1, percent_profit=92.0):
    """A deal exactly as the server sends it the moment an order is accepted.

    This is the shape that produced fabricated wins on the live demo account:
    ``close_timestamp`` is already set — to the second the option is *due* to
    expire — while ``close_price`` is still ``0``, because no exit price exists
    yet, and ``profit`` is already filled in with the payout the trade would win.
    Every field that looks like a result is the broker's projection of one.
    """
    deal = FakeDeal(deal_id=deal_id, closed=True,
                    profit=amount * percent_profit / 100)
    deal.close_price = 0
    deal.percent_profit = percent_profit
    deal.amount = amount
    return deal


def sdk_deal(**overrides):
    """A genuine SDK ``Deal``, as the server sends it the moment an order fills.

    Deliberately the real model rather than a stub: the bug this guards against
    was this project's *belief* about what a deal contains, and a fake built from
    that same belief would have agreed with it.
    """
    from pocket_option.models import Asset, Command, Deal

    def stamp(offset):
        return dt.datetime.fromtimestamp(BASE + offset, tz=dt.timezone.utc)

    values = dict(
        id=uuid.uuid4(), command=Command.CALL, asset=Asset.EURUSD_otc, uid=7,
        amount=Decimal("1"), is_demo=1, profit=Decimal("0.92"),
        percent_profit=92.0, percent_loss=100.0,
        open_time=stamp(0), close_time=stamp(60),
        open_timestamp=BASE, close_timestamp=BASE + 60,
        open_price=Decimal("1.17275"), close_price=Decimal("0"),
        copy_ticket="", is_copy_signal=False, currency="USD")
    values.update(overrides)
    return Deal(**values)


class ScheduledStorage(FakeStorage):
    """A storage whose deal has been scheduled but has not traded yet."""

    def __init__(self, **kwargs):
        super().__init__(stored_closed=False, **kwargs)
        self.projected = scheduled_deal(self.deal_id)

    async def get_deal(self, *, deal_id=None, request_id=None):
        return self.projected

    async def check_deal_result(self, *, deal=None, wait_time=600):
        self.check_calls += 1
        return self.projected


class LazyStorage(FakeStorage):
    """The close lands *before* the listener is registered.

    ``check_deal_result`` waits on a fresh event, so a close that already
    arrived makes it time out and raise while the settled deal sits in storage.
    The first look is too early (the deal is still open); a later one is not.
    """

    def __init__(self, **kwargs):
        super().__init__(stored_closed=False, **kwargs)
        self.lookups = 0

    async def get_deal(self, *, deal_id=None, request_id=None):
        self.lookups += 1
        return self.open_copy if self.lookups == 1 else self.final

    async def check_deal_result(self, *, deal=None, wait_time=600):
        self.check_calls += 1
        raise TimeoutError("the close arrived before this was registered")


class FakeClient:
    """A client with just the surface ``attach``/``verify`` touch."""

    def __init__(self, *, is_demo=1, has_auth=True, balance=None):
        self.authorization_data = (
            types.SimpleNamespace(is_demo=is_demo) if has_auth else None)
        self._balance = balance
        self._balance_handler = None
        self.update_balance_calls = 0
        self.on = types.SimpleNamespace(
            deals_success_open=self._register,
            deals_fail_open=self._register,
            deals_success_close=self._register,
            deals_update_opened=self._register,
            deals_update_closed=self._register,
            balance_success_update=self._register_balance)

    @staticmethod
    def _register(handler):
        return handler

    def _register_balance(self, handler):
        self._balance_handler = handler
        return handler

    async def update_balance(self):
        """Answer the way the SDK does: by firing an event, not by returning."""
        self.update_balance_calls += 1
        if self._balance_handler is not None and self._balance is not None:
            self._balance_handler(
                types.SimpleNamespace(balance=self._balance, is_demo=1))


class NoBalanceClient(FakeClient):
    """A client that offers no way to ask for a balance at all."""

    update_balance = None


def armed(**kwargs):
    """A broker that is attached, enabled and using a fake storage.

    ``enabled`` is set directly because ``verify`` is exercised on its own; the
    point of the other tests is what happens *after* the fence has passed.
    """
    broker = DemoBroker(**kwargs)
    storage = FakeStorage()
    broker.storage = storage
    broker.client = FakeClient()
    broker.enabled = True
    broker.refusal = ""
    return broker, storage


def order(entry_at=BASE + PERIOD, **overrides):
    values = dict(asset="EURUSD_otc", direction="CALL", entry_at=entry_at,
                  expiry_at=entry_at + PERIOD, amount=1.0, duration=PERIOD)
    values.update(overrides)
    return Order(**values)


def run(coro):
    return asyncio.run(coro)


async def place(broker, target=None, *, lead=5.0):
    """Submit one order and drive the clock past its entry second.

    Goes through ``submit`` rather than calling the worker directly — that is
    what the minute loop does, and it is the only path that registers the order
    and takes the clock into account.
    """
    target = order() if target is None else target
    vc = VirtualClock(start=target.entry_at - lead)
    broker.clock = vc
    broker.submit(target)
    await vc.advance(lead + 5.0, steps=int(lead + 5.0))
    return target


async def place_and_stop(broker, target=None):
    """As above, then stop cleanly so no task outlives the test's loop."""
    target = await place(broker, target)
    await broker.shutdown()
    return target


# ---------------------------------------------------------------------------
class TestOutcomes(unittest.TestCase):
    def test_profit_decides_the_outcome(self):
        self.assertEqual(outcome_of_profit(0.85), WIN)
        self.assertEqual(outcome_of_profit(-1.0), LOSS)
        self.assertEqual(outcome_of_profit(0.0), PUSH, "a refund is neither")
        self.assertEqual(outcome_of_profit(None), UNKNOWN)

    def test_an_open_deal_is_not_a_settlement(self):
        # Reporting an unclosed deal would invent a result out of no data.
        self.assertIsNone(settlement_of(FakeDeal(closed=False)))
        self.assertIsNone(settlement_of(None))

    def test_a_scheduled_deal_is_not_a_settlement(self):
        # The server sets close_timestamp when the deal *opens* — to the second
        # the option is due to expire — while close_price is still 0 and profit
        # is already filled in with the payout the trade would win. Trusting
        # ``Deal.closed`` therefore reports every order as an instant win of
        # exactly its potential profit, which is what the live demo run did:
        # orders "settled" in the same millisecond they were placed.
        deal = scheduled_deal()

        self.assertTrue(deal.closed, "it does look closed — that is the trap")
        self.assertAlmostEqual(deal.profit, 0.92, msg="and it looks like a win")
        self.assertFalse(has_settled(deal))
        self.assertIsNone(settlement_of(deal))

    def test_an_absent_exit_price_is_not_an_exit_price(self):
        # Zero and null mean the same thing here: the deal has not traded yet.
        for price in (0, 0.0, None, ""):
            with self.subTest(close_price=price):
                deal = scheduled_deal()
                deal.close_price = price
                self.assertFalse(has_settled(deal))

    def test_a_real_exit_price_is_what_makes_it_settled(self):
        deal = scheduled_deal()
        deal.close_price = 1.0850

        self.assertTrue(has_settled(deal))
        self.assertEqual(settlement_of(deal).outcome, WIN)

    def test_the_sdk_model_itself_is_refused_while_only_scheduled(self):
        """The fix checked against the broker's own model, not a stand-in.

        ``FakeDeal`` is written to the shape this project *believed* a deal had.
        The model below is what the SDK actually returns, with its own types —
        ``close_price`` is a ``Decimal`` and ``closed`` a property. A stub cannot
        catch the model disagreeing with the belief, which is exactly how the
        fabricated settlements got through: every field that reads like a result
        is present and populated the moment the order is accepted.
        """
        accepted = sdk_deal()

        self.assertTrue(accepted.closed, "the server sets close_timestamp at open")
        self.assertEqual(accepted.profit, Decimal("0.92"), "the projected payout")
        self.assertEqual(accepted.close_price, Decimal("0"), "no exit price yet")
        self.assertFalse(has_settled(accepted))
        self.assertIsNone(settlement_of(accepted))

        # The same deal, once it really has an exit price to read.
        settled = sdk_deal(close_price=Decimal("1.17301"), profit=Decimal("-1.0"))

        self.assertTrue(has_settled(settled))
        self.assertEqual(settlement_of(settled).outcome, LOSS)

    def test_a_closed_deal_becomes_a_settlement_with_its_numbers(self):
        settlement = settlement_of(FakeDeal(closed=True, profit=-1.0))

        self.assertEqual(settlement.outcome, LOSS)
        self.assertEqual(settlement.deal_id, "deal-1")
        self.assertAlmostEqual(settlement.open_price, 1.0842)
        self.assertAlmostEqual(settlement.close_price, 1.0850)
        self.assertAlmostEqual(settlement.profit, -1.0)
        self.assertAlmostEqual(settlement.payout, 85.0)

    def test_the_journal_block_omits_nothing_it_was_given(self):
        block = settlement_of(FakeDeal(closed=True, profit=0.85)).as_dict()

        self.assertEqual(block["outcome"], WIN)
        self.assertEqual(block["deal_id"], "deal-1")
        self.assertIn("entry", block)
        self.assertIn("exit", block)

    def test_the_order_key_is_the_journal_join_key(self):
        # Same shape as journal.signal_id, so an order, its signal and its
        # settlement all address the same trade.
        from journal import signal_id
        self.assertEqual(order_key("EURUSD_otc", BASE),
                         signal_id("EURUSD_otc", BASE))


class TestTheDemoFence(unittest.TestCase):
    """The refusals that keep automated orders off a funded account."""

    def test_verify_accepts_the_server_saying_the_account_is_demo(self):
        broker = DemoBroker()
        broker.attach(FakeClient(is_demo=1))

        allowed, reason = run(broker.verify())

        self.assertTrue(allowed)
        self.assertTrue(broker.enabled)
        self.assertEqual(reason, "")

    def test_verify_refuses_when_the_server_says_the_account_is_not_demo(self):
        # The session string came from a live login whatever .env claims.
        broker = DemoBroker()
        broker.attach(FakeClient(is_demo=0))

        allowed, reason = run(broker.verify())

        self.assertFalse(allowed)
        self.assertFalse(broker.enabled)
        self.assertIn("NOT demo", reason)

    def test_verify_refuses_when_the_account_type_was_never_reported(self):
        broker = DemoBroker()
        broker.attach(FakeClient(has_auth=False))

        allowed, _ = run(broker.verify())

        self.assertFalse(allowed)
        self.assertFalse(broker.enabled)

    def test_a_broker_that_never_authorised_cannot_place_anything(self):
        broker = DemoBroker()
        self.assertFalse(broker.enabled)
        self.assertIsNotNone(broker.validate())

    def test_the_balance_is_read_from_the_event_it_arrives_on(self):
        # The SDK has no get_balance(): the figure comes back as an event, so a
        # getter that does not exist would silently read nothing forever.
        broker = DemoBroker()
        client = FakeClient(balance=10_000.0)
        broker.attach(client)

        allowed, _ = run(broker.verify())

        self.assertTrue(allowed)
        self.assertEqual(client.update_balance_calls, 1)
        self.assertAlmostEqual(broker.balance, 10_000.0)
        self.assertTrue(broker.balance_is_demo)
        self.assertIn("10000", broker.summary())

    def test_a_client_with_no_balance_api_is_still_tradeable(self):
        # Nothing about the fence may depend on a nicety, and this must not hang
        # waiting for an answer that cannot come.
        broker = DemoBroker()
        broker.attach(NoBalanceClient())

        allowed, _ = run(broker.verify())

        self.assertTrue(allowed)
        self.assertIsNone(broker.balance)
        self.assertIsNone(broker.balance_is_demo)

    def test_every_order_carries_the_demo_flag_explicitly(self):
        # Not only in the config: the flag travels with the request.
        broker, storage = armed()
        run(place_and_stop(broker))

        self.assertEqual(len(storage.opened), 1)
        self.assertEqual(storage.opened[0]["is_demo"], 1)

    def test_the_order_is_translated_into_the_sdk_s_own_terms(self):
        from pocket_option.models import Asset, DealAction

        broker, storage = armed()
        target = order(direction="PUT", amount=2.0, duration=60)
        run(place_and_stop(broker, target))
        sent = storage.opened[0]

        self.assertEqual(sent["asset"], Asset.EURUSD_otc)
        self.assertEqual(sent["action"], DealAction.PUT)
        self.assertEqual(sent["time"], 60)
        self.assertAlmostEqual(float(sent["amount"]), 2.0)
        self.assertTrue(sent["check_limits"], "the SDK's own limits are kept")

    def test_an_unknown_symbol_is_refused_before_a_socket_call(self):
        # ``Asset("NOT_A_MARKET")`` does *not* raise — the enum's ``_missing_``
        # invents a member for any string, so a ValueError guard would never
        # fire and this typo would reach the broker. It is refused locally
        # because the name is checked against the enum's real members.
        broker, storage = armed()
        target = order(asset="NOT_A_MARKET")
        run(place_and_stop(broker, target))

        self.assertEqual(storage.opened, [], "nothing was sent")
        self.assertIn("NOT_A_MARKET", broker.orders[target.key].error)
        self.assertIn("NOT_A_MARKET", broker.failures[0])

    def test_an_amount_outside_the_api_limits_is_refused(self):
        # Two paths, because the amount comes from the order when it names one
        # and from the broker otherwise — and only the value actually sent is
        # worth checking.
        for amount in (0.0, -5.0, 1e9):
            with self.subTest(amount=amount):
                broker, storage = armed(amount=amount)
                self.assertIsNotNone(broker.validate(), f"{amount} should be refused")

                # amount=0 means "no amount of my own", so the broker's bad
                # value is the one that would be sent.
                target = order(amount=0)
                run(place_and_stop(broker, target))
                self.assertEqual(storage.opened, [], f"{amount} reached the broker")
                self.assertIn("amount", broker.orders[target.key].error)

    def test_an_amount_the_order_names_is_checked_instead_of_the_default(self):
        # The broker is configured sensibly; the order is not. Refusing this is
        # the point of validating the order rather than the broker.
        broker, storage = armed(amount=1.0)
        target = order(amount=1e9)
        run(place_and_stop(broker, target))

        self.assertEqual(storage.opened, [], "an out-of-range order was sent")
        self.assertIn("amount", broker.orders[target.key].error)

    def test_a_duration_outside_the_api_limits_is_refused(self):
        # Same fence as the amount, and reachable: a 0-second expiry would be
        # a trade that opens and closes at the same instant.
        for duration in (0, 4, 100_000):
            with self.subTest(duration=duration):
                broker, storage = armed(duration=duration)
                self.assertIsNotNone(broker.validate(),
                                     f"{duration}s should be refused")

                target = order(duration=duration or 0)
                run(place_and_stop(broker, target))
                self.assertEqual(storage.opened, [],
                                 f"{duration}s reached the broker")
                self.assertIn("duration", broker.orders[target.key].error)


class TestPlacingAndSettling(unittest.TestCase):
    def test_an_order_is_held_until_its_entry_second(self):
        """The whole measurement depends on this: not a second early."""
        async def scenario():
            broker, storage = armed()
            vc = VirtualClock(start=BASE)
            broker.clock = vc
            target = order(entry_at=BASE + PERIOD)

            broker.submit(target)
            await vc.advance(30.0, steps=30)
            early = list(storage.opened)

            await vc.advance(40.0, steps=40)
            placed = list(storage.opened)
            await broker.shutdown()
            return early, placed

        early, placed = run(scenario())
        self.assertEqual(early, [], "placed before its boundary")
        self.assertEqual(len(placed), 1, "never placed at all")

    def test_a_settlement_already_in_storage_is_read_without_waiting(self):
        """The SDK trap: check_deal_result after the close waits out its timeout."""
        broker, storage = armed()
        storage.stored_closed = True
        storage.fail_check = AssertionError("check_deal_result should not be called")

        run(place_and_stop(broker))
        settlement = broker.settlement_for("EURUSD_otc", BASE + PERIOD)

        self.assertEqual(storage.check_calls, 0)
        self.assertIsNotNone(settlement)
        self.assertEqual(settlement.outcome, WIN)

    def test_a_close_that_has_not_arrived_is_waited_for(self):
        broker, storage = armed()
        storage.stored_closed = False

        run(place_and_stop(broker))
        settlement = broker.settlement_for("EURUSD_otc", BASE + PERIOD)

        self.assertEqual(storage.check_calls, 1)
        self.assertIsNotNone(settlement)

    def test_a_scheduled_deal_never_becomes_a_recorded_result(self):
        # The regression that matters. A deal the broker has only *scheduled*
        # must leave no result behind, so the bot's own bar-close label stands
        # instead of being overwritten by the broker's projection of a payout.
        broker, _ = armed()
        broker.storage = storage = ScheduledStorage()
        target = run(place_and_stop(broker))

        self.assertTrue(broker.orders[target.key].placed, "the order did go out")
        self.assertEqual(storage.check_calls, 1)
        self.assertIsNone(broker.settlement_for(target.asset, target.entry_at))
        self.assertEqual(broker.settlements, {}, "no result may be invented")

    def test_a_close_that_landed_before_the_listener_is_still_found(self):
        # The other half of reading storage first: the first look is too early,
        # check_deal_result then times out because the close event has already
        # fired, and the settled deal is in storage by the second look.
        broker, _ = armed()
        broker.storage = storage = LazyStorage()
        target = run(place_and_stop(broker))

        settlement = broker.settlement_for(target.asset, target.entry_at)
        self.assertEqual(storage.check_calls, 1)
        self.assertIsNotNone(settlement, "the settled deal was in storage")
        self.assertEqual(settlement.outcome, WIN)

    def test_a_settlement_that_never_arrives_leaves_the_order_unsettled(self):
        broker, storage = armed()
        storage.stored_closed = False
        storage.fail_check = TimeoutError("no close")

        target = run(place_and_stop(broker))

        self.assertIsNone(broker.settlement_for("EURUSD_otc", BASE + PERIOD))
        self.assertEqual(broker.orders[target.key].deal_id, "deal-1",
                         "the order was placed, only the result is missing")

    def test_a_failed_order_is_recorded_rather_than_raised(self):
        broker, storage = armed()
        storage.fail_open = RuntimeError("min_amount")

        target = run(place_and_stop(broker))
        placed = broker.orders[target.key]

        self.assertFalse(placed.placed)
        self.assertIn("min_amount", placed.error)
        self.assertEqual(len(broker.failures), 1)
        self.assertEqual(broker.settlements, {}, "no result may be invented")

    def test_a_refund_settles_as_a_push(self):
        broker, storage = armed()
        storage.final = FakeDeal(closed=True, profit=0.0)

        run(place_and_stop(broker))
        settlement = broker.settlement_for("EURUSD_otc", BASE + PERIOD)

        self.assertEqual(settlement.outcome, PUSH)

    def test_the_deal_and_its_fill_are_kept_on_the_order(self):
        broker, storage = armed()
        target = run(place_and_stop(broker))
        placed = broker.orders[target.key]

        self.assertTrue(placed.placed)
        self.assertEqual(placed.deal_id, "deal-1")
        self.assertEqual(placed.request_id, 77)
        self.assertAlmostEqual(placed.open_price, 1.0842)
        self.assertIsNotNone(placed.placed_at)

    def test_a_second_order_for_the_same_trade_is_ignored(self):
        # The same market cannot signal twice on one boundary, so a duplicate
        # key means a bug upstream — it must not become a duplicate order.
        async def scenario():
            broker, storage = armed()
            first = order()
            await place(broker, first)
            broker.submit(order())          # same asset and second
            await broker.shutdown()
            return broker, storage

        broker, storage = run(scenario())
        self.assertEqual(len(broker.orders), 1)
        self.assertEqual(len(storage.opened), 1, "one order, not two")

    def test_the_order_carries_the_amount_that_was_actually_sent(self):
        # submit freezes the request onto the order so the record cannot drift.
        broker, storage = armed(amount=3.0)
        target = order(amount=0, duration=0)
        run(place_and_stop(broker, target))

        self.assertAlmostEqual(target.amount, 3.0)
        self.assertEqual(target.duration, PERIOD)
        self.assertAlmostEqual(float(storage.opened[0]["amount"]), 3.0)


class TestReporting(unittest.TestCase):
    def test_summary_says_plainly_when_it_is_not_trading(self):
        broker = DemoBroker()
        self.assertIn("OFF", broker.summary())

        broker.disable("the account is not demo")
        self.assertIn("OFF", broker.summary())
        self.assertIn("not demo", broker.summary())

    def test_summary_names_the_demo_account_when_it_is(self):
        broker, _ = armed(amount=2.0, duration=60)
        broker.balance = 10_000.0

        text = broker.summary()

        self.assertIn("ON", text)
        self.assertIn("demo", text)
        self.assertIn("2", text)

    def test_disable_stops_orders_without_killing_the_process(self):
        broker, _ = armed()
        broker.disable("the balance ran out")

        self.assertFalse(broker.enabled)
        self.assertIsNotNone(broker.validate())


class TestShutdown(unittest.TestCase):
    def test_shutdown_cancels_orders_that_have_not_run_yet(self):
        async def scenario():
            broker, storage = armed()
            vc = VirtualClock(start=BASE)
            broker.clock = vc
            broker.submit(order(entry_at=BASE + 3600))
            await vc.advance(1.0, steps=2)

            inflight_before = broker.in_flight()
            await broker.shutdown()
            return inflight_before, broker.in_flight(), storage.opened

        before, after, opened = run(scenario())
        self.assertEqual(before, 1, "the order should be waiting for its boundary")
        self.assertEqual(after, 0)
        self.assertEqual(opened, [], "a cancelled order must never be sent")

    def test_shutdown_is_harmless_with_nothing_in_flight(self):
        broker, _ = armed()

        async def scenario():
            await broker.shutdown()
            await broker.shutdown()

        run(scenario())
        self.assertEqual(broker.in_flight(), 0)


if __name__ == "__main__":
    unittest.main()
