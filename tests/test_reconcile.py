"""Tests for the broker reconciliation.

Two things are being defended here. First, the parsing: the deal comes from the
broker's model, not ours, so the normalisers are tested against the SDK's real
enums and against milliseconds-vs-seconds timestamps. Second, the arithmetic of
the conclusion: this tool exists to say whether there is enough evidence, so a
verdict that flatters a small sample would be worse than no tool at all.
"""

import datetime as dt
import json
import tempfile
import types
import unittest
import uuid
from decimal import Decimal
from pathlib import Path

import reconcile
from config import Config
from journal import SignalJournal, load_journal
from reconcile import (
    LOSS, PUSH, UNKNOWN, WIN, BrokerDeal, break_even_win_rate, deal_from_model,
    detect_clock_offset, epoch_seconds, match, normalize_asset, normalize_direction,
    pip_size, report, trades_needed, wilson,
)

BASE = 1_700_000_040.0        # on a minute boundary


def sdk_types():
    """The real SDK enums, so normalisation is tested against the actual values."""
    from pocket_option.models import Asset, DealAction
    return Asset, DealAction


def sdk_command(name: str):
    """The direction enum a settled ``Deal`` carries (integer-valued)."""
    from pocket_option.models import Command
    return Command[name]


def real_deal(command=None, **overrides):
    """A genuine SDK ``Deal``, not a stand-in.

    This matters: ``deal_from_model`` reads the broker's model, and the two
    enums involved are different types with the same member names. A hand-rolled
    stub cannot catch a mix-up between them, which is exactly the bug that got
    through the first version of this file.
    """
    from pocket_option.models import Asset, Command, Deal

    def stamp(offset: float) -> dt.datetime:
        return dt.datetime.fromtimestamp(BASE + offset, tz=dt.timezone.utc)

    values = dict(
        id=uuid.uuid4(), command=Command.CALL, asset=Asset.EURUSD_otc, uid=7,
        amount=Decimal("1"), is_demo=1, profit=Decimal("0.85"),
        percent_profit=85.0, percent_loss=100.0,
        open_time=stamp(0), close_time=stamp(60),
        open_timestamp=BASE, close_timestamp=BASE + 60,
        open_price=Decimal("1.0843"), close_price=Decimal("1.0851"),
        copy_ticket="", is_copy_signal=False, currency="USD",
    )
    values.update(overrides)
    if command is not None:
        values["command"] = command
    return Deal(**values)


def deal(**overrides) -> BrokerDeal:
    values = dict(asset="EURUSD_otc", direction="CALL", entry_at=BASE,
                  exit_at=BASE + 60, open_price=1.0843, close_price=1.0851,
                  profit=0.85, percent_profit=85.0, is_demo=True)
    values.update(overrides)
    return BrokerDeal(**values)


def cfg(**overrides) -> Config:
    values = dict(feed="simulated", candle_period=60, expiry="1m", expiry_seconds=60,
                  lead_seconds=10, cooldown_seconds=120, signal_journal_path="s.jsonl")
    values.update(overrides)
    return Config(**values)


class ReconcileCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build_journal(self, entries) -> "reconcile.LoadedJournal":
        """``entries`` is a list of (entry_at, outcome or None)."""
        path = Path(self.tmp.name) / "signals.jsonl"
        journal = SignalJournal(path)
        for entry_at, outcome in entries:
            journal.record_signal(asset="EURUSD_otc", direction="CALL",
                                  entry_at=entry_at, expiry_at=entry_at + 60,
                                  payout=85, score=2, confidence=0.5)
            if outcome:
                journal.record_result(asset="EURUSD_otc", entry_at=entry_at,
                                      outcome=outcome, entry_price=1.08,
                                      exit_price=1.09)
        return load_journal(path)


class TestNormalisation(unittest.TestCase):
    def test_asset_enum_string_and_qualified_forms_agree(self):
        Asset, _ = sdk_types()
        self.assertEqual(normalize_asset(Asset.EURUSD_otc), "EURUSD_otc")
        self.assertEqual(normalize_asset("EURUSD_otc"), "EURUSD_otc")
        self.assertEqual(normalize_asset("Asset.EURUSD_otc"), "EURUSD_otc")
        self.assertEqual(normalize_asset(None), "")

    def test_directions_map_onto_call_and_put(self):
        _, DealAction = sdk_types()
        self.assertEqual(normalize_direction(DealAction.CALL), "CALL")
        self.assertEqual(normalize_direction(DealAction.PUT), "PUT")
        self.assertEqual(normalize_direction("call"), "CALL")
        self.assertEqual(normalize_direction("Sell"), "PUT")
        self.assertIsNone(normalize_direction("sideways"))
        self.assertIsNone(normalize_direction(None))

    def test_the_direction_enum_a_settled_deal_actually_carries_is_understood(self):
        # ``Deal.command`` is ``Command`` — an *integer* enum, CALL=0, PUT=1 —
        # while ``DealAction`` is the string enum you send. Reading ``.value``
        # off the former gives "0", which a lookup table will happily map onto a
        # direction; this test exists because that inversion shipped once.
        from pocket_option.models import Command

        self.assertEqual(normalize_direction(Command.CALL), "CALL")
        self.assertEqual(normalize_direction(Command.PUT), "PUT")

    def test_a_bare_number_is_refused_rather_than_turned_into_a_direction(self):
        # 0/1 are not universally call/put, and guessing wrong is silent.
        self.assertIsNone(normalize_direction(0))
        self.assertIsNone(normalize_direction(1))
        self.assertIsNone(normalize_direction("0"))
        self.assertIsNone(normalize_direction(True))

    def test_millisecond_timestamps_are_scaled_to_seconds(self):
        fields = ("open_timestamp", "open_time", "open_ms")
        self.assertEqual(epoch_seconds({"open_timestamp": BASE}, *fields), BASE)
        self.assertEqual(epoch_seconds({"open_ms": BASE * 1000}, *fields), BASE)
        self.assertIsNone(epoch_seconds({"open_timestamp": None}, *fields))

    def test_string_numbers_are_accepted(self):
        self.assertEqual(
            epoch_seconds({"open_time": str(int(BASE))}, "open_timestamp", "open_time"),
            float(BASE))


class TestDealParsing(ReconcileCase):
    def test_a_dict_deal_is_understood(self):
        parsed = deal_from_model({
            "asset": "EURUSD_otc", "command": "call", "profit": "0.85",
            "percent_profit": "85", "open_price": "1.0843", "close_price": "1.0851",
            "open_timestamp": BASE, "close_timestamp": BASE + 60, "is_demo": 1,
        })

        self.assertEqual(parsed.asset, "EURUSD_otc")
        self.assertEqual(parsed.direction, "CALL")
        self.assertEqual(parsed.entry_at, BASE)
        self.assertAlmostEqual(parsed.open_price, 1.0843)
        self.assertAlmostEqual(parsed.profit, 0.85)
        self.assertTrue(parsed.is_demo)
        self.assertEqual(parsed.outcome, WIN)

    def test_a_deal_object_built_from_the_sdk_enums_is_understood(self):
        Asset, DealAction = sdk_types()
        parsed = deal_from_model(types.SimpleNamespace(
            id="abc", asset=Asset.EURUSD_otc, command=DealAction.PUT,
            profit=-1.0, percent_profit=None, open_price=1.0843,
            close_price=1.0830, open_timestamp=BASE, close_timestamp=BASE + 60,
            is_demo=1, amount=1))

        self.assertEqual(parsed.direction, "PUT")
        self.assertEqual(parsed.outcome, LOSS)
        self.assertEqual(parsed.deal_id, "abc")

    def test_a_genuine_sdk_deal_parses_with_its_direction_intact(self):
        parsed = deal_from_model(real_deal())

        self.assertEqual(parsed.asset, "EURUSD_otc")
        self.assertEqual(parsed.direction, "CALL")
        self.assertEqual(parsed.outcome, WIN)
        self.assertAlmostEqual(parsed.open_price, 1.0843)
        self.assertAlmostEqual(parsed.profit, 0.85)
        self.assertEqual(parsed.entry_at, BASE)
        self.assertTrue(parsed.deal_id)

        put = deal_from_model(real_deal(command=sdk_command("PUT")))
        self.assertEqual(put.direction, "PUT")

    def test_a_deal_without_an_asset_is_rejected_rather_than_guessed(self):
        self.assertIsNone(deal_from_model({"command": "call", "profit": 1}))
        self.assertIsNone(deal_from_model({}))

    def test_a_refund_is_neither_a_win_nor_a_loss(self):
        self.assertEqual(deal(profit=0.0).outcome, PUSH)
        self.assertEqual(deal(profit=None).outcome, UNKNOWN)
        self.assertEqual(deal(profit=0.01).outcome, WIN)
        self.assertEqual(deal(profit=-0.01).outcome, LOSS)


class TestMatching(ReconcileCase):
    def test_a_deal_pairs_with_the_signal_on_the_same_market_and_minute(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(entry_at=BASE + 3)])

        self.assertEqual(len(rec.pairs), 1)
        self.assertEqual(rec.pairs[0].offset, 3.0)
        self.assertTrue(rec.pairs[0].agreed)

    def test_a_deal_beyond_the_tolerance_is_not_matched(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(entry_at=BASE + 90)], tolerance=20.0)

        self.assertEqual(rec.pairs, [])
        self.assertEqual(len(rec.unmatched_signals), 1)
        self.assertEqual(len(rec.unmatched_deals), 1)

    def test_a_deal_on_another_market_is_not_matched(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(asset="GBPUSD_otc", entry_at=BASE)])

        self.assertEqual(rec.pairs, [])

    def test_the_closest_deal_wins_and_is_claimed_once(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(entry_at=BASE + 15, deal_id="far"),
                              deal(entry_at=BASE + 1, deal_id="near")])

        self.assertEqual(len(rec.pairs), 1)
        self.assertEqual(rec.pairs[0].deal.deal_id, "near")
        self.assertEqual([d.deal_id for d in rec.unmatched_deals], ["far"])

    def test_a_deal_without_a_timestamp_is_excluded_from_matching(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(entry_at=None)])

        self.assertEqual(rec.pairs, [])
        self.assertEqual(len(rec.unmatched_deals), 1)

    def test_several_signals_pair_with_their_own_deals(self):
        journal = self.build_journal([(BASE, "WIN"), (BASE + 300, "LOSS")])
        rec = match(journal, [deal(entry_at=BASE + 2), deal(entry_at=BASE + 302)])

        self.assertEqual([p.signal.our_outcome for p in rec.pairs], ["WIN", "LOSS"])


class TestClockOffset(ReconcileCase):
    """The broker's timestamps are on a different clock, and it has to be measured.

    Live on the demo account: five consecutive deals opened 7197-7198 seconds
    after the second they were signalled for. Against a 20-second matching
    tolerance that is not a near miss — every deal falls outside the window, and
    the tool reports "no overlap between the two records" as though the trades had
    never been placed. It is not a round 7200 either, so the rule behind it cannot
    be derived here; it is read off the deals instead.
    """

    LIVE_OFFSET = 7198.0

    def test_a_two_hour_offset_is_measured_from_the_deals(self):
        journal = self.build_journal([(BASE + i * 60, "WIN") for i in range(5)])
        deals = [deal(entry_at=BASE + i * 60 + self.LIVE_OFFSET) for i in range(5)]

        offset, hits, considered = detect_clock_offset(journal.trades, deals)

        self.assertAlmostEqual(offset, self.LIVE_OFFSET, places=6)
        self.assertEqual((hits, considered), (5, 5))

    def test_the_offset_makes_deals_pair_that_otherwise_match_nothing(self):
        journal = self.build_journal([(BASE + i * 60, "WIN") for i in range(5)])
        deals = [deal(entry_at=BASE + i * 60 + self.LIVE_OFFSET) for i in range(5)]
        offset, _, _ = detect_clock_offset(journal.trades, deals)

        blind = match(journal, deals, tolerance=20.0)
        self.assertEqual(blind.pairs, [], "the raw clocks are two hours apart")

        aligned = match(journal, deals, tolerance=20.0, clock_offset=offset)
        self.assertEqual(len(aligned.pairs), 5)
        self.assertTrue(all(abs(p.offset) < 1e-6 for p in aligned.pairs),
                        "what is left over is the jitter the report shows")

    def test_a_deal_at_another_minute_does_not_drag_the_cluster(self):
        journal = self.build_journal([(BASE + i * 60, "WIN") for i in range(6)])
        deals = [deal(entry_at=BASE + i * 60 + self.LIVE_OFFSET) for i in range(4)]
        deals.append(deal(entry_at=BASE + 600 + self.LIVE_OFFSET))  # nothing to pair
        deals.append(deal(entry_at=BASE + 305))                     # already aligned

        offset, hits, _ = detect_clock_offset(journal.trades, deals)

        self.assertAlmostEqual(offset, self.LIVE_OFFSET, places=6)
        self.assertEqual(hits, 4, "the four real ones, not the strays")

    def test_an_already_aligned_account_measures_no_offset(self):
        journal = self.build_journal([(BASE + i * 60, "WIN") for i in range(3)])
        deals = [deal(entry_at=BASE + i * 60 + 2) for i in range(3)]

        offset, _, _ = detect_clock_offset(journal.trades, deals)

        self.assertAlmostEqual(offset, 2.0, places=6)
        self.assertEqual(len(match(journal, deals, 20.0, offset).pairs), 3)

    def test_one_coincidence_is_not_a_measurement(self):
        journal = self.build_journal([(BASE, "WIN")])

        offset, hits, considered = detect_clock_offset(
            journal.trades, [deal(entry_at=BASE + 5)])

        self.assertIsNone(offset)
        self.assertEqual((hits, considered), (0, 1))

    def test_another_market_cannot_calibrate_the_clock(self):
        journal = self.build_journal([(BASE + i * 60, "WIN") for i in range(3)])
        deals = [deal(asset="GBPUSD_otc", entry_at=BASE + i * 60 + self.LIVE_OFFSET)
                 for i in range(3)]

        offset, _, _ = detect_clock_offset(journal.trades, deals)

        self.assertIsNone(offset, "a deal on another market is another trade")

    def test_nothing_to_measure_is_not_an_error(self):
        self.assertEqual(detect_clock_offset([], [deal()]), (None, 0, 1))
        self.assertEqual(detect_clock_offset([], []), (None, 0, 0))


class TestComparison(ReconcileCase):
    def test_a_disagreeing_broker_is_counted_as_a_disagreement(self):
        journal = self.build_journal([(BASE, "WIN"), (BASE + 300, "WIN")])
        rec = match(journal, [deal(entry_at=BASE, profit=0.85),
                              deal(entry_at=BASE + 300, profit=-1.0)])

        agreed, compared = rec.agreements()
        self.assertEqual((agreed, compared), (1, 2))
        self.assertFalse(rec.pairs[1].agreed)

    def test_a_refund_is_left_out_of_the_agreement_denominator(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(profit=0.0)])

        self.assertEqual(rec.agreements(), (0, 0), "nothing comparable")
        self.assertIsNone(rec.pairs[0].agreed)

    def test_an_unsettled_signal_is_left_out_of_the_comparison(self):
        journal = self.build_journal([(BASE, None)])
        rec = match(journal, [deal()])

        self.assertEqual(rec.agreements(), (0, 0))
        self.assertEqual(len(rec.pairs), 1, "still matched, just not comparable")

    def test_a_deal_that_went_the_other_way_is_flagged(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(direction="PUT")])

        self.assertFalse(rec.pairs[0].direction_agreed)

    def test_an_unknown_direction_is_not_a_mismatch(self):
        journal = self.build_journal([(BASE, "WIN")])
        rec = match(journal, [deal(direction=None)])

        self.assertIsNone(rec.pairs[0].direction_agreed)

    def test_slippage_is_signed_by_what_it_costs_the_trade(self):
        journal = self.build_journal([(BASE, "WIN")])
        journal.trades[0].our_entry = 1.0840
        journal.trades[0].direction = "CALL"
        pip = pip_size("EURUSD_otc")

        # A CALL filled above our reference: adverse, so positive.
        above = match(journal, [deal(open_price=1.0845)])
        self.assertAlmostEqual(above.pairs[0].entry_slippage_pips(pip), 5.0, places=6)

        # A PUT filled below our reference is the adverse case too.
        journal.trades[0].direction = "PUT"
        below = match(journal, [deal(direction="PUT", open_price=1.0835)])
        self.assertAlmostEqual(below.pairs[0].entry_slippage_pips(pip), 5.0, places=6)

    def test_slippage_needs_both_prices(self):
        journal = self.build_journal([(BASE, "WIN")])
        journal.trades[0].our_entry = None
        rec = match(journal, [deal()])

        self.assertIsNone(rec.pairs[0].entry_slippage_pips(pip_size("EURUSD_otc")))

    def test_payouts_are_read_as_percentages_however_they_arrive(self):
        journal = self.build_journal([(BASE, "WIN"), (BASE + 300, "WIN")])
        rec = match(journal, [deal(percent_profit=85.0),
                              deal(entry_at=BASE + 300, percent_profit=0.82)])

        self.assertEqual(sorted(rec.payouts()), [82.0, 85.0])

    def test_the_broker_tally_separates_wins_losses_and_refunds(self):
        journal = self.build_journal([(BASE, "WIN"), (BASE + 300, "WIN"),
                                      (BASE + 600, "WIN")])
        rec = match(journal, [deal(profit=1.0), deal(entry_at=BASE + 300, profit=-1.0),
                              deal(entry_at=BASE + 600, profit=0.0)])

        self.assertEqual(rec.broker_tally(), {WIN: 1, LOSS: 1, PUSH: 1, UNKNOWN: 0})


class TestStatistics(unittest.TestCase):
    def test_break_even_matches_the_payout_arithmetic(self):
        self.assertAlmostEqual(break_even_win_rate(85), 1 / 1.85, places=9)
        self.assertAlmostEqual(break_even_win_rate(60), 0.625, places=9)
        self.assertAlmostEqual(break_even_win_rate(100), 0.5, places=9)
        self.assertEqual(break_even_win_rate(0), 1.0)

    def test_wilson_interval_brackets_the_rate_and_widens_when_thin(self):
        low, high = wilson(24, 38)
        self.assertLess(low, 24 / 38)
        self.assertGreater(high, 24 / 38)
        self.assertGreater(high - low, 0.2, "38 trades cannot pin a rate down")

        thin_low, thin_high = wilson(1, 2)
        self.assertGreater(thin_high - thin_low, 0.5)

    def test_wilson_survives_a_perfect_or_empty_record(self):
        self.assertEqual(wilson(0, 0), (0.0, 1.0))
        low, high = wilson(10, 10)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_no_sample_size_rescues_a_rate_below_break_even(self):
        self.assertIsNone(trades_needed(0.50, 0.549))
        self.assertIsNone(trades_needed(0.549, 0.549))

    def test_a_bigger_gap_needs_fewer_trades(self):
        wide = trades_needed(0.70, 0.549)
        narrow = trades_needed(0.56, 0.549)

        self.assertIsNotNone(wide)
        self.assertLess(wide, narrow)

    def test_the_needed_sample_is_the_expensive_kind_of_number(self):
        # A two-point edge over break-even is not provable in a week of signals.
        self.assertGreater(trades_needed(0.57, 0.549), 1000)

    def test_pip_size_follows_the_quote_convention(self):
        self.assertAlmostEqual(pip_size("EURUSD_otc"), 0.0001)
        self.assertAlmostEqual(pip_size("USDJPY_otc"), 0.01)
        self.assertAlmostEqual(pip_size("EURUSD_otc", digits=5), 0.0001)
        self.assertAlmostEqual(pip_size("USDJPY_otc", digits=3), 0.01)


class TestReport(ReconcileCase):
    def run_report(self, entries, deals, **kwargs):
        import io
        import contextlib
        journal = self.build_journal(entries)
        tolerance = kwargs.pop("tolerance", 20.0)
        offset = kwargs.pop("clock_offset", 0.0)
        rec = match(journal, deals, tolerance, offset)
        rec.unparsed_deals = kwargs.pop("unparsed", 0)
        rec.clock_offset = offset
        rec.offset_given = kwargs.pop("offset_given", False)
        rec.offset_hits = kwargs.pop("offset_hits", 0)
        rec.offset_considered = kwargs.pop("offset_considered", 0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report(rec, cfg(), tolerance, kwargs.pop("digits", {}))
        return buffer.getvalue()

    def test_an_empty_journal_says_so_instead_of_reporting_a_win_rate(self):
        text = self.run_report([], [])

        self.assertIn("Nothing to reconcile", text)
        self.assertNotIn("Break-even", text)

    def test_matched_deals_produce_an_agreement_rate_and_a_verdict(self):
        text = self.run_report(
            [(BASE + i * 60, "WIN") for i in range(4)],
            [deal(entry_at=BASE + i * 60, profit=1.0) for i in range(4)])

        self.assertIn("Label agreement: 4/4 = 100%", text)
        self.assertIn("Break-even at 85% payout: 54.1%", text)
        self.assertIn("not enough evidence yet", text)

    def test_a_disagreement_is_named_in_both_directions(self):
        text = self.run_report([(BASE, "WIN")], [deal(profit=-1.0)])

        self.assertIn("we said WIN, broker said LOSS", text)

    def test_a_measured_offset_is_reported_as_measured(self):
        text = self.run_report(
            [(BASE + i * 60, "WIN") for i in range(4)],
            [deal(entry_at=BASE + i * 60 + 7198, profit=1.0) for i in range(4)],
            clock_offset=7198.0, offset_hits=4, offset_considered=4)

        self.assertIn("broker clock is +7198s from ours", text)
        self.assertIn("measured", text)
        self.assertIn("4 of 4", text)
        self.assertIn("Label agreement: 4/4 = 100%", text)

    def test_an_offset_from_the_command_line_is_not_called_measured(self):
        text = self.run_report([(BASE, "WIN")], [deal(entry_at=BASE + 7198)],
                               clock_offset=7198.0, offset_given=True)

        self.assertIn("given with --offset", text)
        self.assertNotIn("measured", text)

    def test_an_unmatchable_record_suggests_the_clocks_may_disagree(self):
        # The failure that prompted all of this: the timestamps are hours apart,
        # no offset could be measured, and the old report blamed the operator.
        text = self.run_report([(BASE, "WIN")], [deal(entry_at=BASE + 7198)])

        self.assertIn("no offset", text)
        self.assertIn("--offset", text)

    def test_signals_without_deals_and_deals_without_signals_are_both_reported(self):
        text = self.run_report([(BASE, "WIN"), (BASE + 300, "WIN")],
                               [deal(entry_at=BASE)])

        self.assertIn("1 signal(s) with no deal", text)

    def test_a_deal_outside_the_bot_is_excluded_and_said_to_be(self):
        text = self.run_report([(BASE, "WIN")],
                               [deal(entry_at=BASE), deal(entry_at=BASE + 3600)])

        self.assertIn("1 deal(s) with no signal", text)
        self.assertIn("traded outside the bot", text)

    def test_nothing_in_common_is_reported_plainly(self):
        text = self.run_report([(BASE, "WIN")], [deal(entry_at=BASE + 3600)])

        self.assertIn("No overlap", text)

    def test_a_refund_is_reported_separately_from_wins_and_losses(self):
        text = self.run_report([(BASE, "WIN")], [deal(profit=0.0)])

        self.assertIn("refunded", text)

    def test_slippage_is_reported_in_pips_with_a_sign(self):
        entries = [(BASE, "WIN")]
        journal = self.build_journal(entries)
        journal.trades[0].our_entry = 1.0840
        rec = match(journal, [deal(open_price=1.0845)])
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report(rec, cfg(), 20.0, {"EURUSD_otc": 5})

        self.assertIn("positive = filled worse", buffer.getvalue())
        self.assertIn("+5.0", buffer.getvalue())

    def test_the_sample_size_caveat_is_always_printed(self):
        text = self.run_report([(BASE, "WIN")], [deal()])

        self.assertIn("trades *you* placed", text)


class TestDealsDump(ReconcileCase):
    def test_dumped_deals_can_be_read_back(self):
        path = Path(self.tmp.name) / "deals.json"
        path.write_text(json.dumps([
            {"asset": "EURUSD_otc", "command": "call", "profit": 0.85,
             "percent_profit": 85, "open_timestamp": BASE, "close_timestamp": BASE + 60,
             "open_price": 1.0843, "close_price": 1.0851, "is_demo": True},
            {"asset": None, "command": "call"},
        ]), encoding="utf-8")
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            deals, unparsed = reconcile.load_deals(str(path))

        self.assertEqual(len(deals), 1)
        self.assertEqual(unparsed, 1)
        self.assertEqual(deals[0].outcome, WIN)

    def test_a_dump_that_is_a_dict_with_a_deals_key_is_accepted(self):
        path = Path(self.tmp.name) / "deals.json"
        path.write_text(json.dumps({"deals": [{"asset": "EURUSD_otc", "profit": 1}]}),
                        encoding="utf-8")
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            deals, _ = reconcile.load_deals(str(path))

        self.assertEqual(len(deals), 1)

    def test_a_dumped_deal_keeps_its_direction_when_read_back(self):
        # The dump is the artifact used to check parsing against real data, so
        # serialising ``Command.CALL`` through ``str()`` — which is "0" — would
        # quietly destroy the very field it exists to verify.
        path = Path(self.tmp.name) / "deals.json"
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            reconcile._write_dump(str(path), [real_deal(),
                                              real_deal(command=sdk_command("PUT"))])
            deals, unparsed = reconcile.load_deals(str(path))

        self.assertEqual(unparsed, 0)
        self.assertEqual([d.direction for d in deals], ["CALL", "PUT"])
        self.assertEqual([d.outcome for d in deals], [WIN, WIN], "profit survives as a number")

        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw[0]["command"], "CALL", "the dump is readable by a human")


if __name__ == "__main__":
    unittest.main()
