"""Tests for the per-trade signal journal.

The journal is the audit trail the reconciliation stands on, and it is written
by a process that can be killed at any moment — so the tests care as much about
what it does with a damaged file as with a healthy one.
"""

import json
import tempfile
import unittest
from pathlib import Path

from journal import (
    KIND_RESULT, KIND_SIGNAL, SCHEMA_VERSION, ExpectedValue, SignalJournal,
    JournalledTrade, LoadedJournal, format_ev, load_journal, signal_id,
)

BASE = 1_700_000_040.0


class JournalCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "signals.jsonl"
        self.journal = SignalJournal(self.path)

    def write_lines(self, *records) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def signal(self, **overrides):
        values = dict(asset="EURUSD_otc", direction="CALL", entry_at=BASE,
                      expiry_at=BASE + 60, payout=85, score=2, confidence=0.5,
                      votes=["RSI", "BB"], sent_at=BASE - 10)
        values.update(overrides)
        self.journal.record_signal(**values)


class TestSignalId(unittest.TestCase):
    def test_the_id_is_market_plus_the_entry_second(self):
        self.assertEqual(signal_id("EURUSD_otc", BASE), f"EURUSD_otc@{int(BASE)}")

    def test_prices_do_not_change_the_id(self):
        # Two ticks inside the same second are the same signal.
        self.assertEqual(signal_id("EURUSD_otc", BASE), signal_id("EURUSD_otc", BASE + 0.5))

    def test_different_markets_or_seconds_are_different_signals(self):
        self.assertNotEqual(signal_id("EURUSD_otc", BASE), signal_id("GBPUSD_otc", BASE))
        self.assertNotEqual(signal_id("EURUSD_otc", BASE), signal_id("EURUSD_otc", BASE + 60))


class TestWriting(JournalCase):
    def test_a_signal_round_trips(self):
        self.signal()
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 1)
        trade = loaded.trades[0]
        self.assertEqual(trade.asset, "EURUSD_otc")
        self.assertEqual(trade.direction, "CALL")
        self.assertEqual(trade.entry_at, BASE)
        self.assertEqual(trade.expiry_at, BASE + 60)
        self.assertEqual(trade.payout, 85)
        self.assertEqual(trade.votes, ("RSI", "BB"))
        self.assertFalse(trade.settled, "no result yet")

    def test_a_result_folds_into_its_signal(self):
        self.signal()
        self.journal.record_result(asset="EURUSD_otc", entry_at=BASE, outcome="WIN",
                                   entry_price=1.0843, exit_price=1.0851,
                                   settled_at=BASE + 110)
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 1, "one trade, not two records")
        trade = loaded.trades[0]
        self.assertTrue(trade.settled)
        self.assertEqual(trade.our_outcome, "WIN")
        self.assertAlmostEqual(trade.our_entry, 1.0843)
        self.assertAlmostEqual(trade.our_exit, 1.0851)

    def test_records_are_appended_not_rewritten(self):
        self.signal(entry_at=BASE)
        self.signal(entry_at=BASE + 60)
        self.signal(entry_at=BASE + 120)

        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 3)
        self.assertAlmostEqual(load_journal(self.path).trades[1].entry_at, BASE + 60)

    def test_every_line_carries_a_schema_version(self):
        self.signal()
        record = json.loads(self.path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["kind"], KIND_SIGNAL)
        self.assertEqual(record["version"], SCHEMA_VERSION)

    def test_a_disabled_journal_writes_nothing(self):
        SignalJournal(self.path, enabled=False).record_signal(
            asset="EURUSD_otc", direction="CALL", entry_at=BASE, expiry_at=BASE + 60)

        self.assertFalse(self.path.exists())

    def test_an_unwritable_path_does_not_raise(self):
        # A directory where the file should be: recording fails, trading must not.
        blocked = Path(self.tmp.name) / "occupied"
        blocked.mkdir()
        SignalJournal(blocked).record_signal(
            asset="EURUSD_otc", direction="CALL", entry_at=BASE, expiry_at=BASE + 60)

    def test_the_parent_directory_is_created(self):
        nested = Path(self.tmp.name) / "a" / "b" / "signals.jsonl"
        SignalJournal(nested).record_signal(
            asset="EURUSD_otc", direction="CALL", entry_at=BASE, expiry_at=BASE + 60)

        self.assertTrue(nested.exists())


class TestReading(JournalCase):
    def test_a_missing_file_is_an_empty_journal(self):
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 0)
        self.assertEqual(loaded.bad_lines, 0)

    def test_a_truncated_final_line_is_skipped_not_fatal(self):
        self.signal()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"kind":"signal","asset":"EURUSD_otc","direc')   # killed mid-write

        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 1, "the good record survives")
        self.assertEqual(loaded.bad_lines, 1)

    def test_junk_and_unknown_kinds_are_counted(self):
        self.write_lines(
            {"kind": KIND_SIGNAL, "asset": "EURUSD_otc", "direction": "CALL",
             "entry_at": BASE, "expiry_at": BASE + 60},
            "not json at all",
            {"kind": "something-else", "asset": "EURUSD_otc"},
            {"kind": KIND_RESULT, "id": "EURUSD_otc@" + str(int(BASE)), "outcome": "WIN"},
        )
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded.trades[0].settled, "the matching result still lands")
        self.assertEqual(loaded.bad_lines, 2)

    def test_a_signal_without_a_direction_or_time_is_unusable(self):
        self.write_lines(
            {"kind": KIND_SIGNAL, "asset": "EURUSD_otc", "entry_at": BASE},
            {"kind": KIND_SIGNAL, "asset": "EURUSD_otc", "direction": "CALL"},
            {"kind": KIND_SIGNAL, "direction": "CALL", "entry_at": BASE,
             "expiry_at": BASE + 60},
        )
        self.assertEqual(len(load_journal(self.path)), 0)

    def test_a_result_with_no_signal_is_counted_not_invented(self):
        self.write_lines({"kind": KIND_RESULT, "id": "EURUSD_otc@1", "outcome": "WIN",
                          "asset": "EURUSD_otc", "entry_at": 1})
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 0)
        self.assertEqual(loaded.unknown_results, 1)

    def test_two_results_for_one_signal_keep_the_last(self):
        self.signal()
        self.journal.record_result(asset="EURUSD_otc", entry_at=BASE, outcome="WIN")
        self.journal.record_result(asset="EURUSD_otc", entry_at=BASE, outcome="LOSS")

        self.assertEqual(load_journal(self.path).trades[0].our_outcome, "LOSS")

    def test_trades_come_back_in_time_order(self):
        for offset in (120, 0, 60):
            self.signal(entry_at=BASE + offset, expiry_at=BASE + offset + 60)

        entries = [t.entry_at for t in load_journal(self.path)]
        self.assertEqual(entries, sorted(entries))


class TestQueries(JournalCase):
    def setUp(self):
        super().setUp()
        for offset, outcome in ((0, "WIN"), (60, "LOSS"), (120, "WIN"), (180, None)):
            self.signal(entry_at=BASE + offset, expiry_at=BASE + offset + 60)
            if outcome:
                self.journal.record_result(asset="EURUSD_otc", entry_at=BASE + offset,
                                           outcome=outcome)
        self.loaded = load_journal(self.path)

    def test_settled_and_unsettled_are_separated(self):
        self.assertEqual(len(self.loaded.settled), 3)
        self.assertEqual(len(self.loaded.unsettled), 1)
        self.assertEqual(self.loaded.unsettled[0].entry_at, BASE + 180)

    def test_outcomes_and_win_rate(self):
        self.assertEqual(self.loaded.outcomes(), {"WIN": 2, "LOSS": 1})
        self.assertAlmostEqual(self.loaded.our_win_rate(), 2 / 3)

    def test_a_window_selects_by_entry_time(self):
        window = self.loaded.between(BASE + 30, BASE + 150)

        self.assertEqual([t.entry_at for t in window], [BASE + 60, BASE + 120])

    def test_a_trade_carries_its_own_id(self):
        self.assertEqual(self.loaded.trades[0].id, f"EURUSD_otc@{int(BASE)}")

    def test_an_empty_journal_has_a_zero_win_rate(self):
        self.assertEqual(load_journal(Path(self.tmp.name) / "none.jsonl").our_win_rate(), 0.0)

    def test_price_lookup_by_name(self):
        trade = JournalledTrade(asset="EURUSD_otc", direction="CALL", entry_at=BASE,
                                expiry_at=BASE + 60, our_entry=1.0, our_exit=1.1)
        self.assertEqual(trade.price_of("entry"), 1.0)
        self.assertEqual(trade.price_of("exit"), 1.1)


class TestTheBrokerSide(JournalCase):
    """The block the automated build adds: what the account actually did.

    The journal predates auto-trading, so the old shape — no broker block at all
    — has to keep meaning "nobody traded this", and must never be read as "the
    broker said nothing".
    """

    def settle(self, entry_at=BASE, outcome="WIN", broker=None, **overrides):
        values = dict(asset="EURUSD_otc", entry_at=entry_at, outcome=outcome,
                      entry_price=1.0843, exit_price=1.0851)
        values.update(overrides)
        self.journal.record_result(broker=broker, **values)

    def broker_block(self, outcome="WIN", **overrides):
        block = dict(outcome=outcome, deal_id="42", open_at=BASE,
                     close_at=BASE + 60, entry=1.0844, exit=1.0852,
                     profit=0.85, payout=85.0)
        block.update(overrides)
        return block

    def settled_with(self, *pairs):
        """Signals settled with ``(our label, broker block or None)`` each."""
        for i, (ours, theirs) in enumerate(pairs):
            entry_at = BASE + i * 60
            self.signal(entry_at=entry_at, expiry_at=entry_at + 60)
            self.settle(entry_at=entry_at, outcome=ours,
                        broker=self.broker_block(outcome=theirs) if theirs else None)
        return load_journal(self.path)

    def test_a_broker_block_round_trips(self):
        self.signal()
        self.settle(broker=self.broker_block())

        trade = load_journal(self.path).trades[0]

        self.assertTrue(trade.has_broker)
        self.assertEqual(trade.broker_outcome, "WIN")
        self.assertEqual(trade.deal_id, "42")
        self.assertAlmostEqual(trade.broker_entry, 1.0844)
        self.assertAlmostEqual(trade.broker_exit, 1.0852)
        self.assertAlmostEqual(trade.broker_profit, 0.85)
        self.assertAlmostEqual(trade.broker_payout, 85.0)
        self.assertAlmostEqual(trade.broker_close_at, BASE + 60)

    def test_the_broker_s_answer_is_the_outcome_of_record(self):
        # Our label is a reading of two closes; theirs is the money.
        self.signal()
        self.settle(outcome="LOSS", broker=self.broker_block(outcome="WIN"))

        trade = load_journal(self.path).trades[0]

        self.assertEqual(trade.our_outcome, "LOSS")
        self.assertEqual(trade.outcome_of_record, "WIN")

    def test_a_journal_written_before_auto_trading_still_reads(self):
        # The exact line shape the old build wrote: no broker key at all.
        self.write_lines(
            {"kind": KIND_SIGNAL, "id": f"EURUSD_otc@{int(BASE)}",
             "asset": "EURUSD_otc", "direction": "CALL", "entry_at": BASE,
             "expiry_at": BASE + 60, "version": SCHEMA_VERSION},
            {"kind": KIND_RESULT, "id": f"EURUSD_otc@{int(BASE)}",
             "outcome": "WIN", "version": SCHEMA_VERSION})
        loaded = load_journal(self.path)

        self.assertEqual(len(loaded), 1)
        trade = loaded.trades[0]
        self.assertFalse(trade.has_broker, "nobody traded this one")
        self.assertEqual(trade.outcome_of_record, "WIN", "our own label stands")
        self.assertEqual(loaded.placed, [])

    def test_the_broker_win_rate_excludes_refunds(self):
        loaded = self.settled_with(("WIN", "WIN"), ("WIN", "PUSH"), ("LOSS", "LOSS"))

        self.assertEqual(loaded.broker_outcomes(), {"WIN": 1, "PUSH": 1, "LOSS": 1})
        self.assertAlmostEqual(loaded.broker_win_rate(), 0.5, "1 of the 2 decided")

    def test_a_refund_alone_leaves_no_rate_to_report(self):
        loaded = self.settled_with(("WIN", "PUSH"))

        self.assertIsNone(loaded.broker_win_rate(),
                          "a refund decides nothing, so there is no rate")

    def test_agreement_counts_only_decided_trades_on_both_sides(self):
        loaded = self.settled_with(
            ("WIN", "WIN"),      # agree
            ("LOSS", "LOSS"),    # agree
            ("LOSS", "WIN"),     # comparable, and they disagree
            ("WIN", "PUSH"),     # a refund is not a disagreement
            ("WIN", None),       # not traded, so nothing to compare against
        )

        self.assertEqual(loaded.agreements(), (2, 3), "2 matched of 3 comparable")

    def test_an_unplaced_order_is_recorded_but_decides_nothing(self):
        loaded = self.settled_with(("WIN", "UNPLACED"))

        self.assertTrue(loaded.placed, "it was attempted, so it is on the record")
        self.assertEqual(loaded.broker_outcomes(), {"UNPLACED": 1})
        self.assertIsNone(loaded.broker_win_rate())

    def test_the_block_keeps_only_what_it_was_given(self):
        # A sparse block: an absent profit must stay absent, not become 0.0.
        self.signal()
        self.settle(broker={"outcome": "UNPLACED", "error": "min_amount"})
        loaded = load_journal(self.path)

        self.assertEqual(loaded.trades[0].broker_outcome, "UNPLACED")
        self.assertIsNone(loaded.trades[0].broker_profit)
        self.assertEqual(loaded.trades[0].deal_id, "")
        record = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertNotIn("profit", record["broker"])

    def test_numbers_written_as_strings_come_back_as_numbers(self):
        self.signal()
        self.settle(broker=self.broker_block(entry="1.0844", profit="0.85"))

        trade = load_journal(self.path).trades[0]

        self.assertAlmostEqual(trade.broker_entry, 1.0844)
        self.assertAlmostEqual(trade.broker_profit, 0.85)


class TestWhatTheRecordIsWorth(unittest.TestCase):
    """The per-trade return, and the interval that says how little is known.

    A win rate cannot be read on its own here, because the payout is not fixed:
    the same 53% of wins loses money at a 92% payout (break-even 52.1%) and
    makes it at 60% (break-even 62.5%). These tests are about the number that
    *is* readable, and about the sample sizes at which it starts to mean
    anything.
    """

    def trade(self, outcome, payout=0, broker_payout=None, **overrides):
        values = dict(asset="EURUSD_otc", direction="CALL", entry_at=BASE,
                      expiry_at=BASE + 60, payout=payout, our_outcome=outcome)
        if broker_payout is not None:
            # The money's own answer, which is the outcome of record.
            values["broker_outcome"] = outcome
            values["broker_payout"] = broker_payout
        values.update(overrides)
        return JournalledTrade(**values)

    def sample(self, wins, losses, payout=92, pushes=0):
        """A loaded journal of ``wins``/``losses``/``pushes`` at one payout."""
        trades = [self.trade("WIN", payout) for _ in range(wins)]
        trades += [self.trade("LOSS", payout) for _ in range(losses)]
        trades += [self.trade("PUSH", payout) for _ in range(pushes)]
        return LoadedJournal(trades=trades)

    def test_a_win_pays_its_payout_and_a_loss_takes_the_stake(self):
        ev = self.sample(wins=1, losses=1).expected_value()

        self.assertEqual(ev.n, 2)
        self.assertAlmostEqual(ev.mean, (0.92 - 1.0) / 2)

    def test_a_refund_returns_the_stake_untouched(self):
        ev = self.sample(wins=1, losses=1, pushes=1).expected_value()

        self.assertEqual(ev.n, 3, "a refund is a data point at zero, not a gap")
        self.assertAlmostEqual(ev.mean, (0.92 - 1.0 + 0.0) / 3)

    def test_the_broker_s_payout_is_the_one_that_paid(self):
        # Quoted 60% at signal time, settled at 92%: the money is what happened.
        loaded = LoadedJournal(trades=[self.trade("WIN", 60, broker_payout=92.0)])

        self.assertAlmostEqual(loaded.expected_value().mean, 0.92)

    def test_a_loss_needs_no_payout_because_its_return_does_not_depend_on_one(self):
        loaded = LoadedJournal(trades=[self.trade("LOSS", payout=0)])

        self.assertEqual(loaded.expected_value().n, 1)
        self.assertAlmostEqual(loaded.expected_value().mean, -1.0)

    def test_a_win_with_no_payout_is_left_out_and_counted(self):
        # Guessing a payout for it would invent the number the whole figure
        # rests on; leaving it out can only pull the average down, never up.
        loaded = LoadedJournal(trades=[self.trade("WIN", payout=0),
                                       self.trade("LOSS", payout=92)])

        ev = loaded.expected_value()

        self.assertEqual(ev.n, 1, "only the loss is priced")
        self.assertAlmostEqual(ev.mean, -1.0)
        self.assertEqual(ev.unpriced, 1)
        self.assertEqual(ev.excluded, 1)

    def test_a_settlement_that_is_neither_is_left_out_and_counted(self):
        loaded = LoadedJournal(trades=[self.trade("UNKNOWN", payout=92),
                                       self.trade("WIN", payout=92)])

        ev = loaded.expected_value()

        self.assertEqual(ev.n, 1)
        self.assertEqual(ev.unjudged, 1)
        self.assertEqual(ev.unpriced, 0, "an unreadable answer is not an unpriced win")

    def test_an_unsettled_signal_is_not_a_trade(self):
        loaded = LoadedJournal(trades=[self.trade(None, payout=92)])

        ev = loaded.expected_value()

        self.assertEqual(ev.n, 0)
        self.assertEqual(ev.excluded, 0, "never settled is not the same as dropped")

    def test_a_majority_of_wins_can_still_be_a_losing_record(self):
        # The whole reason this exists: 52% reads like a winning record and is
        # a hair under break-even at a 92% payout.
        loaded = self.sample(wins=52, losses=48, payout=92)

        self.assertGreater(loaded.our_win_rate(), 0.5)
        self.assertLess(loaded.expected_value().mean, 0.0)

    def test_the_interval_narrows_as_the_sample_grows(self):
        small = self.sample(wins=5, losses=5, payout=60).expected_value()
        large = self.sample(wins=500, losses=500, payout=60).expected_value()

        self.assertAlmostEqual(small.mean, large.mean, places=6)
        self.assertLess(large.high - large.low, small.high - small.low)

    def test_a_losing_sample_clears_zero_on_the_other_side(self):
        ev = self.sample(wins=550, losses=450, payout=60).expected_value()

        self.assertLess(ev.high, 0.0)
        self.assertEqual(ev.verdict, "under break-even")

    def test_a_sample_that_straddles_break_even_says_so(self):
        ev = self.sample(wins=52, losses=48, payout=92).expected_value()

        self.assertLess(ev.low, 0.0)
        self.assertGreater(ev.high, 0.0)
        self.assertEqual(ev.verdict, "no edge shown yet")

    def test_one_trade_has_no_interval_to_report(self):
        ev = self.sample(wins=1, losses=0, payout=92).expected_value()

        self.assertEqual(ev.n, 1)
        self.assertIsNone(ev.low, "one point has no spread")
        self.assertIsNone(ev.high)
        self.assertEqual(ev.verdict, "too few to judge")

    def test_nothing_to_report_renders_as_nothing(self):
        self.assertEqual(format_ev(ExpectedValue()), "")
        self.assertEqual(format_ev(LoadedJournal().expected_value()), "")

    def test_the_line_carries_the_mean_the_interval_and_the_count(self):
        small = format_ev(self.sample(wins=5, losses=5, payout=60).expected_value())

        self.assertIn("EV -0.200 per trade", small)
        self.assertIn("95% CI", small)
        self.assertIn("n=10", small)
        self.assertIn("no edge shown yet", small)
        self.assertNotIn("<", small, "this goes into an HTML message unescaped")
        self.assertNotIn(">", small)

    def test_the_line_is_ascii_so_the_log_can_hold_it(self):
        # The startup line goes through logging to a stream Windows opens in the
        # locale codepage: one emoji and the whole line is dropped, not mangled.
        line = format_ev(self.sample(wins=3, losses=2, payout=92).expected_value())

        line.encode("ascii")

    def test_the_line_names_what_it_left_out(self):
        loaded = LoadedJournal(trades=[self.trade("WIN", payout=0),
                                       self.trade("LOSS", payout=92),
                                       self.trade("UNKNOWN", payout=92)])
        line = format_ev(loaded.expected_value())

        self.assertIn("1 win(s) unpriced", line)
        self.assertIn("1 unreadable", line)

    def test_nothing_priced_but_something_dropped_says_why(self):
        # The failure this guards: an unpriced win silently vanishing reads as
        # "no trades yet", when in fact there was a trade and its payout is gone.
        loaded = LoadedJournal(trades=[self.trade("WIN", payout=0)])

        line = format_ev(loaded.expected_value())

        self.assertEqual(line, "EV unknown | 1 win(s) unpriced")

    def test_the_report_is_built_from_a_file_the_bot_wrote(self):
        # End to end: a signal and its result, through the writer and the reader.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "signals.jsonl"
        journal = SignalJournal(path)
        for i, outcome in enumerate(("WIN", "LOSS")):
            entry_at = BASE + i * 60
            journal.record_signal(asset="EURUSD_otc", direction="CALL",
                                  entry_at=entry_at, expiry_at=entry_at + 60,
                                  payout=92)
            journal.record_result(asset="EURUSD_otc", entry_at=entry_at,
                                  outcome=outcome, entry_price=1.0, exit_price=1.1)

        ev = load_journal(path).expected_value()

        self.assertEqual(ev.n, 2)
        self.assertAlmostEqual(ev.mean, -0.04)
        self.assertIn("n=2", format_ev(ev))


if __name__ == "__main__":
    unittest.main()
