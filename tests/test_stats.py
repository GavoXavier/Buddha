"""Tests for win/loss accounting, streaks, hourly buckets and persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from stats import StatsTracker, Tally

# 2023-11-14 22:13:20 UTC — a known instant for the timezone arithmetic.
AT = 1_700_000_000.0


class StatsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.path = self.dir / "stats.json"
        self.legacy = self.dir / "winrate.json"
        self.stats = StatsTracker(str(self.path), str(self.legacy))

    def new_tracker(self, tz: float = 3.0) -> StatsTracker:
        return StatsTracker(str(self.path), str(self.legacy), tz_offset_hours=tz)


class TestTally(unittest.TestCase):
    def test_win_rate(self):
        t = Tally()
        self.assertEqual(t.total, 0)
        self.assertEqual(t.win_rate, 0.0, "no division by zero on an empty tally")
        t.record("WIN")
        t.record("LOSS")
        t.record("WIN")
        self.assertEqual((t.wins, t.losses, t.total), (2, 1, 3))
        self.assertAlmostEqual(t.win_rate, 2 / 3)

    def test_anything_that_is_not_a_win_is_a_loss(self):
        t = Tally()
        t.record("WIN")
        t.record("LOSS")
        t.record("WIN")
        t.record("LOSS")
        t.record("LOSS")
        self.assertEqual(t.wins, 2)
        self.assertEqual(t.losses, 3)

    def test_round_trip(self):
        t = Tally(wins=4, losses=1)
        self.assertEqual(Tally.from_dict(t.as_dict()), t)
        self.assertEqual(Tally.from_dict({}), Tally(), "missing keys are zeroes")


class TestRecording(StatsCase):
    def test_records_land_in_the_totals_and_recent_list(self):
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.record("EURUSD_otc", "PUT", "LOSS", at=AT + 60)

        self.assertEqual(self.stats.total, 2)
        self.assertEqual(self.stats.wins, 1)
        self.assertEqual(self.stats.losses, 1)
        self.assertEqual(self.stats.win_rate, 0.5)
        self.assertEqual(len(self.stats.recent), 2)
        self.assertEqual(self.stats.recent[-1]["asset"], "EURUSD_otc")

    def test_per_asset_tally(self):
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.record("GBPJPY_otc", "PUT", "LOSS", at=AT)

        self.assertEqual(self.stats.asset_tally("EURUSD_otc").win_rate, 1.0)
        self.assertEqual(self.stats.asset_tally("GBPJPY_otc").win_rate, 0.0)
        self.assertEqual(self.stats.asset_tally("NOPE_otc").total, 0)

    def test_streaks(self):
        for outcome in ("WIN", "WIN", "WIN"):
            self.stats.record("A", "CALL", outcome, at=AT)
        self.assertEqual(self.stats.streak, 3)
        self.assertEqual(self.stats.best_streak, 3)

        for outcome in ("LOSS", "LOSS"):
            self.stats.record("A", "CALL", outcome, at=AT)
        self.assertEqual(self.stats.streak, -2)
        self.assertEqual(self.stats.consecutive_losses(), 2)
        self.assertEqual(self.stats.worst_streak, -2)
        self.assertEqual(self.stats.best_streak, 3, "the best streak is remembered")

        self.stats.record("A", "CALL", "WIN", at=AT)
        self.assertEqual(self.stats.streak, 1, "a win resets the losing streak")
        self.assertEqual(self.stats.consecutive_losses(), 0)

    def test_hour_bucket_uses_the_market_timezone(self):
        self.stats.record("A", "CALL", "WIN", at=AT)      # 22:13 UTC
        self.assertEqual(self.stats.hour_tally(1).total, 1, "UTC+3 -> 01:00")
        self.assertEqual(self.stats.hour_tally(22).total, 0)

        utc = StatsTracker(str(self.dir / "utc.json"), None, tz_offset_hours=0.0)
        utc.record("A", "CALL", "WIN", at=AT)
        self.assertEqual(utc.hour_tally(22).total, 1)
        self.assertEqual(utc.hour_tally(1).total, 0)

    def test_hour_bucket_wraps_past_midnight(self):
        self.stats.record("A", "CALL", "WIN", at=AT + 3600)   # 23:13 UTC -> 02:00
        self.assertEqual(self.stats.hour_tally(2).total, 1)

    def test_recent_list_is_capped(self):
        for i in range(70):
            self.stats.record("A", "CALL", "WIN", at=AT + i)
        self.assertEqual(len(self.stats.recent), 50)
        self.assertEqual(self.stats.recent[-1]["at"], AT + 69, "keeps the newest")


class TestPersistence(StatsCase):
    def test_survives_a_restart(self):
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.record("GBPJPY_otc", "PUT", "LOSS", at=AT)
        self.stats.save(force=True)

        reloaded = self.new_tracker()
        self.assertEqual(reloaded.total, 2)
        self.assertEqual(reloaded.asset_tally("EURUSD_otc").wins, 1)
        self.assertEqual(reloaded.hour_tally(1).losses, 1, "hour keys reload as ints")
        self.assertEqual(reloaded.streak, -1)

    def test_writes_are_debounced_but_can_be_forced(self):
        self.stats.record("A", "CALL", "WIN", at=AT)
        self.assertEqual(json.loads(self.path.read_text())["global"]["wins"], 1)

        self.stats.record("A", "CALL", "WIN", at=AT)   # save() is debounced
        self.assertEqual(json.loads(self.path.read_text())["global"]["wins"], 1)

        self.stats.save(force=True)
        self.assertEqual(json.loads(self.path.read_text())["global"]["wins"], 2)

    def test_missing_file_starts_empty(self):
        self.assertEqual(self.new_tracker().total, 0)

    def test_corrupt_file_is_ignored_not_fatal(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.new_tracker().total, 0)

    def test_unknown_schema_version_starts_fresh(self):
        self.path.write_text(json.dumps({"version": 99, "global": {"wins": 5}}),
                             encoding="utf-8")
        self.assertEqual(self.new_tracker().total, 0)

    def test_legacy_winrate_file_is_kept_but_not_counted(self):
        self.legacy.write_text(json.dumps({"wins": 7, "losses": 3}), encoding="utf-8")

        imported = self.new_tracker()
        self.assertEqual((imported.legacy.wins, imported.legacy.losses), (7, 3))
        # Not folded into the measured record: those totals have no trades
        # behind them, so counting them would put a headline above tables that
        # cannot add up to it.
        self.assertEqual(imported.total, 0)
        self.assertEqual((imported.wins, imported.losses), (0, 0))

    def test_legacy_totals_invent_no_streak(self):
        self.legacy.write_text(json.dumps({"wins": 7, "losses": 3}), encoding="utf-8")
        imported = self.new_tracker()
        # The old code set best_streak = wins, claiming seven wins in a row
        # that no result ever recorded.
        self.assertEqual(imported.best_streak, 0)
        self.assertEqual(imported.streak, 0)

    def test_legacy_is_reported_apart_from_the_record(self):
        self.legacy.write_text(json.dumps({"wins": 7, "losses": 3}), encoding="utf-8")
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.save(force=True)

        text = self.new_tracker().summary_text()
        self.assertIn("1  (1W / 0L)", text)
        self.assertIn("Legacy record (imported, not counted above): 7W/3L 70%", text)

    def test_legacy_is_imported_once_and_then_loaded(self):
        self.legacy.write_text(json.dumps({"wins": 7, "losses": 3}), encoding="utf-8")
        self.stats.record("A", "CALL", "WIN", at=AT)
        self.stats.save(force=True)

        self.stats.legacy = Tally(wins=7, losses=3)
        self.stats.save(force=True)
        # Changing the legacy file must not change a record already adopted.
        self.legacy.write_text(json.dumps({"wins": 99, "losses": 99}), encoding="utf-8")
        self.assertEqual(self.new_tracker().legacy.total, 10)

    def test_legacy_import_does_not_override_real_stats(self):
        self.stats.record("A", "CALL", "WIN", at=AT)
        self.stats.save(force=True)
        self.legacy.write_text(json.dumps({"wins": 99, "losses": 99}), encoding="utf-8")
        reloaded = self.new_tracker()
        self.assertEqual(reloaded.total, 1)
        self.assertEqual(reloaded.legacy.total, 198)

    def test_missing_legacy_file_is_fine(self):
        self.assertFalse(self.legacy.exists())
        self.assertEqual(self.new_tracker().total, 0)


class TestTheFoldedInLegacyIsTakenBackOut(StatsCase):
    """The record repaired on load, for files the old importer inflated.

    Observed live on 2026-09-15: the log read ``12W/7L = 63%`` for a journal
    holding 15 settled trades, 9W/6L, with the difference being exactly the
    3W/1L printed under it as "kept out of that figure". The old importer set
    ``overall`` to the winrate.json totals and the real results were added on
    top of them; the fix stops new files being written that way, and this puts
    an already-written one right.
    """

    REAL = (9, 6)
    LEGACY = (3, 1)

    def folded_in_file(self, real=REAL, legacy=LEGACY, with_legacy_field=False):
        """An inflated stats file, in either of the two shapes that exist."""
        wins, losses = real
        doc = {
            "version": 1,
            "global": {"wins": wins + legacy[0], "losses": losses + legacy[1]},
            "by_asset": {"EURUSD_otc": {"wins": wins, "losses": losses}},
            "by_hour": {"20": {"wins": wins, "losses": losses}},
            "recent": [],
        }
        if with_legacy_field:
            doc["legacy"] = {"wins": legacy[0], "losses": legacy[1]}
        self.path.write_text(json.dumps(doc), encoding="utf-8")
        self.legacy.write_text(json.dumps({"wins": legacy[0], "losses": legacy[1]}),
                               encoding="utf-8")
        return self.new_tracker()

    def test_the_totals_come_back_out_of_the_record(self):
        repaired = self.folded_in_file()

        self.assertEqual((repaired.wins, repaired.losses), self.REAL)
        self.assertEqual(repaired.legacy.total, 4, "still kept, still apart")

    def test_the_headline_agrees_with_the_tables_again(self):
        text = self.folded_in_file().summary_text()

        self.assertIn("15  (9W / 6L)", text)
        self.assertIn("Legacy record (imported, not counted above): 3W/1L", text)

    def test_the_per_market_and_per_hour_tables_are_untouched(self):
        repaired = self.folded_in_file()

        self.assertEqual(repaired.by_asset["EURUSD_otc"].total, 15)
        self.assertEqual(repaired.by_hour[20].total, 15)

    def test_a_file_from_after_the_fix_is_repaired_as_well(self):
        # The live file: the tally is stored beside the record, so waiting for
        # a file with no "legacy" field would never have reached it.
        self.assertEqual(self.folded_in_file(with_legacy_field=True).wins, 9)

    def test_the_repair_is_not_applied_twice(self):
        self.folded_in_file().save(force=True)

        again = self.new_tracker()
        self.assertEqual((again.wins, again.losses), self.REAL,
                         "a repaired file must survive a reload unsullied")

    def test_a_correct_file_is_left_alone(self):
        self.path.write_text(json.dumps({
            "version": 1,
            "global": {"wins": 9, "losses": 6},
            "by_asset": {"EURUSD_otc": {"wins": 9, "losses": 6}},
            "legacy": {"wins": 3, "losses": 1},
        }), encoding="utf-8")

        self.assertEqual(self.new_tracker().wins, 9)

    def test_a_file_with_no_legacy_tally_to_remove_is_left_alone(self):
        self.path.write_text(json.dumps({
            "version": 1,
            "global": {"wins": 9, "losses": 6},
            "by_asset": {"EURUSD_otc": {"wins": 9, "losses": 6}},
        }), encoding="utf-8")
        self.assertFalse(self.legacy.exists())

        self.assertEqual(self.new_tracker().wins, 9)

    def test_a_difference_that_is_not_the_legacy_total_is_left_alone(self):
        # Something else made the headline disagree with the tables. Removing
        # the legacy tally would be a guess dressed up as a repair.
        self.path.write_text(json.dumps({
            "version": 1,
            "global": {"wins": 14, "losses": 6},
            "by_asset": {"EURUSD_otc": {"wins": 9, "losses": 6}},
        }), encoding="utf-8")
        self.legacy.write_text(json.dumps({"wins": 3, "losses": 1}), encoding="utf-8")

        self.assertEqual(self.new_tracker().wins, 14)


class TestReporting(StatsCase):
    def test_empty_report(self):
        self.assertEqual(self.stats.summary_text(), "No completed trades yet.")
        self.assertEqual(self.stats.hourly_table(), "No hourly data yet.")
        self.assertEqual(self.stats.asset_table(), "No per-market data yet.")

    def test_summary_mentions_the_headline_numbers(self):
        for outcome in ("WIN", "WIN", "LOSS", "WIN"):
            self.stats.record("EURUSD_otc", "CALL", outcome, at=AT)
        text = self.stats.summary_text()
        self.assertIn("4  (3W / 1L)", text)
        self.assertIn("75%", text)

    def test_best_and_worst_assets_need_a_minimum_sample(self):
        for _ in range(3):
            self.stats.record("GOOD_otc", "CALL", "WIN", at=AT)
        for _ in range(3):
            self.stats.record("BAD_otc", "CALL", "LOSS", at=AT)
        self.stats.record("NEW_otc", "CALL", "WIN", at=AT)   # only one trade

        best = [a for a, _ in self.stats.best_assets(min_trades=2)]
        worst = [a for a, _ in self.stats.worst_assets(min_trades=2)]
        self.assertEqual(best[0], "GOOD_otc")
        self.assertEqual(worst[0], "BAD_otc")
        self.assertNotIn("NEW_otc", best, "one trade is not evidence")
        self.assertNotIn("NEW_otc", worst)

    def test_tables_list_every_tracked_asset_and_hour(self):
        self.stats.record("EURUSD_otc", "CALL", "WIN", at=AT)
        self.stats.record("EURUSD_otc", "CALL", "LOSS", at=AT)
        self.assertIn("EURUSD_otc: 1W/1L 50% (2)", self.stats.asset_table())
        self.assertIn("01:00", self.stats.hourly_table())

    def test_reports_are_safe_for_telegram_html(self):
        """Messages go out with parse_mode=HTML: a raw <, > or & breaks the send."""
        for i in range(6):
            self.stats.record(f"EURUSD_otc", "CALL", "WIN" if i % 2 else "LOSS", at=AT + i)
        for text in (self.stats.summary_text(), self.stats.hourly_table(),
                     self.stats.asset_table()):
            for char in ("<", ">", "&"):
                self.assertNotIn(char, text)

    def test_format_local_uses_the_market_timezone(self):
        self.assertEqual(StatsTracker.format_local(AT, tz_offset_hours=3.0), "01:13:20")
        self.assertEqual(StatsTracker.format_local(AT, tz_offset_hours=0.0), "22:13:20")


if __name__ == "__main__":
    unittest.main()
