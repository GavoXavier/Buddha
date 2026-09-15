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

    def test_legacy_winrate_file_is_imported(self):
        self.legacy.write_text(json.dumps({"wins": 7, "losses": 3}), encoding="utf-8")

        imported = self.new_tracker()
        self.assertEqual((imported.wins, imported.losses), (7, 3))
        self.assertEqual(imported.total, 10)

    def test_legacy_import_does_not_override_real_stats(self):
        self.stats.record("A", "CALL", "WIN", at=AT)
        self.stats.save(force=True)
        self.legacy.write_text(json.dumps({"wins": 99, "losses": 99}), encoding="utf-8")
        self.assertEqual(self.new_tracker().total, 1)

    def test_missing_legacy_file_is_fine(self):
        self.assertFalse(self.legacy.exists())
        self.assertEqual(self.new_tracker().total, 0)


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
