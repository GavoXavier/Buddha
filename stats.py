"""Win/loss tracking — overall, per asset, and per hour of day.

A single global win rate hides the two things worth knowing: *which* markets
are actually paying, and *when* the bot is reliable. Both are tracked here and
persisted, so the picture survives restarts and builds over weeks.

Timestamps are stored as epoch seconds and bucketed by hour in a fixed market
timezone (the same one the Telegram messages are rendered in), so the hourly
table means the same thing regardless of where the machine runs.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("pocket.stats")

_SCHEMA_VERSION = 1
_RECENT_LIMIT = 50


@dataclass
class Tally:
    wins: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    def record(self, outcome: str) -> None:
        if outcome == "WIN":
            self.wins += 1
        else:
            self.losses += 1

    def as_dict(self) -> dict:
        return {"wins": self.wins, "losses": self.losses}

    @classmethod
    def from_dict(cls, data: dict) -> "Tally":
        return cls(wins=int(data.get("wins", 0)), losses=int(data.get("losses", 0)))


class StatsTracker:
    """Cumulative results, persisted to JSON."""

    def __init__(self, path: str = "stats.json",
                 legacy_path: Optional[str] = "winrate.json",
                 tz_offset_hours: float = 3.0) -> None:
        self.path = Path(path)
        self.legacy_path = Path(legacy_path) if legacy_path else None
        self.tz_offset_hours = tz_offset_hours

        self.overall = Tally()
        self.by_asset: dict[str, Tally] = {}
        self.by_hour: dict[int, Tally] = {}
        self.streak = 0            # >0 consecutive wins, <0 consecutive losses
        self.best_streak = 0
        self.worst_streak = 0
        self.recent: list[dict] = []
        self._last_save = 0.0
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            self._import_legacy()
            return
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable stats file %s: %s", self.path, exc)
            return

        if doc.get("version") != _SCHEMA_VERSION:
            log.warning("stats file version %s unsupported - starting fresh",
                        doc.get("version"))
            return

        self.overall = Tally.from_dict(doc.get("global", {}))
        self.by_asset = {k: Tally.from_dict(v) for k, v in doc.get("by_asset", {}).items()}
        self.by_hour = {int(k): Tally.from_dict(v) for k, v in doc.get("by_hour", {}).items()}
        self.streak = int(doc.get("streak", 0))
        self.best_streak = int(doc.get("best_streak", 0))
        self.worst_streak = int(doc.get("worst_streak", 0))
        self.recent = list(doc.get("recent", []))[-_RECENT_LIMIT:]

    def _import_legacy(self) -> None:
        """Adopt the old winrate.json totals so the record isn't lost on upgrade."""
        if self.legacy_path is None:
            return
        try:
            with open(self.legacy_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        wins, losses = int(data.get("wins", 0)), int(data.get("losses", 0))
        if wins or losses:
            self.overall = Tally(wins=wins, losses=losses)
            self.best_streak = wins
            log.info("imported %dW/%dL from %s", wins, losses, self.legacy_path)

    def save(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_save < 5.0:
            return
        self._last_save = now
        doc = {
            "version": _SCHEMA_VERSION,
            "updated": now,
            "global": self.overall.as_dict(),
            "by_asset": {k: v.as_dict() for k, v in self.by_asset.items()},
            "by_hour": {str(k): v.as_dict() for k, v in self.by_hour.items()},
            "streak": self.streak,
            "best_streak": self.best_streak,
            "worst_streak": self.worst_streak,
            "recent": self.recent[-_RECENT_LIMIT:],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not persist stats: %s", exc)

    # -- recording -----------------------------------------------------------
    def record(self, asset: str, direction: str, outcome: str,
               at: Optional[float] = None) -> None:
        at = time.time() if at is None else at
        self.overall.record(outcome)
        self.by_asset.setdefault(asset, Tally()).record(outcome)
        hour = self._hour_of(at)
        self.by_hour.setdefault(hour, Tally()).record(outcome)

        if outcome == "WIN":
            self.streak = self.streak + 1 if self.streak > 0 else 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = self.streak - 1 if self.streak < 0 else -1
            self.worst_streak = min(self.worst_streak, self.streak)

        self.recent.append({
            "asset": asset, "direction": direction,
            "outcome": outcome, "at": at,
        })
        self.recent = self.recent[-_RECENT_LIMIT:]
        self.save()

    def _hour_of(self, at: float) -> int:
        shifted = at + self.tz_offset_hours * 3600.0
        return int((shifted % 86400) // 3600)

    # -- reads ---------------------------------------------------------------
    @property
    def wins(self) -> int:
        return self.overall.wins

    @property
    def losses(self) -> int:
        return self.overall.losses

    @property
    def total(self) -> int:
        return self.overall.total

    @property
    def win_rate(self) -> float:
        return self.overall.win_rate

    def asset_tally(self, asset: str) -> Tally:
        return self.by_asset.get(asset, Tally())

    def hour_tally(self, hour: int) -> Tally:
        return self.by_hour.get(hour, Tally())

    def best_assets(self, min_trades: int = 3, limit: int = 3) -> list[tuple[str, Tally]]:
        rows = [(a, t) for a, t in self.by_asset.items() if t.total >= min_trades]
        rows.sort(key=lambda r: (-r[1].win_rate, -r[1].total))
        return rows[:limit]

    def worst_assets(self, min_trades: int = 3, limit: int = 3) -> list[tuple[str, Tally]]:
        rows = [(a, t) for a, t in self.by_asset.items() if t.total >= min_trades]
        rows.sort(key=lambda r: (r[1].win_rate, -r[1].total))
        return rows[:limit]

    def best_hours(self, min_trades: int = 3, limit: int = 3) -> list[tuple[int, Tally]]:
        rows = [(h, t) for h, t in self.by_hour.items() if t.total >= min_trades]
        rows.sort(key=lambda r: (-r[1].win_rate, -r[1].total))
        return rows[:limit]

    def consecutive_losses(self) -> int:
        return -self.streak if self.streak < 0 else 0

    def iter_assets(self) -> Iterator[tuple[str, Tally]]:
        return iter(sorted(self.by_asset.items()))

    # -- reporting -----------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """Compact human-readable report (Telegram-safe: no raw <, > or &)."""
        if self.total == 0:
            return ["No completed trades yet."]
        lines = [
            f"Trades: {self.total}  ({self.wins}W / {self.losses}L)",
            f"Win rate: {self.win_rate:.0%}",
        ]
        if self.streak > 0:
            lines.append(f"Streak: {self.streak} wins")
        elif self.streak < 0:
            lines.append(f"Streak: {self.consecutive_losses()} losses")
        lines.append(f"Best streak: {self.best_streak}W  Worst: {abs(self.worst_streak)}L")

        best = self.best_assets(min_trades=2)
        if best:
            lines.append("")
            lines.append("Best markets:")
            for asset, tally in best:
                lines.append(f"  {asset}: {tally.win_rate:.0%} ({tally.wins}/{tally.total})")

        hours = self.best_hours(min_trades=2)
        if hours:
            lines.append("")
            lines.append("Best hours (local):")
            for hour, tally in hours:
                lines.append(f"  {hour:02d}:00 - {tally.win_rate:.0%} ({tally.wins}/{tally.total})")
        return lines

    def summary_text(self) -> str:
        return "\n".join(self.summary_lines())

    def hourly_table(self) -> str:
        """Every tracked hour with its record, for a full report."""
        if not self.by_hour:
            return "No hourly data yet."
        rows = []
        for hour in sorted(self.by_hour):
            tally = self.by_hour[hour]
            if tally.total == 0:
                continue
            rows.append(f"{hour:02d}:00  {tally.wins}W/{tally.losses}L  "
                        f"{tally.win_rate:.0%}  ({tally.total})")
        return "\n".join(rows) if rows else "No hourly data yet."

    def asset_table(self) -> str:
        if not self.by_asset:
            return "No per-market data yet."
        rows = []
        for asset, tally in sorted(self.by_asset.items(),
                                   key=lambda r: (-r[1].win_rate, -r[1].total)):
            if tally.total == 0:
                continue
            rows.append(f"{asset}: {tally.wins}W/{tally.losses}L "
                        f"{tally.win_rate:.0%} ({tally.total})")
        return "\n".join(rows) if rows else "No per-market data yet."

    @staticmethod
    def format_local(at: float, tz_offset_hours: float = 3.0) -> str:
        tz = _dt.timezone(_dt.timedelta(hours=tz_offset_hours))
        return _dt.datetime.fromtimestamp(at, tz=tz).strftime("%H:%M:%S")
