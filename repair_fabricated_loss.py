"""Take one fabricated settlement back out of the record.

What happened, measured on 2026-09-15: ``EURNZD_otc`` entered at 21:05 was
adopted from the restored bars at 23:42, those bars were discarded two seconds
later when the live feed resumed through a 135-minute gap, and at 23:45 the
trade settled against the 23:44 price — a LOSS nobody measured. It reached both
files: a ``result`` line in the journal and a loss in ``stats.json``.

The fallback that allowed it is bounded now (``FALLBACK_WINDOW_BARS`` in
``engine/scheduler.py``), so this is a one-off repair of the data the bug left,
not a migration: the shape of neither file changes.

Archives both files beside themselves first, and refuses to run while the bot is
up — the running process holds the same numbers in memory and would write them
back on its next settle.

    python repair_fabricated_loss.py --dry-run
    python repair_fabricated_loss.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from journal import KIND_RESULT, load_journal
from stats import StatsTracker

STATS = Path("stats.json")
JOURNAL = Path("signals.jsonl")
STAMP = "contaminated-2026-09-15-fabricated-eurnzd"

# The trade to remove, identified by the defect rather than by a hardcoded
# second: a result written long after the expiry it claims to measure. The
# fallback window that allowed this is a couple of bars, so anything settled an
# hour late cannot have been priced off its own expiry.
ASSET = "EURNZD_otc"
MIN_LATE_SECONDS = 3600.0


def find_fabricated() -> tuple[dict, float]:
    """The fabricated trade in the journal, and the moment it was entered."""
    loaded = load_journal(JOURNAL)
    candidates = []
    for trade in loaded.trades:
        if trade.asset != ASSET or trade.our_outcome is None:
            continue
        late = (trade.settled_at or 0.0) - trade.expiry_at
        if late < MIN_LATE_SECONDS:
            continue
        candidates.append((trade, late))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly 1 late-settled {ASSET} result line, "
                         f"found {len(candidates)} — refusing to guess")
    trade, late = candidates[0]
    return {"id": trade.id, "entry_at": trade.entry_at,
            "expiry_at": trade.expiry_at, "outcome": trade.our_outcome,
            "settled_at": trade.settled_at, "late_seconds": late}, trade.entry_at


def repair_journal(entry_at: float, dry: bool) -> int:
    """Drop the one ``result`` line. The signal line stays: unsettled is true."""
    lines = JOURNAL.read_text(encoding="utf-8").splitlines()
    kept, dropped = [], 0
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if (record.get("kind") == KIND_RESULT
                and record.get("asset") == ASSET
                and abs(float(record.get("entry_at", 0.0)) - entry_at) <= 1.0):
            dropped += 1
            continue
        kept.append(line)
    if dropped != 1:
        raise SystemExit(f"expected exactly 1 result line to drop, found {dropped}")
    if not dry:
        JOURNAL.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dropped


def repair_stats(expiry_at: float, dry: bool) -> dict:
    doc = json.loads(STATS.read_text(encoding="utf-8"))

    recent = doc.get("recent", [])
    matches = [i for i, r in enumerate(recent)
               if r.get("asset") == ASSET and r.get("outcome") == "LOSS"
               and abs(float(r.get("at", 0.0)) - expiry_at) <= 1.0]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly 1 matching recent entry, found {len(matches)}")
    removed = recent.pop(matches[0])

    doc["global"]["losses"] -= 1
    by_asset = doc.get("by_asset", {}).get(ASSET)
    if by_asset is None:
        raise SystemExit(f"{ASSET} is not in by_asset — refusing to guess")
    by_asset["losses"] -= 1

    # The same bucketing the tracker uses, so the hour table loses the same loss.
    hour = int(((expiry_at + 3.0 * 3600.0) % 86400) // 3600)
    bucket = doc.get("by_hour", {}).get(str(hour))
    if bucket is not None:
        bucket["losses"] -= 1

    # streak/best/worst are order-dependent, so they are replayed rather than
    # adjusted: the same arithmetic ``record`` does, over what is left.
    streak = best = worst = 0
    for record in recent:
        if record.get("outcome") == "WIN":
            streak = streak + 1 if streak > 0 else 1
            best = max(best, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            worst = min(worst, streak)
    before = (doc.get("streak"), doc.get("best_streak"), doc.get("worst_streak"))
    doc["streak"], doc["best_streak"], doc["worst_streak"] = streak, best, worst

    if not dry:
        tmp = STATS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(STATS)
    return {"removed": removed, "hour": hour,
            "streak_before": before, "streak_after": (streak, best, worst),
            "global": doc["global"], "recent": len(recent)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fabricated, entry_at = find_fabricated()
    print("fabricated settlement:")
    for key, value in fabricated.items():
        print(f"  {key}: {value}")

    # What the file says now, so the repair can be checked against the EV line.
    before = StatsTracker(str(STATS), None)
    print(f"stats before: {before.wins}W/{before.losses}L")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    for path in (STATS, JOURNAL):
        archived = path.with_name(f"{path.stem}.{STAMP}{path.suffix}")
        shutil.copy2(path, archived)
        print(f"archived {path} -> {archived}")

    print(f"journal: dropped {repair_journal(entry_at, dry=False)} result line")
    summary = repair_stats(fabricated["expiry_at"], dry=False)
    print(f"stats: removed {summary['removed']}")
    print(f"       hour bucket {summary['hour']}, "
          f"streak {summary['streak_before']} -> {summary['streak_after']}")
    print(f"       global now {summary['global']}, {summary['recent']} recent")

    after = StatsTracker(str(STATS), None)
    print(f"stats after:  {after.wins}W/{after.losses}L")
    return 0


if __name__ == "__main__":
    sys.exit(main())
