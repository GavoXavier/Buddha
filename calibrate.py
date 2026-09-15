"""Sweep the selectivity dials over the store, on bars held back from the sweep.

    python calibrate.py                  # the held-out read, once the store is long enough
    python calibrate.py --min-hours 0    # sweep anyway, for a short store
    python calibrate.py --split 0.5      # half train, half held out
    python calibrate.py --payout 92      # the break-even to compare against

**Why this exists.** ``.env`` has six selectivity dials, and the README can say
what each one *means* but not what each one *buys*. Choosing them by eye against a
single replay is the classic way to fit a strategy to the noise in one week of
data: the best-looking row of a sweep is the row that got luckiest, and nothing in
the row says so.

**Why the split.** Every market's history is cut by *time* into a training window
and a window held back from the sweep. The training column is what a search would
have picked; the held-out column is what happened next. Reading both side by side
is the cheapest available defence against overfitting, and on a store this short
it usually shows the two disagreeing — which is the lesson, not a failure.

**Why a flat payout.** The candle store records prices, not payouts, so the replay
cannot compute a return per trade the way the journal does. A single assumed
payout converts accuracy into a break-even accuracy, and the assumption is printed
with the table rather than buried: a 54% rate is a losing record at 85% and a
winning one at 92%.

**What it will not do.** It does not choose a setting, does not say "best", and
refuses to sweep a store too short to tell anything from anything. The last of
these is a feature: a tool that produced a confident-looking table out of five
hours of bars would be worse than no tool, because the table would be read.

**What the numbers are not.** The replay judges engine behaviour, not fills. The
store holds only bars this bot aggregated from ticks it received, which is the
honest input — but it is one broker's OTC book over one period, no spread, no
slippage, and no payout. A setting that wins here has been shown not to be absurd,
and nothing more.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest import Replay, Store, _clock
from config import Config, ConfigError, load_config
from reconcile import break_even_win_rate, trades_needed, wilson

# How many hours of store a sweep needs before its columns mean anything. Two days
# is not a statistical threshold — it is the point at which the *warm-up* stops
# dominating the sample: the trend veto alone takes 4h40m of one unbroken run.
DEFAULT_MIN_HOURS = 48.0

# One dial varied at a time from the shipped settings, never a cross product. A
# full grid over six dials is unreadable and its best row is a coincidence; the
# question worth asking first is what each dial does on its own, and what it costs.
_DIALS: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("MIN_SCORE", (1, 3)),
    ("MIN_COMPONENTS", (1, 3)),
    ("TREND_FLAT_MIN_SCORE", (0, 5)),
    ("MIN_CONFIDENCE", (0.2, 0.4)),
    ("TREND_EMA_LEN", (20, 30)),
    ("USE_TREND", (False,)),
)


def variations(base):
    """The baseline plus one dial moved at a time: [(label, SignalConfig)].

    The baseline is first so it is the row every other row is read against — the
    shipped settings are the ones that have to be beaten, not the ones assumed.
    """
    rows = [("shipped", base)]
    for name, values in _DIALS:
        field = name.lower()
        for value in values:
            if getattr(base, field) == value:
                continue
            rows.append((f"{name}={value}", replace(base, **{field: value})))
    return rows


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------
def split_at(store: Store, fraction: float) -> float:
    """The wall-clock instant that divides the store into train and held-out.

    Chosen on the *whole store's* span rather than per market, so every market is
    trained on the same period and tested on the same later one. Per-market cuts
    would let a market with a short history train on bars that are, in wall-clock
    terms, the held-out period of another — which is not a held-out period at all.
    """
    marks = [t for bars in store.bars.values() for t in (bars[0].time, bars[-1].time)]
    if not marks:
        return 0.0
    first, last = min(marks), max(marks)
    return first + (last - first) * fraction


def outcome_of_window(replay: Replay) -> dict:
    """What one replay of one window produced, in the terms the table prints."""
    decided = len(replay.trades)
    wins = replay.wins
    low, high = wilson(wins, decided)
    return {
        "signals": decided,
        "wins": wins,
        "rate": wins / decided if decided else 0.0,
        "low": low,
        "high": high,
        "unsettled": replay.unsettled,
        "cold_gate": replay.cold_gate,
    }


def verdict_of(row: dict, break_even: float) -> str:
    """One word per window, and never a word the sample cannot support.

    The rule is ``reconcile.py``'s: the *lower* bound has to clear break-even
    before the word "beats" is allowed. A rate above break-even whose interval
    spans it is called what it is — indistinguishable.
    """
    if row["signals"] < 2:
        return "too few"
    if row["low"] > break_even:
        return "beats"
    if row["high"] < break_even:
        return "under"
    return "no edge"


def describe_pending(hours: float, min_hours: float, rate_per_hour: float,
                     break_even: float) -> str:
    """Why it declined to sweep, and how much longer to wait."""
    short = min_hours - hours
    lines = [
        f"The store holds {hours:.1f}h of bars; a sweep needs at least "
        f"{min_hours:.0f}h.",
        "",
        "This is not caution for its own sake. Every row of a sweep is a win/loss",
        "count over a handful of trades, and the trend veto alone takes 4h40m of",
        "one unbroken run to exist at all — so on a store this short the columns",
        "would be measuring warm-up, not dials.",
    ]
    if rate_per_hour > 0:
        # How long until the sample could settle a question at all. The rate is
        # the engine's own, measured now; the target is the smallest edge the
        # break-even leaves open to a 20-point-above rate, which is the optimistic
        # end of what anyone would hope for.
        needed = trades_needed(0.70, break_even)
        if needed:
            lines.append("")
            lines.append(
                f"At the engine's current {rate_per_hour:.2f} signals/hour and "
                f"{break_even:.1%} break-even, a 70% rate would need about "
                f"{needed} settled trades to clear the lower bound — roughly "
                f"{needed / rate_per_hour / 24:.0f} more days of uptime."
            )
    lines.append("")
    lines.append(f"Wait about {short:.0f}h more, or pass --min-hours 0 to sweep the "
                 f"short store anyway and read the columns as what they are.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def print_table(rows: list[tuple[str, dict, dict]], break_even: float) -> None:
    header = (f"{'dials':<24}{'train':>6}{'rate':>7}{'95% CI':>14}"
              f"{'|':>3}{'held out':>10}{'rate':>7}{'95% CI':>14}  verdict")
    print(header)
    print("-" * len(header))
    for label, train, test in rows:
        print(f"{label:<24}{train['signals']:>6}{train['rate']:>7.0%}"
              f"{_ci(train):>14}{'|':>3}{test['signals']:>10}{test['rate']:>7.0%}"
              f"{_ci(test):>14}  {verdict_of(test, break_even)}")


def _ci(row: dict) -> str:
    if row["signals"] < 2:
        return "—"
    return f"{row['low']:.0%}..{row['high']:.0%}"


def reading(rows: list[tuple[str, dict, dict]], break_even: float) -> str:
    """What, if anything, the table is entitled to say."""
    held = [test for _label, _train, test in rows]
    biggest = max((row["signals"] for row in held), default=0)
    cleared = [label for label, _train, test in rows
               if verdict_of(test, break_even) == "beats"]
    if cleared:
        return ("Held out, these beat break-even on the lower bound: "
                + ", ".join(cleared) + ". That is not a verdict — re-run after more "
                "uptime before changing a dial on it, because a sweep read once is "
                "a sweep read wrong.")
    return (f"Nothing here is distinguishable from {break_even:.1%} break-even. The "
            f"largest held-out sample is {biggest} trade(s), and every interval "
            f"covers the break-even. Read the train column as what a search would "
            f"have picked and the held-out column as what happened next — where they "
            f"disagree, the train column was noise.")


# ---------------------------------------------------------------------------
def sweep(cfg: Config, directory: str, hours: float, split: float,
          payout: float, min_hours: float) -> int:
    full = Store(directory, cfg, hours=hours)
    span = full.span_hours()
    break_even = break_even_win_rate(payout)

    print(f"Candle store: {directory}")
    print(f"Period {cfg.candle_period}s · expiry {cfg.expiry} · "
          f"cooldown {cfg.cooldown_seconds}s · lead {cfg.lead_seconds}s")
    if not full.bars:
        print("\nNo bars in the store for this feed and bar length — nothing to sweep.")
        return 1

    bars = sum(len(b) for b in full.bars.values())
    print(f"{bars} bars over {len(full.bars)} markets, {span:.1f}h")

    if span < min_hours:
        # The refusal comes before any table, so there is no table to misread.
        baseline = Replay(full).run()
        rate = len(baseline.trades) / span if span > 0 else 0.0
        print()
        print(describe_pending(span, min_hours, rate, break_even))
        return 1

    cut = split_at(full, split)
    train = Store(directory, cfg, hours=hours, window=(0.0, cut))
    test = Store(directory, cfg, hours=hours, window=(cut, float("inf")))
    train_hours, test_hours = train.span_hours(), test.span_hours()

    print(f"Split at {_clock(cut)}: train {train_hours:.1f}h, "
          f"held out {test_hours:.1f}h")
    print(f"Assumed payout {payout:.0f}% → break-even {break_even:.1%} "
          f"(the store records no payouts, so this is an assumption, not a reading)")
    print()

    rows: list[tuple[str, dict, dict]] = []
    for label, signal in variations(cfg.signal):
        rows.append((label,
                     outcome_of_window(Replay(train, signal).run()),
                     outcome_of_window(Replay(test, signal).run())))
    print_table(rows, break_even)
    print()
    print(reading(rows, break_even))

    # Warm-up is paid once per window by every row, so it does not bias the
    # comparison — but it does shrink both samples, and a reader should know by how
    # much of the held-out window nothing could be judged at all.
    cold = sum(test["cold_gate"] for _l, _t, test in rows[:1])
    if cold:
        print()
        print(f"The shipped row passed over {cold} setup(s) in the held-out window "
              f"because the trend veto was cold. Every window pays its own warm-up "
              f"({cfg.signal.trend_ema_len + cfg.signal.trend_slope_bars + 1} bars "
              f"of one unbroken run), so a short window is mostly warm-up.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the selectivity dials, on bars held back from the sweep.")
    parser.add_argument("--dir", default=None,
                        help="candle store directory (default: from .env)")
    parser.add_argument("--hours", type=float, default=0.0,
                        help="only the most recent N hours of the store (0 = all)")
    parser.add_argument("--split", type=float, default=0.6,
                        help="fraction of the span used for training (default 0.6)")
    parser.add_argument("--payout", type=float, default=85.0,
                        help="payout %% to compare against (default 85)")
    parser.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS,
                        help=f"refuse to sweep a store shorter than this "
                             f"(default {DEFAULT_MIN_HOURS:.0f})")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    if not 0.0 < args.split < 1.0:
        print("--split must be between 0 and 1", file=sys.stderr)
        return 2

    directory = args.dir or cfg.candle_store_dir
    if args.dir:
        cfg = replace(cfg, candle_store_dir=args.dir)
    return sweep(cfg, directory, args.hours, args.split, args.payout,
                 args.min_hours)


if __name__ == "__main__":
    raise SystemExit(main())
