"""Sweep the selectivity dials over the store, on bars held back from the sweep.

    python calibrate.py                  # the held-out read, once there is a sample
    python calibrate.py --min-trades 0   # sweep anyway, and read it as what it is
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
refuses to sweep a sample too small to tell anything from anything. The last of
these is a feature: a tool that produced a confident-looking table out of five
hours of bars would be worse than no tool, because the table would be read. The
gate is a *trade count on the held-out window*, not an hours figure, because the
store is a rolling window — see ``describe_pending``, which says plainly whether
more uptime would even help.

**What the numbers are not.** The replay judges engine behaviour, not fills. The
store holds only bars this bot aggregated from ticks it received, which is the
honest input — but it is one broker's OTC book over one period, no spread, no
slippage, and no payout. A setting that wins here has been shown not to be absurd,
and nothing more.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest import Replay, Store, _clock
from config import Config, ConfigError, load_config
from reconcile import break_even_win_rate, trades_needed, wilson

# How many settled trades the held-out window must produce before a column is
# allowed to be printed at all. Not a wall-clock threshold, and deliberately not an
# hours threshold either: the store is a rolling window (``Store.ceiling_hours``),
# so extra uptime past the cap adds no sample, and how many trades an hour of store
# yields is a property of the signal rate rather than of the clock. The number is
# the lenient end of what ``reconcile.trades_needed`` asks for — that function wants
# about 37 trades at 85% payout before a 70% rate could be told from break-even —
# so a column that clears this gate has earned a reading, not a verdict.
DEFAULT_MIN_TRADES = 30

# The rate a sweep hopes for, used only to say how large a sample would be needed
# to prove anything. Optimistic on purpose: if even this cannot be reached, the
# refusal is about the rate and not about the target.
_HOPED_RATE = 0.70

# How long a window must be, in multiples of the trend veto's own warm-up, before a
# low signal rate is read as the engine's rate rather than as the warm-up still
# finishing. Below this the tool says the rate is a floor, because on a window that
# short it is one — the veto needs 56 bars of one unbroken run before it can judge
# anything at all, so the first judgeable bars are a small part of such a window.
_WARM_UP_DOMINANCE = 3.0

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


def describe_pending(store: Store, held_out: int, held_hours: float,
                     min_trades: int, break_even: float, split: float) -> str:
    """Why it declined to sweep — and whether waiting would change the answer.

    The distinction this prints is the one that is easy to get wrong: a store with
    too few trades in it is not necessarily a store that needs more *time*. Past
    ``max_bars`` the store slides rather than grows, so the sample it can hold at a
    given signal rate is capped, and when the cap is below the gate no amount of
    uptime reaches it. Saying "wait a few more days" there would be advice that
    cannot come true.

    The rate is measured on the held-out window and the ceiling is applied to that
    same window — the last ``1 - split`` of the store — so every figure here is the
    quantity the gate was actually computed from.
    """
    span = store.span_hours()
    ceiling = store.ceiling_hours()
    held_ceiling = (1.0 - split) * ceiling
    rate_per_hour = held_out / held_hours if held_hours > 0 else 0.0
    warm_bars = (store.cfg.signal.trend_ema_len + store.cfg.signal.trend_slope_bars
                 + 1)
    warm_hours = warm_bars * store.cfg.candle_period / 3600.0
    lines = [
        f"Held out, the shipped dials settled {held_out} trade(s) in the "
        f"{held_hours:.1f}h window; a column is printed from {min_trades}.",
        "",
        "Not caution for its own sake. Every row of a sweep is a win/loss count over",
        f"a handful of trades, and the trend veto alone takes {warm_bars} bars "
        f"({warm_hours:.1f}h) of one unbroken run to exist at all — on a short "
        f"window the columns would be measuring warm-up, not dials.",
    ]

    needed = trades_needed(_HOPED_RATE, break_even)
    if needed:
        lines.append("")
        lines.append(
            f"The gate is the lenient end of the statistics: at {break_even:.1%} "
            f"break-even a {_HOPED_RATE:.0%} rate would need about {needed} settled "
            f"trades to clear its lower bound."
        )

    # What the held-out window can hold at this rate once the store has rolled
    # over. This is the ceiling on the sample, and it does not depend on how long
    # the bot has been up.
    reachable = rate_per_hour * held_ceiling
    lines.append("")
    lines.append(
        f"The store is a rolling window, not an archive: it keeps the newest "
        f"{store.cfg.max_bars} bars per market and drops the oldest, so at "
        f"{store.cfg.candle_period}s a bar it spans at most {ceiling:.1f}h "
        f"({held_ceiling:.1f}h of that held out). Past that, uptime slides the "
        f"window instead of growing it."
    )
    lines.append(
        f"At the {rate_per_hour:.2f} signals/hour measured here, a full held-out "
        f"window would hold about {reachable:.0f} settled trade(s) — under the gate."
        if reachable <= min_trades else
        f"At the {rate_per_hour:.2f} signals/hour measured here, a full held-out "
        f"window would hold about {reachable:.0f} settled trade(s), which clears "
        f"the gate."
    )

    if reachable <= min_trades and rate_per_hour > 0:
        # The cap, not the clock, is what is too small — and unlike "wait longer"
        # that is something the reader can act on, so name the size it would take.
        # MAX_BARS is the one dial here that changes no strategy: it decides how
        # much history the sweep may look at, not what the engine does with a bar.
        #
        # Zero signals/hour is deliberately excluded: the arithmetic is
        # undefined, and a rate measured as zero is a statement about the window
        # rather than about the dials.
        store_hours = min_trades / rate_per_hour / (1.0 - split)
        bars = math.ceil(store_hours * 3600.0 / store.cfg.candle_period)
        lines.append("")
        lines.append(
            f"So no amount of uptime prints this table at this rate — the cap is "
            f"what is short, not the clock. {min_trades} held-out trades needs "
            f"about {store_hours:.0f}h of store, which is MAX_BARS={bars} at "
            f"{store.cfg.candle_period}s bars; it is {store.cfg.max_bars} today. "
            f"Until it is raised the store is not merely failing to grow, it is "
            f"discarding every bar older than {ceiling:.1f}h — so raising it is "
            f"worth doing before that history is gone, not after. It is not a "
            f"strategy change (it decides how much history the sweep may read, "
            f"not what the engine does with a bar) but it wants its own restart."
        )

    if reachable > min_trades:
        # The rate is enough given time; the only question is how much time.
        hours = min_trades / rate_per_hour
        lines.append("")
        lines.append(
            f"At this rate {min_trades} held-out trades is about {hours:.1f}h of "
            f"window, so it is reachable — roughly "
            f"{max(0.0, hours - held_hours):.0f}h more of uptime from here."
        )
    elif span >= _WARM_UP_DOMINANCE * warm_hours:
        # The store is long enough that warm-up cannot explain a rate this low, so
        # the rate is what it is and waiting does not change the answer.
        lines.append("")
        lines.append(
            f"The store is past warm-up ({warm_hours:.1f}h of its {span:.1f}h), so "
            f"that rate is not an artefact of it: waiting cannot close the gap, "
            f"because the sample is capped by the rate and the rate is the thing "
            f"the sweep exists to change."
        )
    else:
        # Short enough that the measured rate is a floor rather than a forecast —
        # which is the honest thing to say, because it is also the case where
        # waiting *would* help.
        lines.append("")
        lines.append(
            f"That rate is measured over a store that is still mostly warm-up "
            f"({warm_hours:.1f}h of {span:.1f}h), so it is a floor and not a "
            f"forecast — it can rise as the window fills. If it does not, no amount "
            f"of uptime prints this table."
        )
    lines.append("")
    lines.append("Pass --min-trades 0 to sweep the sample that exists and read the "
                 "columns as what they are. Nothing has been swept here, so there is "
                 "no table to misread.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
# A module constant rather than a local, so "the table was not printed" is
# something a test can assert on rather than infer from a word that also occurs
# in the prose above it.
TABLE_HEADER = (f"{'dials':<24}{'train':>6}{'rate':>7}{'95% CI':>14}"
                f"{'|':>3}{'held out':>10}{'rate':>7}{'95% CI':>14}  verdict")


def print_table(rows: list[tuple[str, dict, dict]], break_even: float) -> None:
    print(TABLE_HEADER)
    print("-" * len(TABLE_HEADER))
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
          payout: float, min_trades: int) -> int:
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

    cut = split_at(full, split)
    train = Store(directory, cfg, hours=hours, window=(0.0, cut))
    test = Store(directory, cfg, hours=hours, window=(cut, float("inf")))
    train_hours, test_hours = train.span_hours(), test.span_hours()

    print(f"Split at {_clock(cut)}: train {train_hours:.1f}h, "
          f"held out {test_hours:.1f}h")
    print(f"Assumed payout {payout:.0f}% → break-even {break_even:.1%} "
          f"(the store records no payouts, so this is an assumption, not a reading)")

    # The gate is measured on the held-out window with the shipped dials, because
    # that is the sample the verdict column would be read from. The refusal comes
    # before any table, so there is no table to misread.
    baseline = outcome_of_window(Replay(test).run())
    if baseline["signals"] < min_trades:
        print()
        print(describe_pending(full, baseline["signals"], test_hours, min_trades,
                               break_even, split))
        return 1

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
    cold = baseline["cold_gate"]
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
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES,
                        help=f"refuse to sweep unless the held-out window settles "
                             f"at least this many trades (default "
                             f"{DEFAULT_MIN_TRADES})")
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
                 args.min_trades)


if __name__ == "__main__":
    raise SystemExit(main())
