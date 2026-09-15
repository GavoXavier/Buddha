"""Check the bot's own win/loss labels against the broker's settlements.

    python reconcile.py                     # read the deals in your account
    python reconcile.py --dump deals.json   # save what it saw, for offline work
    python reconcile.py --deals deals.json  # reconcile from a saved dump

**Why this exists.** Everything the bot prints as WIN or LOSS is *its own*
calculation: entry is the close of the bar ending at T, exit is the close of the
bar ending at T+60, both read from the tick stream. But the broker settles the
trade on *their* price path, and for OTC assets that is measurably not the same
series — the two diverged by 6-18 pips over five minutes when this project
measured them. So the label in your Telegram chat is an estimate of a settlement
the bot never sees.

This tool closes that gap without placing a single order. It reads the deals in
your Pocket Option account (the ones you placed by hand from the signals), pairs
each one with the signal that produced it, and reports:

  * whether the broker agreed with our WIN/LOSS;
  * how far the broker's fill was from the bar close the signal was based on
    (the slippage the signal never knew about);
  * the payouts actually paid, and therefore the win rate that just breaks even;
  * whether there is enough evidence to conclude anything at all.

**Nothing here trades.** It only reads. The order-placing path is a separate,
later step precisely so that this question gets answered first.

Reading the account needs ``POCKET_SSID``; the journal needs the bot to have been
running with ``JOURNAL_SIGNALS=1`` (the default). Signals the bot sent before the
journal existed cannot be reconciled — there is no record of them.
"""

from __future__ import annotations

import argparse
import asyncio
import decimal
import enum
import json
import math
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import Config, ConfigError, load_config
from telegram.sender import configure_clock
from journal import JournalledTrade, LoadedJournal, load_journal

# Matching tolerance: how far apart our entry second and the broker's open
# timestamp may be and still count as the same trade. Generous enough for clock
# skew and for a hand-placed order a few seconds late; tight enough that two
# different minutes cannot be confused.
DEFAULT_TOLERANCE = 20.0

# How far apart the broker's clock and ours are allowed to be when the offset is
# measured from the data, rather than assumed. Wide enough for any timezone on
# the planet; the search is over candidate pairs, not over seconds, so the width
# costs nothing.
CLOCK_SEARCH_SECONDS = 12 * 3600

WIN, LOSS, PUSH, UNKNOWN = "WIN", "LOSS", "PUSH", "UNKNOWN"


# ---------------------------------------------------------------------------
# broker-side deals
# ---------------------------------------------------------------------------
@dataclass
class BrokerDeal:
    """One settled deal from the account, in our own terms."""

    asset: str
    direction: Optional[str]
    entry_at: Optional[float]
    exit_at: Optional[float]
    open_price: Optional[float]
    close_price: Optional[float]
    profit: Optional[float]
    percent_profit: Optional[float]
    is_demo: Optional[bool]
    deal_id: str = ""
    raw: dict = dc_field(default_factory=dict)

    @property
    def outcome(self) -> str:
        """The broker's verdict. A refunded trade is neither a win nor a loss."""
        if self.profit is None:
            return UNKNOWN
        if self.profit > 0:
            return WIN
        if self.profit < 0:
            return LOSS
        return PUSH


def normalize_asset(value: Any) -> str:
    """``Asset.EURUSD_otc``, ``"EURUSD_otc"`` and ``"eurusd_otc"`` are one asset."""
    if value is None:
        return ""
    text = str(getattr(value, "value", value)).strip()
    if "." in text and text.split(".")[0].isidentifier():
        text = text.split(".", 1)[1]      # "Asset.EURUSD_otc" -> "EURUSD_otc"
    return text


def normalize_direction(value: Any) -> Optional[str]:
    """Anything the broker uses for a direction, reduced to CALL or PUT.

    The SDK uses two different enums for this, and only one is a string enum.
    ``DealAction`` is what you *send* (``"call"``/``"put"``); a settled ``Deal``
    carries ``Command``, an **integer** enum where ``CALL`` is 0 and ``PUT`` is 1.
    The enum's *name* is therefore the only field that means the same thing in
    both — reading ``.value`` turns ``Command.CALL`` into the string "0", and a
    lookup table that maps "0" onto a direction inverted every real deal this
    project read. A bare number is ambiguous, so it is refused rather than
    guessed at: an unknown direction is excluded from the comparison, whereas a
    wrong one is a silent lie.
    """
    if value is None:
        return None
    if isinstance(value, enum.Enum):
        text = value.name                      # Command.CALL -> "CALL"
    elif isinstance(value, str):
        text = value
    else:
        return None
    text = text.strip().lower()
    if text in ("call", "buy", "up", "higher", "c"):
        return "CALL"
    if text in ("put", "sell", "down", "lower", "p"):
        return "PUT"
    return None


def plain(value: Any) -> Any:
    """A JSON-safe value that keeps an enum's *name*.

    ``str(Command.CALL)`` is "0", so a dump written with ``str`` loses the
    direction it was made to check. Decimals become floats for the same reason:
    the dump is read by a human when parsing is in question.
    """
    if value is None:
        return value
    # Before the numeric check: ``Command`` is an IntEnum, so it *is* an int and
    # a primitive-first test would hand the enum straight back.
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def first_number(obj: Any, *names: str) -> Optional[float]:
    """The first present, numeric attribute among ``names``."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def epoch_seconds(obj: Any, *names: str) -> Optional[float]:
    """A timestamp in seconds, from whichever field is populated.

    The model carries several: ``open_timestamp``/``close_timestamp`` in seconds
    and ``open_ms``/``close_ms`` in milliseconds. Anything that looks like
    milliseconds is scaled down rather than being compared against a seconds
    clock and silently matching nothing.
    """
    value = first_number(obj, *names)
    if value is None:
        return None
    if value > 1e11:                 # milliseconds (1e11 s is year 5138)
        return value / 1000.0
    return value


def deal_from_model(deal: Any) -> Optional[BrokerDeal]:
    """Reduce an SDK ``Deal`` (or a dict from a dump) to a ``BrokerDeal``.

    Defensive on purpose: the SDK's model is the broker's, not ours, and this has
    to survive a field being renamed or arriving as a string. An unrecognisable
    deal returns None and is counted, never guessed at.
    """
    asset = normalize_asset(_get(deal, "asset"))
    if not asset:
        return None

    raw: dict = {}
    if isinstance(deal, dict):
        raw = deal
    else:
        for name in ("id", "asset", "command", "profit", "percent_profit",
                     "open_price", "close_price", "open_timestamp",
                     "close_timestamp", "is_demo", "amount", "currency"):
            value = getattr(deal, name, None)
            if value is not None:
                raw[name] = plain(value)

    return BrokerDeal(
        asset=asset,
        direction=normalize_direction(
            _get(deal, "command") if _get(deal, "command") is not None
            else _get(deal, "action")),
        entry_at=epoch_seconds(deal, "open_timestamp", "open_time", "open_ms"),
        exit_at=epoch_seconds(deal, "close_timestamp", "close_time", "close_ms"),
        open_price=first_number(deal, "open_price"),
        close_price=first_number(deal, "close_price"),
        profit=first_number(deal, "profit"),
        percent_profit=first_number(deal, "percent_profit"),
        is_demo=_truthy(_get(deal, "is_demo")),
        deal_id=str(_get(deal, "id") or ""),
        raw=raw,
    )


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _truthy(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
@dataclass
class Pair:
    """One of our signals and the broker deal that settles it."""

    signal: JournalledTrade
    deal: BrokerDeal
    offset: float                 # deal.entry_at - signal.entry_at, in seconds

    @property
    def agreed(self) -> Optional[bool]:
        """Did the broker's verdict match the label we printed?"""
        if not self.signal.settled or self.deal.outcome in (UNKNOWN, PUSH):
            return None
        return self.signal.our_outcome == self.deal.outcome

    @property
    def direction_agreed(self) -> Optional[bool]:
        if self.deal.direction is None:
            return None
        return self.deal.direction == self.signal.direction

    def entry_slippage_pips(self, pip: float) -> Optional[float]:
        """How much worse the broker filled us than the price we measured.

        Positive means adverse: for a CALL we were filled above our reference,
        for a PUT below it — in both cases the trade starts further from where
        the signal thought it was.
        """
        if self.deal.open_price is None or self.signal.our_entry is None or not pip:
            return None
        sign = 1.0 if self.signal.direction == "CALL" else -1.0
        return (self.deal.open_price - self.signal.our_entry) * sign / pip


@dataclass
class Reconciliation:
    journal: LoadedJournal
    deals: list[BrokerDeal]
    pairs: list[Pair] = dc_field(default_factory=list)
    unparsed_deals: int = 0
    unreadable_deals: int = 0
    # The broker-clock offset the matching used, how it was arrived at, and
    # whether the operator supplied it. Kept so the report can say which, rather
    # than presenting a measured correction as though it were a stated fact.
    clock_offset: float = 0.0
    offset_hits: int = 0
    offset_considered: int = 0
    offset_given: bool = False

    @property
    def unmatched_signals(self) -> list[JournalledTrade]:
        paired = {id(p.signal) for p in self.pairs}
        return [t for t in self.journal.trades if id(t) not in paired]

    @property
    def unmatched_deals(self) -> list[BrokerDeal]:
        paired = {id(p.deal) for p in self.pairs}
        return [d for d in self.deals if id(d) not in paired]

    @property
    def compared(self) -> list[Pair]:
        return [p for p in self.pairs if p.agreed is not None]

    def agreements(self) -> tuple[int, int]:
        compared = self.compared
        return sum(1 for p in compared if p.agreed), len(compared)

    def payouts(self) -> list[float]:
        """The payouts actually paid, as percentages of the stake."""
        out = []
        for pair in self.pairs:
            percent = pair.deal.percent_profit
            if percent is None:
                continue
            out.append(percent * 100 if 0 < percent <= 1 else percent)
        return out

    def broker_tally(self) -> dict[str, int]:
        counts = {WIN: 0, LOSS: 0, PUSH: 0, UNKNOWN: 0}
        for pair in self.pairs:
            counts[pair.deal.outcome] = counts.get(pair.deal.outcome, 0) + 1
        return counts


def detect_clock_offset(signals: Sequence[JournalledTrade],
                        deals: Sequence[BrokerDeal],
                        tolerance: float = DEFAULT_TOLERANCE,
                        search: float = CLOCK_SEARCH_SECONDS
                        ) -> tuple[Optional[float], int, int]:
    """The constant the broker's clock runs ahead of ours, measured, not assumed.

    The broker's ``open_timestamp`` is on a different clock from ours. Measured
    live over five consecutive demo deals: the broker's open sat 7197-7198
    seconds after the second we had signalled for — a constant offset of about
    two hours, not the second or two of placement latency ``DEFAULT_TOLERANCE``
    was built for, and not a round 7200 either, so the timezone rule behind it is
    not something this file can derive.

    It is therefore read off the data. For every same-asset (signal, deal) pair
    the difference between the two timestamps is a candidate; the true offset is
    the candidate with the most company within ``tolerance``, because a deal
    placed for a signal scatters a second or two around it, while the same deal
    measured against some other minute sits a whole bar away and never joins the
    cluster.

    Returns ``(offset, aligned, considered)``. ``offset`` is None when no cluster
    has more than one member: one coincidental pair is not a measurement, and
    applying it would move every other deal further from its signal.
    """
    timed = [d for d in deals if d.entry_at is not None]
    if not timed or not signals:
        return None, 0, len(timed)

    deltas: list[float] = []
    for deal in timed:
        for signal in signals:
            if signal.asset != deal.asset:
                continue
            delta = (deal.entry_at or 0.0) - signal.entry_at
            if abs(delta) <= search:
                deltas.append(delta)
    if not deltas:
        return None, 0, len(timed)

    cluster: list[float] = []
    for candidate in deltas:
        near = [d for d in deltas if abs(d - candidate) <= tolerance]
        if len(near) > len(cluster):
            cluster = near
    if len(cluster) < 2:
        return None, 0, len(timed)
    return median(cluster), len(cluster), len(timed)


def match(journal: LoadedJournal, deals: Sequence[BrokerDeal],
          tolerance: float = DEFAULT_TOLERANCE,
          clock_offset: float = 0.0) -> Reconciliation:
    """Pair each deal with the signal it settles, greedily and one-to-one.

    Both sides are sorted by time and walked together, so a deal may be claimed
    by at most one signal and vice versa. A deal with no signal is as informative
    as a signal with no deal, which is why neither is dropped.

    ``clock_offset`` is subtracted from every deal's timestamp before comparing,
    so that a broker clock running hours off ours does not put every deal outside
    the tolerance and report a silent zero. See ``detect_clock_offset``.
    """
    result = Reconciliation(journal=journal, deals=list(deals),
                            clock_offset=clock_offset)

    open_signals = sorted(journal.trades, key=lambda t: t.entry_at)
    open_deals = sorted(
        (d for d in deals if d.entry_at is not None), key=lambda d: d.entry_at or 0.0)
    claimed: set[int] = set()

    for signal in open_signals:
        best: Optional[tuple[float, BrokerDeal]] = None
        for deal in open_deals:
            if id(deal) in claimed or deal.asset != signal.asset:
                continue
            offset = (deal.entry_at or 0.0) - clock_offset - signal.entry_at
            if abs(offset) > tolerance:
                continue
            if best is None or abs(offset) < abs(best[0]):
                best = (offset, deal)
        if best is None:
            continue
        offset, deal = best
        claimed.add(id(deal))
        result.pairs.append(Pair(signal=signal, deal=deal, offset=offset))

    result.pairs.sort(key=lambda p: p.signal.entry_at)
    return result


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — sane at small n and near 0/1, unlike normal ±."""
    if total <= 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)


def break_even_win_rate(payout_percent: float) -> float:
    """The win rate at which a payout of ``payout_percent`` neither wins nor loses."""
    if payout_percent <= 0:
        return 1.0
    return 1.0 / (1.0 + payout_percent / 100.0)


def trades_needed(observed: float, threshold: float, z: float = 1.96,
                  limit: int = 500_000) -> Optional[int]:
    """Settled trades required before ``observed`` could be told from ``threshold``.

    The criterion is the one the report prints: the *lower* bound of the Wilson
    interval has to clear the threshold. A normal approximation on its own is not
    usable here — it collapses as the rate approaches 1 (``p(1-p) -> 0``), which
    is exactly the flattering small-sample case this function exists to catch: at
    a 100% observed rate it would claim a handful of trades proves an edge that a
    break-even of 54% leaves wide open.

    Returns None when the observed rate is at or below the threshold (no sample
    size turns a losing rate into a winning one), or when the required sample is
    beyond ``limit``.
    """
    if observed <= threshold:
        return None

    def clears(n: int) -> bool:
        return wilson(int(round(observed * n)), n, z)[0] > threshold

    high = 10
    while high <= limit and not clears(high):
        high *= 2
    if high > limit:
        return None

    low = high // 2                      # known not to clear
    while low + 1 < high:                # narrow to the smallest n that does
        mid = (low + high) // 2
        if clears(mid):
            high = mid
        else:
            low = mid
    return high


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def pip_size(asset: str, digits: Optional[int] = None) -> float:
    """One pip in price terms. Falls back to the usual convention by quote."""
    if digits is None:
        digits = 3 if asset.upper().startswith("USD") and "JPY" in asset.upper() else 5
        if "JPY" in asset.upper():
            digits = 3
    return 10.0 ** -(digits - 1)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def report(rec: Reconciliation, cfg: Config, tolerance: float,
           digits_by_asset: dict[str, int], journal_path: str = "") -> None:
    journal = rec.journal
    print(f"Signal journal: {journal_path or cfg.signal_journal_path}")
    print(f"  {len(journal)} signals, {len(journal.settled)} settled by the bot, "
          f"{len(journal.unsettled)} unsettled")
    if journal.bad_lines or journal.unknown_results:
        print(f"  {journal.bad_lines} unreadable line(s), "
              f"{journal.unknown_results} result(s) with no matching signal")
    if not len(journal):
        print()
        print("Nothing to reconcile. The journal only records signals sent while")
        print("JOURNAL_SIGNALS=1, so enable it and let the bot run first.")
        return

    print()
    print(f"Deals read: {len(rec.deals)}")
    if rec.unparsed_deals or rec.unreadable_deals:
        print(f"  {rec.unparsed_deals} with no usable asset, "
              f"{rec.unreadable_deals} unreadable")
    stamped = [d.entry_at for d in rec.deals if d.entry_at is not None]
    if stamped:
        print(f"  spanning {_clock(min(stamped))} to {_clock(max(stamped))}")
    demo = [d for d in rec.deals if d.is_demo]
    if demo:
        print(f"  {len(demo)} of them on the demo account")
    if rec.clock_offset:
        if rec.offset_given:
            how = "given with --offset"
        else:
            how = (f"measured — {rec.offset_hits} of {rec.offset_considered} "
                   f"timestamped deal(s) land on a signal at this offset")
        print(f"  broker clock is {rec.clock_offset:+.0f}s from ours ({how})")
        if abs(rec.clock_offset) > tolerance:
            print(f"    the matching tolerance is {tolerance:.0f}s, so every deal "
                  f"is aligned by this before it is paired")
    print()

    print(f"Matched: {len(rec.pairs)} signal(s) have a deal within {tolerance:.0f}s")
    print(f"  {len(rec.unmatched_signals)} signal(s) with no deal "
          f"(not placed, or placed outside the window)")
    print(f"  {len(rec.unmatched_deals)} deal(s) with no signal "
          f"(traded outside the bot — excluded below)")
    print()

    if not rec.pairs:
        print("No overlap between the two records, so there is nothing to compare.")
        if rec.deals and not rec.clock_offset:
            print()
            print("The two records may simply be on different clocks: no offset")
            print("could be measured from the data. Pass --offset SECONDS (the")
            print("broker's timestamp minus yours) to align them by hand.")
        print("Place trades by hand from the signals and run this again.")
        return

    agreed, compared = rec.agreements()
    if compared:
        low, high = wilson(agreed, compared)
        print(f"Label agreement: {agreed}/{compared} = {agreed / compared:.0%} "
              f"(95% CI {low:.0%}-{high:.0%})")
        disagreements: dict[str, int] = {}
        for pair in rec.compared:
            if pair.agreed:
                continue
            key = f"we said {pair.signal.our_outcome}, broker said {pair.deal.outcome}"
            disagreements[key] = disagreements.get(key, 0) + 1
        for key, count in sorted(disagreements.items(), key=lambda kv: -kv[1]):
            print(f"  {key}: {count}")
    else:
        print("No comparable outcomes yet (nothing settled on both sides).")

    # A refunded trade is a real outcome and is reported whether or not anything
    # was comparable — it is neither a win nor a loss, so it is left out of every
    # rate below rather than silently counted as a loss.
    pushes = rec.broker_tally()[PUSH]
    if pushes:
        print(f"  refunded (profit 0, neither win nor loss): {pushes} — excluded")

    mismatched = [p for p in rec.pairs if p.direction_agreed is False]
    if mismatched:
        print(f"  ⚠ {len(mismatched)} deal(s) went the other way from the signal — "
              f"those are not the bot's trades")
    print()

    slips = [s for s in (p.entry_slippage_pips(
        pip_size(p.signal.asset, digits_by_asset.get(p.signal.asset)))
        for p in rec.pairs) if s is not None]
    if slips:
        print("Fill vs our reference price (signed, pips; positive = filled worse):")
        print(f"  median {median(slips):+.1f}   mean {sum(slips) / len(slips):+.1f}   "
              f"p90 {percentile(slips, 0.9):+.1f}   worst {max(slips):+.1f}")
        adverse = sum(1 for s in slips if s > 0)
        print(f"  {adverse}/{len(slips)} filled worse than the price we signalled on")
        print()

    payouts = rec.payouts()
    if payouts:
        print(f"Payouts actually paid: median {median(payouts):.0f}%, "
              f"range {min(payouts):.0f}%-{max(payouts):.0f}%")
    outcome = rec.broker_tally()
    decided = outcome[WIN] + outcome[LOSS]
    if decided:
        rate = outcome[WIN] / decided
        low, high = wilson(outcome[WIN], decided)
        print(f"Broker's outcome on these deals: {outcome[WIN]}W / {outcome[LOSS]}L"
              + (f" / {outcome[PUSH]} refunded" if outcome[PUSH] else "")
              + f" = {rate:.0%} (95% CI {low:.0%}-{high:.0%})")
        if payouts:
            even = break_even_win_rate(median(payouts))
            print(f"Break-even at {median(payouts):.0f}% payout: {even:.1%}")
            needed = trades_needed(rate, even)
            if needed is None:
                print("Verdict: the rate is at or below break-even — no sample size "
                      "turns this into a profit.")
            elif decided < needed:
                print(f"Verdict: not enough evidence yet. Distinguishing {rate:.0%} "
                      f"from the {even:.1%} break-even needs roughly {needed} "
                      f"settled trades; you have {decided}.")
            else:
                print(f"Verdict: {decided} settled trades is enough to separate "
                      f"{rate:.0%} from the {even:.1%} break-even — but check the "
                      f"dates and the account before believing it.")
    print()
    print("Caveats: this compares the bot's labels against the broker's settlements")
    print("for trades *you* placed. It says nothing about trades the bot never")
    print("signalled, and demo results are not live results.")


def _clock(at: float) -> str:
    import datetime
    from telegram.sender import MARKET_TZ
    return datetime.datetime.fromtimestamp(at, tz=MARKET_TZ).strftime("%m-%d %H:%M")


# ---------------------------------------------------------------------------
# fetching deals
# ---------------------------------------------------------------------------
async def fetch_deals(cfg: Config, settle_window: float, dump: Optional[str]
                      ) -> tuple[list[BrokerDeal], dict[str, int], int]:
    """Read the account's settled deals, and the digits of each asset.

    Attaches ``MemoryDealsStorage`` before the socket connects: the server pushes
    the deal history during authorisation, so a listener attached afterwards
    would miss it.
    """
    from pocket_option.contrib.deals import MemoryDealsStorage
    from data.pocket_option import PocketOptionFeed

    storage_holder: dict = {}
    feed = PocketOptionFeed(
        cfg.pocket_ssid, cfg.pocket_uid, cfg.pocket_is_demo,
        region=cfg.pocket_region or None, is_fast_history=True,
        on_client=lambda client: storage_holder.update(
            storage=MemoryDealsStorage(client)))

    await feed.connect()
    try:
        # Ask for the open deals too; the closed history arrives during auth.
        try:
            await feed.client.deals_update_opened()
        except Exception as exc:
            print(f"  (could not request open deals: {type(exc).__name__})")

        if settle_window > 0:
            print(f"  waiting {settle_window:.0f}s for the deal history to arrive…")
            await asyncio.sleep(settle_window)

        storage = storage_holder.get("storage")
        if storage is None:
            raise RuntimeError("the deals storage was never attached")
        raw = list(await storage.get_deals(query=None))

        if dump:
            _write_dump(dump, raw)

        deals, unparsed = _parse_all(raw)
        metas = await feed.asset_meta()
        digits = {symbol: meta.digits for symbol, meta in metas.items()}
        return deals, digits, unparsed
    finally:
        await feed.close()


def _parse_all(raw: Iterable[Any]) -> tuple[list[BrokerDeal], int]:
    deals, unparsed = [], 0
    for item in raw:
        deal = deal_from_model(item)
        if deal is None:
            unparsed += 1
            continue
        deals.append(deal)
    return deals, unparsed


def _write_dump(path: str, raw: Iterable[Any]) -> None:
    """Save the raw deals so the parsing can be checked without the account."""
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({k: plain(getattr(item, k, None))
                        for k in ("id", "asset", "command", "profit", "percent_profit",
                                  "percent_loss", "open_price", "close_price",
                                  "open_timestamp", "close_timestamp", "open_ms",
                                  "close_ms", "is_demo", "amount", "currency",
                                  "option_type", "request_id")})
    Path(path).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {len(out)} raw deal(s) to {path}")


def load_deals(path: str) -> tuple[list[BrokerDeal], int]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict):
        doc = doc.get("deals") or []
    deals, unparsed = _parse_all(doc)
    print(f"Read {len(deals)} deal(s) from {path}"
          + (f" ({unparsed} unusable)" if unparsed else ""))
    return deals, unparsed


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the bot's WIN/LOSS labels with the broker's settlements.")
    parser.add_argument("--deals", default=None,
                        help="reconcile from a saved deals dump instead of the account")
    parser.add_argument("--dump", default=None,
                        help="save the raw deals read from the account to this file")
    parser.add_argument("--journal", default=None,
                        help="signal journal path (default: SIGNAL_JOURNAL_PATH)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"seconds of slack when matching (default {DEFAULT_TOLERANCE:.0f})")
    parser.add_argument("--offset", type=float, default=None,
                        help="seconds the broker's clock runs ahead of ours; "
                             "measured from the data when not given")
    parser.add_argument("--settle-window", type=float, default=20.0,
                        help="seconds to wait for the deal history (default 20)")
    parser.add_argument("--since-hours", type=float, default=0.0,
                        help="only signals from the last N hours")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    # The times printed here are compared against the broker's, so they have to
    # be the same clock the signals were sent in — not a hardcoded one.
    configure_clock(cfg.market_tz_offset_hours)

    path = args.journal or cfg.signal_journal_path
    journal = load_journal(path)
    if args.since_hours > 0 and journal.trades:
        newest = max(t.entry_at for t in journal.trades)
        journal = journal.between(newest - args.since_hours * 3600, newest + 1)

    digits_by_asset: dict[str, int] = {}
    if args.deals:
        deals, unparsed = load_deals(args.deals)
    else:
        if not cfg.pocket_ssid:
            print("POCKET_SSID is not set, so the account cannot be read.\n"
                  "Set it in .env, or pass --deals FILE to reconcile offline.",
                  file=sys.stderr)
            return 2
        print("Reading deals from Pocket Option…")
        try:
            deals, digits_by_asset, unparsed = asyncio.run(
                fetch_deals(cfg, args.settle_window, args.dump))
        except Exception as exc:
            print(f"could not read the account: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        print(f"  {len(deals)} deal(s) parsed")

    if args.offset is None:
        clock_offset, hits, considered = detect_clock_offset(
            journal.trades, deals, args.tolerance)
        clock_offset = clock_offset or 0.0
    else:
        clock_offset, hits, considered = args.offset, 0, 0

    rec = match(journal, deals, args.tolerance, clock_offset)
    rec.unparsed_deals = unparsed
    rec.offset_hits = hits
    rec.offset_considered = considered
    rec.offset_given = args.offset is not None
    print()
    report(rec, cfg, args.tolerance, digits_by_asset, journal_path=path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
