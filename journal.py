"""Append-only record of every signal the bot sent, and how it says it ended.

``stats.json`` keeps *aggregates* — win counts per market and per hour. That is
enough to display a win rate, but not enough to check that win rate against
anything, because it no longer knows which trade was which. This journal keeps
the per-trade detail: for every signal, the market, direction and entry second;
and for every settlement, the two prices and the outcome the bot computed.

That detail is what makes the record auditable. ``reconcile.py`` matches these
entries against the deals in the Pocket Option account and reports where the
broker's settlement disagrees with the label the bot printed — which is the one
question the aggregates can never answer.

**Two record kinds, one file.** ``signal`` is written when the message goes out;
``result`` is written when the trade settles, joined to its signal by
``(asset, entry_at)``. Splitting them means the file is pure append — no rewrite,
no read-modify-write, nothing to corrupt if the process dies mid-trade — and a
signal that never got a result is itself a fact worth seeing (the bot stopped
before it could settle, or the feed stalled through the expiry).

A signal that has a ``signal`` line and no ``result`` line is *unsettled*, and the
reconciler reports it rather than assuming an outcome.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

log = logging.getLogger("pocket.journal")

KIND_SIGNAL = "signal"
KIND_RESULT = "result"
SCHEMA_VERSION = 1


def signal_id(asset: str, entry_at: float) -> str:
    """The join key for one signal: market plus the second it was to be entered.

    Deterministic on purpose. The same market cannot signal twice on the same
    boundary (the cooldown forbids it), so this identifies a signal uniquely —
    and the same string is what an automated order would use to recognise a
    deal it had already placed.
    """
    return f"{asset}@{int(entry_at)}"


@dataclass
class JournalledTrade:
    """One signal, with its settlement if there is one."""

    asset: str
    direction: str
    entry_at: float
    expiry_at: float
    payout: int = 0
    score: int = 0
    confidence: float = 0.0
    votes: tuple[str, ...] = ()
    sent_at: Optional[float] = None
    our_entry: Optional[float] = None
    our_exit: Optional[float] = None
    our_outcome: Optional[str] = None
    settled_at: Optional[float] = None
    # The broker's side, when the bot placed the order itself. ``our_outcome``
    # stays what the tick bars claimed; these are what the account actually did.
    # Absent whenever the trade was placed by hand or not at all.
    broker_outcome: Optional[str] = None
    broker_open_at: Optional[float] = None
    broker_close_at: Optional[float] = None
    broker_entry: Optional[float] = None
    broker_exit: Optional[float] = None
    broker_profit: Optional[float] = None
    broker_payout: Optional[float] = None
    deal_id: str = ""

    @property
    def id(self) -> str:
        return signal_id(self.asset, self.entry_at)

    @property
    def has_broker(self) -> bool:
        return self.broker_outcome is not None

    @property
    def outcome_of_record(self) -> Optional[str]:
        """The outcome to report: the broker's when there is one, else ours.

        The order matters. Our label is a reading of two bar closes; the
        broker's is the money. When both exist and disagree, the money is what
        happened.
        """
        if self.broker_outcome is not None:
            return self.broker_outcome
        return self.our_outcome

    @property
    def settled(self) -> bool:
        return self.outcome_of_record is not None

    def price_of(self, which: str) -> Optional[float]:
        return self.our_entry if which == "entry" else self.our_exit


class SignalJournal:
    """Writes the journal. Every failure is logged, never raised.

    A signal that cannot be recorded is a lost audit trail, not a reason to stop
    trading — so a full disk or a permissions problem degrades the record, not
    the bot.
    """

    def __init__(self, path: str | os.PathLike[str] = "signals.jsonl",
                 enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._warned = False

    def record_signal(self, *, asset: str, direction: str, entry_at: float,
                      expiry_at: float, payout: int = 0, score: int = 0,
                      confidence: float = 0.0, votes: Sequence[str] = (),
                      sent_at: Optional[float] = None) -> None:
        self._append({
            "kind": KIND_SIGNAL,
            "id": signal_id(asset, entry_at),
            "asset": asset,
            "direction": direction,
            "entry_at": float(entry_at),
            "expiry_at": float(expiry_at),
            "payout": int(payout),
            "score": int(score),
            "confidence": round(float(confidence), 4),
            "votes": list(votes),
            "sent_at": float(sent_at if sent_at is not None else time.time()),
        })

    def record_result(self, *, asset: str, entry_at: float, outcome: str,
                      entry_price: Optional[float] = None,
                      exit_price: Optional[float] = None,
                      settled_at: Optional[float] = None,
                      broker: Optional[dict] = None) -> None:
        """Record how the trade ended.

        ``outcome`` is always the bot's own label from the tick bars. ``broker``
        is the settlement the account produced, when the bot placed the order —
        nested rather than flattened so that "we did not trade this" stays
        distinguishable from "we traded it and the broker said nothing".
        """
        record = {
            "kind": KIND_RESULT,
            "id": signal_id(asset, entry_at),
            "asset": asset,
            "entry_at": float(entry_at),
            "outcome": outcome,
            "our_entry": _num(entry_price),
            "our_exit": _num(exit_price),
            "settled_at": float(settled_at if settled_at is not None else time.time()),
        }
        if broker:
            record["broker"] = _clean_broker(broker)
        self._append(record)

    def _append(self, record: dict) -> None:
        if not self.enabled:
            return
        record["version"] = SCHEMA_VERSION
        line = json.dumps(record, separators=(",", ":"), default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One write of one whole line, opened per record: an interrupted run
            # can leave a truncated final line, which the reader skips.
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            if not self._warned:
                self._warned = True
                log.warning("cannot write the signal journal at %s: %s "
                            "(reconciliation will have nothing to compare)",
                            self.path, exc)


def _num(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _clean_broker(broker: dict) -> dict:
    """Keep only the broker fields worth storing, with numbers coerced."""
    kept = {}
    for key, value in broker.items():
        if value is None or value == "":
            continue
        if key in ("open_at", "close_at", "entry", "exit", "profit", "payout"):
            number = _opt_float(value)
            if number is not None:
                kept[key] = number
        else:
            kept[key] = value
    return kept


@dataclass
class LoadedJournal:
    """A journal folded back into per-trade records."""

    trades: list[JournalledTrade] = dc_field(default_factory=list)
    bad_lines: int = 0
    unknown_results: int = 0

    @property
    def settled(self) -> list[JournalledTrade]:
        return [t for t in self.trades if t.settled]

    @property
    def unsettled(self) -> list[JournalledTrade]:
        return [t for t in self.trades if not t.settled]

    def outcomes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.settled:
            counts[trade.our_outcome or "?"] = counts.get(trade.our_outcome or "?", 0) + 1
        return counts

    def our_win_rate(self) -> float:
        wins = sum(1 for t in self.settled if t.our_outcome == "WIN")
        n = len(self.settled)
        return wins / n if n else 0.0

    # -- the broker's side ---------------------------------------------------
    @property
    def placed(self) -> list[JournalledTrade]:
        """Signals the bot actually placed an order for."""
        return [t for t in self.trades if t.has_broker]

    def broker_outcomes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.placed:
            key = trade.broker_outcome or "?"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def broker_win_rate(self) -> Optional[float]:
        """The settled win rate, refunds excluded. None when nothing settled."""
        decided = [t for t in self.placed if t.broker_outcome in ("WIN", "LOSS")]
        if not decided:
            return None
        return sum(1 for t in decided if t.broker_outcome == "WIN") / len(decided)

    def agreements(self) -> tuple[int, int]:
        """(times our label matched the broker's, times they were comparable).

        Only trades where *both* sides reached a win-or-loss count. A refund is
        not a loss, and a trade we never labelled is not a disagreement.
        """
        agreed = compared = 0
        for trade in self.placed:
            if trade.broker_outcome not in ("WIN", "LOSS"):
                continue
            if trade.our_outcome not in ("WIN", "LOSS"):
                continue
            compared += 1
            if trade.our_outcome == trade.broker_outcome:
                agreed += 1
        return agreed, compared

    # -- what the record is worth --------------------------------------------
    def expected_value(self) -> "ExpectedValue":
        """The per-trade return, with the uncertainty that is honestly attached.

        A win rate cannot be read on its own here, because the payout is not
        fixed: 53% of trades won is a losing record at a 92% payout (break-even
        is 52.1%) and a comfortable one at 60% (break-even 62.5%). The number
        that *is* comparable across trades is what each one returned per unit
        staked, so that is the one computed — and averaged.

        The interval matters as much as the mean. A handful of trades cannot
        establish a small edge: at 28 settled trades a 2.9-point edge is
        indistinguishable from luck. Reporting the mean without it would turn
        noise into a result, which is the failure mode this whole file exists to
        avoid.

        Only settled trades count, and the arithmetic assumes a flat stake —
        which is what the bot places. A winning trade with no payout on record
        is left out rather than guessed at; that can only pull the average
        *down*, never up, since the missing returns are positive ones.
        """
        returns: list[float] = []
        unpriced = unjudged = 0
        for trade in self.settled:
            value = trade_return(trade)
            if value is None:
                if trade.outcome_of_record == "WIN":
                    unpriced += 1
                else:
                    unjudged += 1
                continue
            returns.append(value)

        n = len(returns)
        if not n:
            return ExpectedValue(unpriced=unpriced, unjudged=unjudged)
        mean = sum(returns) / n
        if n < 2:
            # One trade has no spread to estimate, so there is no interval to
            # print — only the number itself.
            return ExpectedValue(n=n, mean=mean, unpriced=unpriced,
                                 unjudged=unjudged)
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        stdev = math.sqrt(variance)
        half = Z_95 * stdev / math.sqrt(n)
        return ExpectedValue(n=n, mean=mean, stdev=stdev, low=mean - half,
                             high=mean + half, unpriced=unpriced,
                             unjudged=unjudged)

    def between(self, start: float, end: float) -> "LoadedJournal":
        """Only the signals entered in ``[start, end]`` — the reconcile window."""
        kept = [t for t in self.trades if start <= t.entry_at <= end]
        return LoadedJournal(trades=kept, bad_lines=self.bad_lines,
                             unknown_results=self.unknown_results)

    def __iter__(self) -> Iterator[JournalledTrade]:
        return iter(self.trades)

    def __len__(self) -> int:
        return len(self.trades)


# The z for a two-sided 95% interval — the ordinary convention, named so the
# arithmetic below is not a magic number.
Z_95 = 1.96


@dataclass(frozen=True)
class ExpectedValue:
    """What the record returned per trade, and how well that is known."""

    n: int = 0
    mean: float = 0.0
    stdev: float = 0.0
    # Absent when there is one trade or none: a spread needs two points.
    low: Optional[float] = None
    high: Optional[float] = None
    # Settled trades kept out of the average, and why: a win whose payout was
    # never recorded, and a settlement that was neither a win, a loss nor a
    # refund (the broker's answer could not be read).
    unpriced: int = 0
    unjudged: int = 0

    @property
    def excluded(self) -> int:
        return self.unpriced + self.unjudged

    @property
    def verdict(self) -> str:
        """The one-line reading: does the interval clear break-even?"""
        if self.low is None:
            return "too few to judge"
        if self.low > 0:
            return "beats break-even"
        if self.high < 0:
            return "under break-even"
        return "no edge shown yet"


def trade_return(trade: JournalledTrade) -> Optional[float]:
    """What one settled trade returned per unit staked, or None if unknowable.

    A win pays the payout the trade was actually on — the broker's own number
    when the bot placed the order, the quote from signal time otherwise. A loss
    takes the stake, which is true whatever the payout was, so a loss is never
    unknowable. A refund returns it untouched.
    """
    outcome = trade.outcome_of_record
    if outcome == "PUSH":
        return 0.0
    if outcome == "LOSS":
        return -1.0
    if outcome != "WIN":
        return None
    payout = trade.broker_payout or trade.payout
    if payout <= 0:
        return None
    return payout / 100.0


def format_ev(ev: ExpectedValue) -> str:
    """The facts, as one line for the log and for Telegram. "" if none.

    Plain ASCII, on purpose, and that is not a style choice: the startup line
    goes through ``logging`` to a stream Windows opens in the locale codepage,
    where an emoji does not encode and the whole line is lost rather than
    mangled. Callers that want a symbol in a Telegram message add their own.

    Also free of ``<`` and ``>``: the status message is otherwise hand-written
    HTML, and an interval written as ``-0.2..+0.1`` needs no escaping.
    """
    dropped = _exclusions(ev)
    if not ev.n:
        # Nothing priced. If trades were left out, name them: that is why there
        # is no number, and a silent drop would read as an empty record.
        return f"EV unknown | {dropped}" if dropped else ""
    if ev.low is None:
        text = f"EV {ev.mean:+.3f} per trade | n={ev.n}"
    else:
        text = (f"EV {ev.mean:+.3f} per trade | 95% CI "
                f"{ev.low:+.2f}..{ev.high:+.2f} | n={ev.n}")
    if dropped:
        text += f" | {dropped}"
    return f"{text} | {ev.verdict}"


def _exclusions(ev: ExpectedValue) -> str:
    """The trades kept out of the average, and what stopped them counting."""
    parts = []
    if ev.unpriced:
        parts.append(f"{ev.unpriced} win(s) unpriced")
    if ev.unjudged:
        parts.append(f"{ev.unjudged} unreadable")
    return ", ".join(parts)


def load_journal(path: str | os.PathLike[str]) -> LoadedJournal:
    """Read a journal, folding ``result`` lines into their ``signal`` lines.

    Tolerant by design: this reads files written by a bot that may have been
    killed mid-write, so an unparseable or unknown line is counted and skipped
    rather than aborting the whole reconciliation.
    """
    doc = LoadedJournal()
    by_id: dict[str, JournalledTrade] = {}
    path = Path(path)
    if not path.exists():
        return doc

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                doc.bad_lines += 1
                continue
            if not isinstance(record, dict):
                doc.bad_lines += 1
                continue

            kind = record.get("kind")
            if kind == KIND_SIGNAL:
                trade = _trade_from_signal(record)
                if trade is None:
                    doc.bad_lines += 1
                    continue
                by_id[trade.id] = trade
                doc.trades.append(trade)
            elif kind == KIND_RESULT:
                target = by_id.get(record.get("id") or "")
                if target is None:
                    # A result whose signal is missing (a rotated journal, or a
                    # line lost to a hard kill). Counted, not invented.
                    doc.unknown_results += 1
                    continue
                target.our_outcome = record.get("outcome")
                target.our_entry = _opt_float(record.get("our_entry"))
                target.our_exit = _opt_float(record.get("our_exit"))
                target.settled_at = _opt_float(record.get("settled_at"))
                _apply_broker(target, record.get("broker"))
            else:
                doc.bad_lines += 1

    doc.trades.sort(key=lambda t: t.entry_at)
    return doc


def _apply_broker(trade: JournalledTrade, broker) -> None:
    """Fold the broker's settlement into a trade. A missing block is left alone."""
    if not isinstance(broker, dict):
        return
    trade.broker_outcome = broker.get("outcome") or trade.broker_outcome
    trade.broker_open_at = _opt_float(broker.get("open_at"))
    trade.broker_close_at = _opt_float(broker.get("close_at"))
    trade.broker_entry = _opt_float(broker.get("entry"))
    trade.broker_exit = _opt_float(broker.get("exit"))
    trade.broker_profit = _opt_float(broker.get("profit"))
    trade.broker_payout = _opt_float(broker.get("payout"))
    trade.deal_id = str(broker.get("deal_id") or "")


def _trade_from_signal(record: dict) -> Optional[JournalledTrade]:
    asset, direction = record.get("asset"), record.get("direction")
    entry_at, expiry_at = _opt_float(record.get("entry_at")), _opt_float(record.get("expiry_at"))
    if not asset or not direction or entry_at is None or expiry_at is None:
        return None
    votes = record.get("votes") or ()
    return JournalledTrade(
        asset=str(asset),
        direction=str(direction),
        entry_at=entry_at,
        expiry_at=expiry_at,
        payout=int(record.get("payout") or 0),
        score=int(record.get("score") or 0),
        confidence=float(record.get("confidence") or 0.0),
        votes=tuple(str(v) for v in votes) if isinstance(votes, Iterable) else (),
        sent_at=_opt_float(record.get("sent_at")),
    )


def _opt_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
