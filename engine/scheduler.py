"""The trading loop.

Every bar the scheduler wakes ``SIGNAL_LEAD_SECONDS`` before a boundary, asks the
engine to judge all tracked markets, and sends at most one signal — the
best-ranked candidate. The trade is placed on the boundary itself and expires one
bar later, so both entry and exit land on real bar closes:

    ... bar ending at T [entry price] ... bar ending at T+period [exit price]

That is what makes the reported win rate *measured* rather than estimated: both
prices are bar closes the feed already delivered, not tick snapshots taken
whenever a message happened to be read.

The lead is capped at one bar, which leaves two regimes with two timing models.

*Short lead* — the bar being judged is still in progress when the signal goes
out, so the engine sees a snapshot of it (period=60s, lead=10s):

    T-10s  wake, finalise bars, settle due trades, evaluate, rank, send
    T      the trade opens
    T+50s  the bar that closed at T is finalised -> exact entry price
    T+110s the bar closing at T+60 is finalised -> exact exit price, result sent

*Full-bar lead* — the bar closing at the entry boundary has not started yet, so
there is nothing to snapshot and the decision is made on the bars already closed
(period=300s, lead=300s):

    16:00  wake, evaluate on the bar that closed at 16:00, send for entry 16:05
    16:05  the trade opens, as the previous one expires
    16:10  the bar that closed at 16:10 is finalised -> exit price, result sent

Either way the loop owns no market logic of its own: it reads ``MarketState``,
asks the engine for candidates, hands the winner to Telegram, and records the
outcome in ``StatsTracker``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from execution import UNPLACED, DemoBroker, Order
from journal import SignalJournal, load_journal
from market.aggregator import MarketState
from market.clock import Clock, RealClock
from signals.engine import Candle, Readiness, SignalConfig, evaluate, readiness
from signals.ranking import Candidate, describe, select_best
from stats import StatsTracker
from telegram.control import BotController
from telegram.sender import (
    TelegramSender, clock, format_duration, format_startup, format_status,
    format_warmup,
)

log = logging.getLogger("pocket.scheduler")

# How often to repeat the warm-up progress note while waiting for indicators.
_PROGRESS_EVERY = 600.0


def next_boundary(now: float, period: int) -> int:
    """The first bar boundary strictly after ``now`` (epoch seconds)."""
    return (int(math.floor(now / period)) + 1) * period


def outcome_of(direction: str, entry: float, exit_price: float) -> str:
    """WIN if price moved the signal's way between entry and expiry.

    A tie counts as a loss, matching how a flat close settles a binary option.
    """
    if direction == "CALL":
        return "WIN" if exit_price > entry else "LOSS"
    return "WIN" if exit_price < entry else "LOSS"


def _stamp(ts: float) -> str:
    """Local wall clock for the logs. Delegates so it cannot go stale when the
    offset is reconfigured at startup — a bound ``MARKET_TZ`` would not."""
    return clock(ts)


@dataclass
class Cadence:
    """Timing and risk limits for the loop."""

    period: int = 60
    expiry_seconds: int = 60
    lead_seconds: int = 10
    cooldown_seconds: int = 120
    min_confidence: float = 0.0
    martingale_steps: int = 0
    # Stop trading for a while after this many losses in a row (0 disables).
    max_consecutive_losses: int = 0
    pause_minutes: int = 15


@dataclass
class OpenTrade:
    asset: str
    direction: str
    entry_at: float
    expiry_at: float
    confidence: float = 0.0
    score: int = 0
    payout: int = 0
    votes: tuple[str, ...] = ()
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    # The order we placed, if the bot is trading the account itself.
    order: Optional[Order] = None
    # Set when the broker settled it: what the money did, as opposed to what
    # the tick bars say. ``None`` means the broker has not answered yet.
    broker_outcome: Optional[str] = None
    broker: Optional[dict] = None
    # Taken back from the journal at startup rather than opened by this process.
    recovered: bool = False


@dataclass
class CycleResult:
    """What one bar produced — returned by ``run_cycle`` so tests can inspect it."""

    boundary: int
    candidates: list[Candidate] = field(default_factory=list)
    chosen: Optional[Candidate] = None
    resolved: list[OpenTrade] = field(default_factory=list)
    paused: bool = False
    healthy: int = 0


class MinuteScheduler:
    def __init__(self, *, market: MarketState, store=None,
                 sender: Optional[TelegramSender] = None,
                 stats: StatsTracker, controller: BotController, cadence: Cadence,
                 engine_config: SignalConfig, symbols: Sequence[str],
                 payouts: Optional[dict[str, int]] = None,
                 clock: Optional[Clock] = None, journal: Optional[SignalJournal] = None,
                 mode_label: str = "", min_payout: int = 0,
                 broker: Optional[DemoBroker] = None) -> None:
        self.market = market
        self.store = store
        self.sender = sender
        self.stats = stats
        # Places the order when the bot is trading the account itself. Without
        # one the scheduler only signals, exactly as it always has.
        self.broker = broker
        # Per-trade record for reconciling against the broker. Optional so the
        # scheduler still runs without one (the record is an audit trail, not a
        # dependency of trading).
        self.journal = journal
        self.controller = controller
        self.cadence = cadence
        self.engine_config = engine_config
        self.symbols = list(symbols)
        self.payouts = dict(payouts or {})
        self.clock = clock or RealClock()
        self.mode_label = mode_label or f"{len(self.symbols)} markets"
        self.min_payout = min_payout

        self.last_signal_at: dict[str, float] = {}
        self.open_trades: list[OpenTrade] = []
        self.next_entry_at: Optional[int] = None
        self.signals_sent = 0
        self._paused_until: float = 0.0
        self._warmup_notified_at: float = 0.0
        self._ready_notified = False
        self._last_readiness: Optional[Readiness] = None

        if (cadence.expiry_seconds % cadence.period) and cadence.expiry_seconds > 0:
            log.warning("expiry %ss is not a whole number of %ss bars - exit prices "
                        "will fall back to the latest tick",
                        cadence.expiry_seconds, cadence.period)

    # -- main loop -----------------------------------------------------------
    async def run(self) -> None:
        """Loop forever, one decision per bar."""
        self.recover_from_journal()
        await self._announce_startup()
        while not self.controller.stop_requested:
            now = self.clock.now()
            boundary = next_boundary(now, self.cadence.period)
            fire_at = boundary - self.cadence.lead_seconds
            if fire_at <= now:
                boundary += self.cadence.period
                fire_at = boundary - self.cadence.lead_seconds
            self.next_entry_at = boundary
            await self.clock.sleep_until(fire_at)
            if self.controller.stop_requested:
                break
            await self.run_cycle(boundary)
            await self._announce_progress()

    def recover_from_journal(self, now: Optional[float] = None) -> list[OpenTrade]:
        """Pick up what the previous process left open, if the bars can prove it.

        ``open_trades`` and ``last_signal_at`` live in memory only, so a restart
        used to lose both. What that cost was measured on 2026-09-15: a signal
        sent at 20:15 for the 20:15 entry, the process dying at 20:19, and the
        trade never settled — no result in the journal, no outcome in the stats,
        nothing in any log. It stayed the one unsettled signal in
        ``signals.jsonl``, which is exactly what an unsettled signal is supposed
        to mean, so the loss looked like a fact rather than a defect.

        A signal is taken back only when the restored bars can still price it,
        or when the bars it needs have not happened yet: an entry that is still
        ahead is settled from the live feed exactly as it would have been, and
        an exit that is still ahead likewise. Both prices are readings of bar
        closes the feed really delivered, so where they exist the settlement is
        the same one the original session would have produced — and where they
        do not, the trade is left unsettled rather than given an outcome nobody
        observed.

        That is the honest limit, and it is why this cannot resurrect the 20:15
        trade: the hole left by the shrinking universe had already taken its
        bars off the disk before the process died.

        The cooldown clocks are restored at the same time, so a restart cannot
        signal a market the previous process had just signalled.
        """
        if self.journal is None:
            return []
        try:
            loaded = load_journal(self.journal.path)
        except OSError as exc:
            log.warning("could not read %s to recover open trades: %s",
                        self.journal.path, exc)
            return []

        now = self.clock.now() if now is None else now
        for entry in loaded.trades:
            sent = entry.sent_at if entry.sent_at is not None else entry.entry_at
            if sent > self.last_signal_at.get(entry.asset, 0.0):
                self.last_signal_at[entry.asset] = sent

        adopted: list[OpenTrade] = []
        unpriced: list[str] = []
        for entry in loaded.unsettled:
            series = self.market.track(entry.asset)
            if entry.entry_at > now:
                # Signalled but not yet entered — the bar it needs is ahead of
                # us, not missing. A restart in the seconds between the message
                # and the entry is the common case, and dropping it here would
                # manufacture exactly the orphan this method exists to prevent.
                entry_price = None
            else:
                entry_price = series.price_at_boundary(entry.entry_at)
                if entry_price is None:
                    unpriced.append(entry.id)
                    continue
            if entry.expiry_at <= now and series.price_at_boundary(entry.expiry_at) is None:
                unpriced.append(entry.id)
                continue
            trade = OpenTrade(
                asset=entry.asset, direction=entry.direction,
                entry_at=entry.entry_at, expiry_at=entry.expiry_at,
                confidence=entry.confidence, score=entry.score,
                payout=entry.payout, votes=tuple(entry.votes),
                entry_price=entry_price, recovered=True)
            self.open_trades.append(trade)
            adopted.append(trade)

        if adopted:
            log.info("recovered %d trade(s) the previous session left open: %s",
                     len(adopted),
                     ", ".join(f"{t.asset} {t.direction} entered {_stamp(t.entry_at)}"
                               for t in adopted))
        if unpriced:
            log.warning("%d journalled signal(s) cannot be settled — their bars are "
                        "no longer on disk, so there is no price to settle them at. "
                        "Left unsettled: %s", len(unpriced), ", ".join(unpriced))
        return adopted

    async def run_cycle(self, boundary: int) -> CycleResult:
        """One bar's work: settle, judge, send. Safe to call directly in tests."""
        now = self.clock.now()
        self.market.finalize(now)

        result = CycleResult(boundary=boundary)
        result.resolved = await self._resolve_due(now)

        healthy = self.market.healthy(now)
        result.healthy = len(healthy)
        self._last_readiness = self._readiness(healthy)

        if self._blocked(now):
            result.paused = True
            await self._persist()
            return result

        candidates = self._collect(healthy, boundary)
        result.candidates = candidates
        winner = select_best(candidates, self.last_signal_at, now,
                             self.cadence.cooldown_seconds,
                             self.cadence.min_confidence)
        if winner is None:
            if candidates:
                log.info("[%s] %d candidate(s), none eligible: %s",
                         _stamp(boundary), len(candidates), describe(candidates))
            await self._persist()
            return result

        result.chosen = winner
        await self._send_signal(winner, boundary, len(candidates))
        await self._persist()
        return result

    # -- status --------------------------------------------------------------
    def status_text(self) -> str:
        """Short report for the /status command."""
        now = self.clock.now()
        healthy = self.market.healthy(now)
        bars = max((self.market.track(s).bar_count for s in healthy), default=0)
        r = self._readiness(healthy)
        if self._paused_until and now < self._paused_until:
            note = f" (paused {int((self._paused_until - now) / 60) + 1} min)"
        else:
            note = ""
        return format_status(
            running=not self.controller.paused and not note,
            assets=len(self.symbols), healthy=len(healthy), bars=bars,
            bars_needed=max(r.bars_needed, 1), next_entry_at=self.next_entry_at,
            now=now) + note

    # -- internals -----------------------------------------------------------
    def _blocked(self, now: float) -> bool:
        if self.controller.paused:
            log.debug("paused by command - skipping this bar")
            return True
        if self._paused_until:
            if now < self._paused_until:
                log.info("circuit breaker: not trading for another %d min",
                         int((self._paused_until - now) / 60) + 1)
                return True
            self._paused_until = 0.0
        return False

    def _collect(self, healthy: Sequence[str], boundary: int) -> list[Candidate]:
        """Judge every healthy market on the bars it can see at the signal.

        Which bars those are depends on the lead, and the difference is not
        cosmetic. ``buffer_for_boundary`` ends with a *snapshot of the bar still
        in progress*, which is the right input a few seconds before that bar
        closes. At a full-bar lead the bar closing at ``boundary`` has not
        started when the signal goes out, so that "snapshot" is zero seconds
        old — a single tick with ``open == high == low == close``, which would
        drag every indicator window it lands in. Worse, it would do so only when
        a tick happened to land before the cycle ran, so the distortion would
        come and go. At a full-bar lead the decision window is simply the bars
        that have already closed.
        """
        closed_only = self.cadence.lead_seconds >= self.cadence.period
        candidates: list[Candidate] = []
        for symbol in healthy:
            series = self.market.track(symbol)
            if closed_only:
                buffer = series.closed()
            else:
                buffer = series.buffer_for_boundary(boundary)
            if len(buffer) < 2:
                continue
            signal = evaluate(buffer, self.engine_config)
            if signal is None:
                continue
            candidates.append(Candidate(
                asset=symbol, signal=signal,
                payout=self.payouts.get(symbol, 0), bars=len(buffer)))
        return candidates

    async def _send_signal(self, winner: Candidate, boundary: int,
                           candidate_count: int) -> None:
        now = self.clock.now()
        self.last_signal_at[winner.asset] = now
        expiry_at = boundary + self.cadence.expiry_seconds
        trade = OpenTrade(
            asset=winner.asset, direction=winner.signal.direction,
            entry_at=float(boundary), expiry_at=float(expiry_at),
            confidence=winner.confidence, score=winner.signal.score,
            payout=winner.payout, votes=tuple(winner.signal.votes))
        self.open_trades.append(trade)
        self.signals_sent += 1
        self._submit_order(trade)

        if self.journal is not None:
            self.journal.record_signal(
                asset=winner.asset, direction=winner.signal.direction,
                entry_at=float(boundary), expiry_at=float(expiry_at),
                payout=winner.payout, score=winner.signal.score,
                confidence=winner.confidence, votes=winner.signal.votes,
                sent_at=now)

        log.info("[%s] SIGNAL %s %s score=%d conf=%.2f payout=%d%% votes=%s "
                 "(entry %s expiry %s) chosen from %d candidate(s)",
                 _stamp(boundary), winner.asset, winner.signal.direction,
                 winner.score, winner.confidence, winner.payout,
                 ",".join(winner.signal.votes) or "-",
                 _stamp(boundary), _stamp(expiry_at), candidate_count)

        if self.sender is not None:
            try:
                await self.sender.send_signal(
                    winner.signal, winner.asset, self._expiry_label(),
                    entry_at=float(boundary), now=now, payout=winner.payout,
                    martingale_steps=self.cadence.martingale_steps,
                    entry_price=winner.signal.price)
            except Exception as exc:
                log.error("signal send failed: %s", exc)

    def _submit_order(self, trade: OpenTrade) -> None:
        """Hand the trade to the broker, to be opened on its entry second.

        Fire and forget on purpose. ``open_deal`` waits up to 30s for the
        server to confirm and the settlement takes a further minute; awaiting
        either here would stall the minute loop past the next boundary. The
        order runs on its own task and the result is read at settlement.
        """
        if self.broker is None or not self.broker.enabled:
            return
        order = Order(
            asset=trade.asset, direction=trade.direction,
            entry_at=trade.entry_at, expiry_at=trade.expiry_at,
            amount=self.broker.amount,
            duration=self.cadence.expiry_seconds)
        trade.order = order
        self.broker.submit(order)
        log.info("[%s] order submitted for %s %s (entry %s, %g)",
                 _stamp(trade.entry_at), trade.asset, trade.direction,
                 _stamp(trade.entry_at), order.amount)

    def _read_settlement(self, trade: OpenTrade) -> None:
        """Take the broker's verdict on this trade if it has arrived.

        Read, never awaited: by the time a trade settles the order has had a
        full bar to finish. Nothing arriving is left as ``None`` — the trade is
        then reported from the bars and recorded with no broker block, so it
        counts as unconfirmed rather than being given an outcome the account
        never produced.
        """
        if self.broker is None or trade.order is None:
            return
        settlement = self.broker.settlement_for(trade.asset, trade.entry_at)
        if settlement is not None:
            trade.broker_outcome = settlement.outcome
            trade.broker = settlement.as_dict()
            return
        if trade.order.error:
            trade.broker_outcome = UNPLACED
            trade.broker = {"outcome": UNPLACED, "error": trade.order.error}
            log.error("no order was placed for %s %s: %s",
                      trade.asset, _stamp(trade.entry_at), trade.order.error)
            return
        log.warning("no broker settlement for %s %s — reporting the bar label",
                    trade.asset, _stamp(trade.entry_at))

    def _broker_note(self, trade: OpenTrade, label: str) -> str:
        """How the broker's settlement sat against our own label, for the log."""
        if trade.broker_outcome is None:
            return " | broker: no settlement" if self.broker is not None else ""
        if trade.broker_outcome in ("WIN", "LOSS") and trade.broker_outcome != label:
            return f" | broker {trade.broker_outcome}, bars said {label}"
        return f" | broker {trade.broker_outcome}"

    async def _resolve_due(self, now: float) -> list[OpenTrade]:
        """Settle every trade whose entry and exit bars are both final."""
        resolved: list[OpenTrade] = []
        for trade in list(self.open_trades):
            series = self.market.track(trade.asset)

            if trade.entry_price is None and now >= trade.entry_at:
                price = series.price_at_boundary(trade.entry_at)
                if price is not None:
                    trade.entry_price = price

            if now < trade.expiry_at:
                continue

            exit_price = series.price_at_boundary(trade.expiry_at)
            if exit_price is None:
                # An expiry that is not a whole number of bars, or a stalled
                # feed. Give the feed one more bar to produce the exit bar, then
                # fall back to the latest tick rather than hanging forever.
                if now < trade.expiry_at + self.cadence.period:
                    continue
                exit_price = series.last_price
                if exit_price <= 0:
                    continue
            entry_price = trade.entry_price
            if entry_price is None:
                entry_price = series.price_at_boundary(trade.entry_at) or exit_price

            # Our own label, from the tick bars. Always computed and always
            # journalled — it is the claim the broker's settlement is checked
            # against, so it has to be recorded whether or not it is the one
            # reported.
            label = outcome_of(trade.direction, entry_price, exit_price)
            trade.entry_price = entry_price
            trade.exit_price = exit_price

            self._read_settlement(trade)
            outcome = trade.broker_outcome or label
            counted = outcome in ("WIN", "LOSS")

            if counted:
                self.stats.record(trade.asset, trade.direction, outcome,
                                  at=trade.expiry_at)
            if self.journal is not None:
                self.journal.record_result(
                    asset=trade.asset, entry_at=trade.entry_at, outcome=label,
                    entry_price=entry_price, exit_price=exit_price,
                    settled_at=now, broker=trade.broker)
            self.open_trades.remove(trade)
            resolved.append(trade)
            log.info("[%s] %s %s -> %s (entry %s exit %s)%s%s | %dW/%dL = %.0f%%",
                     _stamp(trade.expiry_at), trade.asset, trade.direction, outcome,
                     entry_price, exit_price, self._broker_note(trade, label),
                     " (recovered)" if trade.recovered else "",
                     self.stats.wins, self.stats.losses, self.stats.win_rate * 100)

            if self.sender is not None:
                try:
                    await self.sender.send_confirmation(
                        trade.asset, trade.direction, outcome,
                        self.stats.wins, self.stats.losses, self.stats.win_rate,
                        expiry_at=trade.expiry_at, streak=self._streak_label(),
                        note=(f"↩️ Settled after a restart "
                              f"(entered {_stamp(trade.entry_at)})"
                              if trade.recovered else ""))
                except Exception as exc:
                    log.error("confirmation send failed: %s", exc)

            await self._apply_circuit_breaker(now)

        if resolved:
            self.stats.save()
        return resolved

    async def _apply_circuit_breaker(self, now: float) -> None:
        limit = self.cadence.max_consecutive_losses
        if limit <= 0 or self._paused_until:
            return
        losses = self.stats.consecutive_losses()
        if losses < limit:
            return
        self._paused_until = now + self.cadence.pause_minutes * 60
        log.warning("circuit breaker: %d losses in a row - pausing %d minutes",
                    losses, self.cadence.pause_minutes)
        if self.sender is not None:
            try:
                await self.sender.send_text(
                    f"⚠️ {losses} losses in a row — pausing for "
                    f"{self.cadence.pause_minutes} minutes.")
            except Exception as exc:
                log.error("breaker notice failed: %s", exc)

    def _streak_label(self) -> str:
        if self.stats.streak > 1:
            return f"{self.stats.streak} wins in a row"
        if self.stats.streak < -1:
            return f"{self.stats.consecutive_losses()} losses in a row"
        return ""

    def _expiry_label(self) -> str:
        return format_duration(self.cadence.expiry_seconds)

    def _readiness(self, healthy: Sequence[str]) -> Readiness:
        """Readiness of the most advanced market (they warm up together)."""
        bars: list[Candle] = []
        for symbol in (list(healthy) or self.symbols):
            closed = self.market.track(symbol).closed()
            if len(closed) > len(bars):
                bars = closed
        return readiness(bars, self.engine_config)

    async def _announce_startup(self) -> None:
        if self.sender is None:
            return
        restored = sum(self.market.track(s).bar_count for s in self.symbols)
        await self.sender.send_text(format_startup(
            assets=len(self.symbols), period=self.cadence.period,
            expiry=self._expiry_label(), mode=self.mode_label,
            min_payout=self.min_payout, bars_restored=restored,
            execution="" if self.broker is None else self.broker.summary()))

    async def _announce_progress(self) -> None:
        """One warm-up note, one 'ready' note, then periodic progress."""
        r = self._last_readiness
        if r is None or self.sender is None:
            return
        now = self.clock.now()
        if r.ready:
            if not self._ready_notified:
                self._ready_notified = True
                await self.sender.send_text(format_warmup(
                    r, eta_minutes=0, available=", ".join(r.available)))
            return
        if now - self._warmup_notified_at < _PROGRESS_EVERY:
            return
        self._warmup_notified_at = now
        remaining = max(0, r.bars_needed - r.bars)
        await self.sender.send_text(format_warmup(
            r, eta_minutes=int(remaining * self.cadence.period / 60),
            available=", ".join(r.available)))

    async def _persist(self) -> None:
        if self.store is None:
            return
        try:
            self.store.save_all({s: self.market.track(s) for s in self.symbols})
        except Exception as exc:
            log.warning("persist failed: %s", exc)
