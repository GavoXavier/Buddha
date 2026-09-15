"""Pocket Option signal bot — entry point.

Wiring, and nothing else: build the feed, subscribe the major OTC forex pairs,
let the feed's ticks build candles, and hand those to the scheduler, which sends
at most one signal per minute. Market logic lives in ``signals``, timing in
``engine/scheduler``, candles in ``market``.

    FEED=simulated     python main.py    # real-time, no account needed
    FEED=pocket_option python main.py    # live ticks from your account

The reconnect loop is the outer ``while``: any feed failure — a dead socket, a
silent stream, an expired SSID — tears the session down and starts a new one
with exponential backoff. Candle history is restored from disk on each attempt,
so a reconnect does not cost another hour of warm-up.

The ladder resets only after a session that actually stayed up (see
``HEALTHY_SESSION_SECONDS``). A feed that accepts connections and then dies
seconds later is one outage, not a series of them, and backing off from it
has to mean backing off.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

# Windows consoles default to cp1252, which can't print emoji. Force UTF-8 so
# unicode in log output can never crash the process.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import Config, ConfigError, load_config
from data import DataFeed, SimulatedFeed
from data.pocket_option import PocketOptionFeed
from engine.scheduler import MinuteScheduler
from execution import DemoBroker
from journal import SignalJournal
from market.aggregator import MarketState
from market.clock import Clock, RealClock
from market.store import CandleStore
from market.universe import (
    AssetMeta, describe_skipped, format_universe, select_assets,
)
from stats import StatsTracker
from telegram.control import BotController, TelegramControl
from telegram.sender import TelegramSender, configure_clock

log = logging.getLogger("pocket")

# How often the watchdog checks the feed. Cheap — it reads a transport flag and
# a timestamp — and it is the whole detection delay for a dropped connection,
# so it is short.
WATCH_INTERVAL = 5.0
# Reconnect backoff bounds.
BACKOFF_MIN = 5.0
BACKOFF_MAX = 60.0
# How long a session has to stay up before its ending counts as a new event
# rather than a continuation of the one being backed off from. A socket that
# connects and dies seconds later is not a recovery: resetting the ladder the
# moment ``connect()`` returned meant a flapping feed retried every five
# seconds forever, and the delays only ever grew while connecting was itself
# the thing failing.
HEALTHY_SESSION_SECONDS = 120.0


class FeedStalled(RuntimeError):
    """No market has ticked recently — the session has to be rebuilt."""


def healthy_session_seconds(period: int) -> float:
    """How long a session must last before the reconnect ladder restarts for it.

    One bar, or the floor, whichever is longer. A session that reached a
    boundary did the thing this bot exists to do; anything shorter is the same
    outage still failing, and its delay should keep growing. The floor is what
    makes this hold at a one-minute bar and the period term is what makes it
    hold at a five-minute one, where two minutes is less than a single bar.

    The bar is one period and not several on purpose. This feed drops sessions
    of its own accord every few minutes, and a threshold longer than that gap
    would climb the ladder to its cap and hold it there — a minute of downtime
    after every drop, forever, to defend against a refusal that is not
    happening. Short sessions still climb, so a server that really will not
    have us is still backed off from.
    """
    return max(HEALTHY_SESSION_SECONDS, float(period))


def backoff_step(lived: float, failures: int, backoff: float,
                 healthy: float = HEALTHY_SESSION_SECONDS) -> tuple[int, float, float]:
    """How long to wait after a session that lasted ``lived`` seconds failed.

    Returns ``(failures, delay, backoff)``: the new failure count, the delay to
    sleep before the next attempt, and the delay to start from after that.

    The ladder restarts only for a session that stayed up long enough to be one
    — see ``healthy_session_seconds``. It used to restart the moment
    ``connect()`` returned, which meant a feed that accepted connections and
    then died seconds later retried every five seconds for as long as it kept
    doing it. The delays only ever grew while connecting was itself what
    failed, so the flapping case — the one that actually happened — never
    backed off at all.
    """
    if lived >= healthy:
        failures, backoff = 0, BACKOFF_MIN
    failures += 1
    return failures, backoff, min(backoff * 2, BACKOFF_MAX)


# --------------------------------------------------------------------------
# feed
# --------------------------------------------------------------------------
def build_feed(cfg: Config, broker: DemoBroker | None = None) -> DataFeed:
    if cfg.feed == "pocket_option":
        return PocketOptionFeed(
            cfg.pocket_ssid, cfg.pocket_uid, cfg.pocket_is_demo,
            region=cfg.pocket_region or None, is_fast_history=True,
            # Attached before the socket connects: the server pushes the deal
            # history during authorisation.
            on_client=broker.attach if broker is not None else None)
    return SimulatedFeed(cfg.assets or None)


def build_broker(cfg: Config, clock: Clock | None = None) -> DemoBroker | None:
    """The demo broker, or None when the bot is only signalling.

    Constructing one at all is opt-in. ``AUTO_TRADE=0`` — the default — means no
    broker exists, no orders are ever submitted, and the scheduler behaves
    exactly as it did before this module existed.
    """
    if not cfg.auto_trade:
        return None
    return DemoBroker(amount=cfg.trade_amount, duration=cfg.expiry_seconds,
                      clock=clock)


def resolve_symbols(cfg: Config, metas: dict[str, AssetMeta]) -> list[str]:
    """The markets to trade this session.

    An explicit ``ASSETS`` list is taken as an instruction: those symbols, in
    that order, payout ignored. Otherwise the universe is derived from the live
    asset list — major pairs, OTC only, at or above the payout floor.
    """
    if cfg.assets:
        offered = [a for a in cfg.assets if not metas or a in metas]
        unknown = [a for a in cfg.assets if metas and a not in metas]
        if unknown:
            log.warning("ASSETS names %d symbol(s) the feed does not offer: %s",
                        len(unknown), ", ".join(unknown))
        return offered

    if not metas:
        raise FeedStalled("the feed published no asset list")

    picked = select_assets(metas, cfg.asset_mode, cfg.min_payout, cfg.otc_only)
    if not picked:
        reasons = describe_skipped(metas, cfg.asset_mode, cfg.min_payout, cfg.otc_only)
        raise ConfigError(
            f"no assets match ASSET_MODE={cfg.asset_mode} MIN_PAYOUT={cfg.min_payout} "
            f"(of {len(metas)} offered: " +
            ", ".join(f"{k}={v}" for k, v in reasons.items()) + ")")
    return picked


def resolve_universe(cfg: Config, metas: dict[str, AssetMeta],
                     sticky: list[str]) -> list[str]:
    """The markets for this session, held stable for the life of the process.

    ``select_assets`` reads the *live* payout list, which the server republishes
    on every connect, so resolving per session made the universe churn: 12 to 15
    markets, not the same ones, several times an hour. Every market that dropped
    out stopped receiving ticks, and a market with no ticks accumulates no bars
    — the hole then trips ``MAX_GAP_BARS`` and its series warms up again from
    nothing, which is what truncated the stored series on 2026-09-15 and left a
    signal with no bars left to settle it against.

    So the first resolved universe is kept. A market that later drops off the
    feed simply goes stale and is skipped, exactly as any silent market is,
    while the ones that stay keep every bar they have accumulated. Only the
    membership is sticky; payouts are re-read each session and used fresh for
    ranking, and a symbol the feed has stopped offering is let go rather than
    subscribed to blind.

    ``sticky`` is the caller's list, filled on the first call and reused after.
    """
    if not sticky:
        sticky.extend(resolve_symbols(cfg, metas))
        return list(sticky)

    if not metas:
        return list(sticky)
    kept = [s for s in sticky if s in metas]
    dropped = [s for s in sticky if s not in metas]
    if dropped:
        log.warning("the feed no longer offers %d market(s) held from earlier "
                    "sessions: %s", len(dropped), ", ".join(dropped))
    if not kept:
        # Nothing survived — re-derive rather than trade an empty universe.
        sticky.clear()
        return resolve_universe(cfg, metas, sticky)
    if kept != sticky:
        sticky[:] = kept
    return list(sticky)


# --------------------------------------------------------------------------
# watchdog
# --------------------------------------------------------------------------
async def watch_feed(market: MarketState, symbols: list[str], cfg: Config,
                     controller: BotController, clock: Clock | None = None,
                     feed: DataFeed | None = None) -> None:
    """Raise ``FeedStalled`` when the feed stops working.

    Two tripwires, because they catch different failures at very different
    speeds. A Socket.IO connection can stay open while the server quietly stops
    sending updates — from the inside that looks exactly like a very quiet
    market, and the bot would sit there sending nothing; only silence can
    detect that, so it is waited out. The other failure is a dead transport,
    which the feed knows about immediately, and waiting out the silence window
    for it was most of a flap cycle: on 2026-09-15 engineio aborted the
    connection ~8 seconds after the last tick and the watchdog did not react
    until 95 seconds after it.
    """
    clock = clock or RealClock()
    started = clock.now()
    last_warn = 0.0
    while not controller.stop_requested:
        await clock.sleep_until(clock.now() + WATCH_INTERVAL)
        now = clock.now()
        if feed is not None and not feed.is_connected:
            raise FeedStalled("the feed's connection dropped")
        freshest = min(
            (now - (market.track(s).last_tick_ts or started) for s in symbols),
            default=0.0)
        if freshest > cfg.feed_stale_after:
            raise FeedStalled(
                f"no ticks for {int(freshest)}s on any of {len(symbols)} markets")
        if not market.healthy(now) and now - last_warn > 60.0:
            last_warn = now
            log.warning("no market has a live bar yet (%.0fs since connect)", now - started)


# --------------------------------------------------------------------------
# one connected session
# --------------------------------------------------------------------------
async def trading_session(cfg: Config, feed: DataFeed, sender: TelegramSender | None,
                          stats: StatsTracker, controller: BotController,
                          status_holder: dict, clock: Clock | None = None,
                          broker: DemoBroker | None = None,
                          universe: list[str] | None = None) -> None:
    """Connect, warm up, and trade until something fails or /stop arrives."""
    clock = clock or RealClock()
    metas = await feed.asset_meta()
    if universe is None:
        symbols = resolve_symbols(cfg, metas)
    else:
        symbols = resolve_universe(cfg, metas, universe)
    payouts = {s: metas[s].payout for s in symbols if s in metas}
    log.info("trading %d markets: %s", len(symbols),
             format_universe(symbols, metas) or ", ".join(symbols))
    log.info("payout range in universe: %d%%-%d%%",
             min(payouts.values(), default=0), max(payouts.values(), default=0))

    market = MarketState(cfg.candle_period, cfg.max_bars, cfg.stale_after, clock,
                         max_gap_bars=cfg.max_gap_bars)
    # Register the handler before subscribing so no tick is missed.
    feed.set_tick_handler(market.on_tick)

    store = None
    if cfg.persist_candles:
        # The feed is part of the store's identity: bars from the simulator are
        # a fabricated price path and must never be restored as live history.
        store = CandleStore(cfg.candle_store_dir, cfg.candle_period, cfg.max_bars,
                            feed=cfg.feed)
        restored = 0
        for symbol in symbols:
            candles = store.load(symbol)
            if len(candles) >= 2:
                market.track(symbol).restore(candles)
                restored += 1
        if restored:
            log.info("restored %d market(s) from %s — warm start",
                     restored, cfg.candle_store_dir)

    await feed.subscribe(symbols)

    scheduler = MinuteScheduler(
        market=market, store=store, sender=sender, stats=stats,
        controller=controller, cadence=cfg.cadence(), engine_config=cfg.signal,
        symbols=symbols, payouts=payouts, mode_label=cfg.asset_mode,
        min_payout=cfg.min_payout, clock=clock,
        journal=SignalJournal(cfg.signal_journal_path, cfg.journal_signals),
        broker=broker)
    status_holder["scheduler"] = scheduler

    tasks = [
        asyncio.create_task(scheduler.run(), name="scheduler"),
        asyncio.create_task(watch_feed(market, symbols, cfg, controller, clock,
                                       feed=feed),
                            name="watchdog"),
    ]
    if isinstance(feed, SimulatedFeed):
        tasks.append(asyncio.create_task(feed.run(), name="sim-feed"))

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        if not controller.stop_requested:
            finished = ", ".join(sorted(t.get_name() for t in done))
            raise FeedStalled(f"{finished} stopped unexpectedly")
    finally:
        status_holder["scheduler"] = None
        if broker is not None:
            # No order may outlive the session that placed it.
            try:
                await broker.shutdown()
            except Exception as exc:
                log.warning("could not stop the broker cleanly: %s", exc)
        # Flush the candles now rather than leaving the last (debounced) writes
        # unwritten: broker history is not usable for warm-up, so the bars this
        # session watched are the only history the next session can have.
        if store is not None:
            try:
                store.save_all({s: market.track(s) for s in symbols}, force=True)
                log.info("flushed candles for %d markets", len(symbols))
            except Exception as exc:
                log.warning("final candle flush failed: %s", exc)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------------
# wake lock
# --------------------------------------------------------------------------
class WakeLock:
    """Ask the host to keep running for as long as this process does.

    A laptop left alone sleeps on its own timer, and entering Modern Standby
    freezes Win32 processes: the socket dies while the process still looks
    alive, so nothing in the reconnect loop ever sees a failure to react to.
    The power plan can prevent that machine-wide, but a power plan is a
    property of the machine — a Windows update, a different profile or simply
    unplugging it can undo the setting with nothing here noticing. This asks on
    the bot's own behalf instead, and because the request belongs to the
    calling thread it lapses by itself when the process exits.

    ``ES_SYSTEM_REQUIRED`` is the part that resets the idle timer;
    ``ES_CONTINUOUS`` makes it stand rather than expire after a single call.
    Deliberately not ``ES_DISPLAY_REQUIRED`` — the screen may switch itself
    off, which saves power and costs the bot nothing.

    A no-op where the API does not exist, so callers need not branch on
    platform. Note it cannot stop a *user-initiated* transition (closing the
    lid) or the low-battery hibernate, and nothing can.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._set = None
        if sys.platform == "win32":
            import ctypes
            try:
                self._set = ctypes.windll.kernel32.SetThreadExecutionState
                self._set.restype = ctypes.c_ulong
                self._set.argtypes = [ctypes.c_ulong]
            except (AttributeError, OSError):   # pragma: no cover - exotic host
                self._set = None

    @property
    def supported(self) -> bool:
        return self._set is not None

    def acquire(self) -> bool:
        """Take the lock. False if the host refused it or cannot offer one."""
        if self._set is None:
            return False
        return bool(self._set(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED))

    def release(self) -> None:
        """Drop the lock. ``ES_CONTINUOUS`` alone is how the API clears it."""
        if self._set is not None:
            self._set(self.ES_CONTINUOUS)


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
async def run(cfg: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s")

    sender = None
    if cfg.telegram_enabled:
        sender = TelegramSender(cfg.telegram_bot_token, cfg.telegram_chat_id)
    else:
        log.warning("Telegram not configured — signals will only be logged")

    # One clock for messages, stats and the journal: a trade placed by hand has
    # to land on the second the message names, so the two must not disagree.
    configure_clock(cfg.market_tz_offset_hours)

    stats = StatsTracker(cfg.stats_path, cfg.legacy_stats_path,
                         tz_offset_hours=cfg.market_tz_offset_hours)
    if stats.total:
        log.info("record so far: %dW/%dL = %.0f%%", stats.wins, stats.losses,
                 stats.win_rate * 100)
    if stats.legacy.total:
        log.info("plus %dW/%dL held over from %s, kept out of that figure",
                 stats.legacy.wins, stats.legacy.losses, cfg.legacy_stats_path)

    controller = BotController()
    status_holder: dict = {"scheduler": None}
    broker = build_broker(cfg)

    # Taken before the first connect: a machine that sleeps mid-session is the
    # one failure the reconnect loop cannot see, because the process is frozen
    # rather than broken.
    awake = WakeLock()
    if awake.supported:
        if awake.acquire():
            log.info("holding the system awake while this runs")
        else:
            log.warning("could not take a wake lock — if this machine sleeps, "
                        "the bot will stall with the process still up")

    def status_text() -> str:
        scheduler = status_holder.get("scheduler")
        return scheduler.status_text() if scheduler else "📡 Connecting…"

    control = None
    if sender is not None:
        control = TelegramControl(sender.bot_token, sender.chat_id, controller,
                                  send=sender.send, status=status_text)
        control.attach_stats(stats)
        await control.start()

    failures = 0
    backoff = BACKOFF_MIN
    # The universe is resolved once and then held: see ``resolve_universe`` for
    # why re-deriving it per reconnect costs bars.
    universe: list[str] = []
    healthy_after = healthy_session_seconds(cfg.candle_period)
    # A refused account is a standing condition, not an event: say so once
    # rather than on every reconnect.
    refusal_announced = False
    try:
        while not controller.stop_requested:
            feed = None
            started = time.time()
            try:
                feed = build_feed(cfg, broker)
                await feed.connect()
                if broker is not None:
                    # Only now does the server's own answer about the account
                    # exist. This is the check that cannot be fooled by .env.
                    allowed, reason = await broker.verify()
                    if not allowed:
                        log.error("running signal-only: %s", reason)
                        if sender is not None and not refusal_announced:
                            refusal_announced = True
                            try:
                                await sender.send_text(f"⚠️ {reason}\n"
                                                       "Signals only — no orders.")
                            except Exception:
                                pass
                await trading_session(cfg, feed, sender, stats, controller,
                                      status_holder, broker=broker,
                                      universe=universe)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                lived = time.time() - started
                failures, delay, backoff = backoff_step(lived, failures, backoff,
                                                        healthy_after)
                log.warning("session failed after %.0fs (%s: %s) — reconnecting in %.0fs",
                            lived, type(exc).__name__, exc, delay)
                # Tell the operator, but do not turn an outage into a flood:
                # first failure, then once every ten attempts.
                if sender is not None and (failures == 1 or failures % 10 == 0):
                    try:
                        await sender.send_text(
                            f"⚠️ {type(exc).__name__}: {exc}\n"
                            f"After {lived:.0f}s up. "
                            f"Reconnecting in {delay:.0f}s (attempt {failures}).")
                    except Exception:
                        pass
                await asyncio.sleep(delay)
            finally:
                if feed is not None:
                    try:
                        await feed.close()
                    except Exception:
                        pass
    finally:
        if control is not None:
            await control.stop()
        if broker is not None:
            await broker.shutdown()
        stats.save(force=True)
        if controller.stop_requested:
            log.info("stopped by command")
            if sender is not None:
                try:
                    await sender.send_text("🛑 Bot stopped.")
                except Exception:
                    pass
        if sender is not None:
            await sender.close()
        # Released last, so the machine stays awake through the shutdown work
        # (the stats save and the candle flush) rather than sleeping over it.
        awake.release()


def main() -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        log.error("configuration error: %s", exc)
        raise SystemExit(2) from None
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
