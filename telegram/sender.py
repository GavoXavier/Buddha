"""Telegram Bot API sender (async, aiohttp) and message formatting.

Messages are sent with ``parse_mode=HTML``, so every string built here must
avoid raw ``<``, ``>`` and ``&`` — the only markup used is ``<b>``/``<code>``.

The signal message is a *heads-up*: it names the exact second the trade starts
(the next bar boundary) and counts down to it, so the trade can be placed on the
boundary rather than whenever the message happens to be read. Times are printed
in the configured zone (``MARKET_TZ_OFFSET_HOURS``), because a trade placed by
hand depends on that clock matching the one in front of the reader.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import time
from typing import Optional

import aiohttp

from signals.engine import Readiness, Signal

log = logging.getLogger("pocket.telegram")

_CALL_EMOJI = "\U0001F7E2"  # 🟢
_PUT_EMOJI = "\U0001F534"   # 🔴

# Emoji and wording per outcome. A refund is neither a win nor a loss, and an
# order that never reached the broker is not a loss either — neither may be
# allowed to render as one.
_RESULT_STYLE = {
    "WIN": ("✅", "WIN"),
    "LOSS": ("❌", "LOSS"),
    "PUSH": ("↩️", "REFUNDED"),
    "UNPLACED": ("⚠️", "NOT PLACED"),
}

# The clock every rendered time is printed in. Was a constant for years; it is
# a setting because the entry time in a signal is the second a hand-placed trade
# has to land on. Kenya is East Africa Time (UTC+3) year-round — no DST — which
# is the default. ``main()`` calls ``configure_clock`` from the config.
MARKET_TZ_OFFSET_HOURS = 3.0
MARKET_TZ = _dt.timezone(_dt.timedelta(hours=MARKET_TZ_OFFSET_HOURS))


def configure_clock(offset_hours: float) -> None:
    """Point every rendered clock at ``offset_hours``.

    Rebinds the module globals, so callers that did ``from ... import MARKET_TZ``
    hold a stale zone — that is why the loops format through ``clock()`` instead
    of importing the timezone.

    Fixed-offset zones only: this is a display setting for one reader, not a
    timezone database, so DST is not tracked.
    """
    global MARKET_TZ_OFFSET_HOURS, MARKET_TZ
    MARKET_TZ_OFFSET_HOURS = float(offset_hours)
    MARKET_TZ = _dt.timezone(_dt.timedelta(hours=MARKET_TZ_OFFSET_HOURS))

_NUM_EMOJI = {
    1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣",
    6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟",
}


def parse_expiry_minutes(expiry: str) -> int:
    """Parse an expiry string like '1m', '5m', '1h' into whole minutes."""
    expiry = expiry.strip().lower()
    if not expiry:
        return 1
    if expiry.endswith("m"):
        return int(expiry[:-1])
    if expiry.endswith("h"):
        return int(expiry[:-1]) * 60
    if expiry.endswith("s"):
        return max(1, int(expiry[:-1]) // 60)
    return int(expiry)


def expiry_seconds(expiry: str) -> int:
    """Parse an expiry string into seconds ('1m' -> 60, '90s' -> 90)."""
    expiry = expiry.strip().lower()
    if not expiry:
        return 60
    if expiry.endswith("m"):
        return int(expiry[:-1]) * 60
    if expiry.endswith("h"):
        return int(expiry[:-1]) * 3600
    if expiry.endswith("s"):
        return int(expiry[:-1])
    return int(expiry) * 60


def _format_expiry(minutes: int) -> str:
    return "1 minute" if minutes == 1 else f"{minutes} minutes"


def format_duration(seconds: int) -> str:
    """A bar or expiry length as a short label: 60 -> '1m', 300 -> '5m', 45 -> '45s'."""
    if seconds and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def clock(at: float) -> str:
    """Wall clock in the configured zone — the time printed in every message.

    Public and late-binding on purpose: it reads ``MARKET_TZ`` when called, so a
    ``configure_clock`` at startup reaches every caller, including the ones that
    imported this function long before.
    """
    return _dt.datetime.fromtimestamp(at, tz=MARKET_TZ).strftime("%H:%M:%S")


def _countdown(seconds: int) -> str:
    """Seconds as a readable wait: '5m', '4m 30s', '45s'.

    A full-bar lead is five minutes of waiting, and '(in 300s)' makes the reader
    do arithmetic to find out whether that is minutes or nearly an hour.
    """
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}m" if rest == 0 else f"{minutes}m {rest}s"


def _format_asset(asset: str) -> str:
    """'EURUSD_otc' -> 'EUR/USD OTC'; '#AAPL_otc' -> '#AAPL OTC'."""
    is_otc = asset.endswith("_otc")
    base = asset[:-4] if is_otc else asset
    if len(base) == 6 and base[:3].isalpha() and base[3:].isalpha():
        base = f"{base[:3]}/{base[3:]}"
    return f"{base} OTC" if is_otc else base


def _num_emoji(n: int) -> str:
    return _NUM_EMOJI.get(n, str(n))


def format_signal(signal: Signal, asset: str, expiry: str, entry_at: float,
                  now: Optional[float] = None, payout: int = 0,
                  martingale_steps: int = 0,
                  entry_price: Optional[float] = None) -> str:
    """Render a signal as a Telegram message.

    ``entry_at`` is the boundary the trade must be placed on (epoch seconds);
    ``now`` defaults to the current time and drives the countdown. The expiry is
    named as a clock time as well as a length, so the whole trade reads as three
    fixed moments: sent now, in at ``entry_at``, out at ``entry_at + expiry``.
    """
    now = time.time() if now is None else now
    emoji = _CALL_EMOJI if signal.direction == "CALL" else _PUT_EMOJI
    direction = "BUY" if signal.direction == "CALL" else "SELL"
    minutes = parse_expiry_minutes(expiry)
    seconds_left = max(0, int(round(entry_at - now)))
    closes_at = entry_at + expiry_seconds(expiry)

    lines = [
        f"🪙 <b>{_format_asset(asset)}</b>",
        f"⏳ Expiration {_format_expiry(minutes)} "
        f"(closes {clock(closes_at)})",
    ]
    if seconds_left > 0:
        lines.append(f"✅ Entry at <b>{clock(entry_at)}</b> "
                     f"(in {_countdown(seconds_left)})")
    else:
        lines.append(f"✅ Entry now ({clock(entry_at)})")
    lines.append(f"{emoji} <b>{direction}</b>")

    detail = [f"score {signal.score}", f"{signal.confidence:.0%} conf"]
    if payout:
        detail.append(f"payout {payout}%")
    lines.append("📊 " + " · ".join(detail))
    if signal.votes:
        lines.append("🔎 " + ", ".join(signal.votes))
    if entry_price:
        # This is the close of the bar the decision was made on, not the price
        # the trade will open at. At a 10-second lead the two are near enough;
        # at a full-bar lead they are five minutes apart, and calling it "Price"
        # invites the reader to place at a number that has moved.
        if seconds_left > 0:
            lines.append(f"💵 Price {entry_price} (at signal — you enter at "
                         f"{clock(entry_at)})")
        else:
            lines.append(f"💵 Price {entry_price}")

    if martingale_steps > 0:
        lines += ["", "Use martingale if necessary 👇", ""]
        for i in range(1, martingale_steps + 1):
            t = entry_at + i * minutes * 60
            lines.append(f"{_num_emoji(i)} MARTINGALE AT {clock(t)}")
    return "\n".join(lines)


def format_confirmation(asset: str, direction: str, outcome: str,
                        wins: int, losses: int, win_rate: float,
                        expiry_at: Optional[float] = None,
                        streak: str = "") -> str:
    """Render the result of a completed trade.

    A refund and a failed order are neither wins nor losses, so they get their
    own mark and a note: the line below reports the tally, and a refund is not
    in it. Anything else would print a refund as ❌ LOSS.
    """
    emoji = _CALL_EMOJI if direction == "CALL" else _PUT_EMOJI
    label = "BUY" if direction == "CALL" else "SELL"
    result_emoji, result_name = _RESULT_STYLE.get(outcome, ("❔", outcome))
    lines = [
        f"🪙 {_format_asset(asset)}",
        f"{emoji} {label}",
    ]
    if expiry_at:
        lines.append(f"🏁 Expired {clock(expiry_at)}")
    lines.append(f"{result_emoji} <b>{result_name}</b>")
    lines.append(f"📊 Win rate: {win_rate:.0%} ({wins}W / {losses}L)")
    if outcome in ("PUSH", "UNPLACED"):
        lines.append("(not counted in the win rate)")
    if streak:
        lines.append(f"🔥 {streak}")
    return "\n".join(lines)


def format_warmup(readiness: Readiness, eta_minutes: Optional[int] = None,
                  available: str = "") -> str:
    """Progress note while the buffer fills: what works now, what is still missing."""
    lines = [
        "⏳ <b>Warming up</b>",
        f"Bars: {readiness.bars}/{readiness.bars_needed}",
    ]
    if available:
        lines.append(f"Live indicators: {available}")
    if readiness.missing:
        lines.append("Waiting on: " + ", ".join(readiness.missing))
    if eta_minutes is not None and eta_minutes > 0:
        lines.append(f"Full readiness in ~{eta_minutes} min")
    else:
        lines.append("Fully warmed up — signals active.")
    return "\n".join(lines)


def format_startup(assets: int, period: int, expiry: str, mode: str,
                   min_payout: int, bars_restored: int,
                   execution: str = "") -> str:
    lines = [
        "🤖 <b>Pocket Option signal bot started</b>",
        f"Markets: {assets} ({mode}, payout ≥ {min_payout}%)",
        f"Cadence: one signal per {format_duration(period)} bar · expiry {expiry}",
    ]
    if bars_restored:
        lines.append(f"Restored {bars_restored} bars from disk")
    if execution:
        # Stated on every start, because "is this thing trading my account?" is
        # not a question that should need a config file to answer.
        lines.append(f"⚙️ {execution}")
    return "\n".join(lines)


def format_status(*, running: bool, assets: int, healthy: int, bars: int,
                  bars_needed: int, next_entry_at: Optional[float],
                  now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    lines = [
        "📡 <b>Status</b>",
        f"Bot: {'running' if running else 'paused'}",
        f"Markets: {healthy}/{assets} live",
        f"Bars: {bars}/{bars_needed}",
    ]
    if next_entry_at:
        left = max(0, int(round(next_entry_at - now)))
        lines.append(f"Next entry: {clock(next_entry_at)} (in {_countdown(left)})")
    return "\n".join(lines)


class TelegramSender:
    """Sends messages to one Telegram chat via the Bot API."""

    def __init__(self, bot_token: str, chat_id: str,
                 session: Optional[aiohttp.ClientSession] = None) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._owns_session = session is None
        if session is not None:
            self._session = session
        else:
            # Threaded resolver: the default aiodns (c-ares) can fail to contact
            # DNS servers on some Windows setups.
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            self._session = aiohttp.ClientSession(connector=connector)
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def send(self, text: str, retries: int = 2) -> dict:
        """Post a message, retrying once on rate-limit or transient failure."""
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        timeout = aiohttp.ClientTimeout(total=20)
        for attempt in range(retries + 1):
            try:
                async with self._session.post(url, json=payload, timeout=timeout) as resp:
                    data = await resp.json(content_type=None)
            except Exception as exc:  # network hiccup - retry
                self._last_error = f"{type(exc).__name__}: {exc}"
                if attempt == retries:
                    log.error("telegram send failed: %s", self._last_error)
                    return {"ok": False, "description": self._last_error}
                await asyncio.sleep(1.0 + attempt)
                continue

            if data.get("ok"):
                self._last_error = None
                return data
            self._last_error = str(data.get("description", data))
            retry_after = (data.get("parameters") or {}).get("retry_after")
            if retry_after and attempt < retries:
                await asyncio.sleep(float(retry_after) + 0.5)
                continue
            if attempt == retries:
                log.warning("telegram rejected message: %s", self._last_error)
            else:
                await asyncio.sleep(1.0 + attempt)
        return {"ok": False, "description": self._last_error}

    async def close(self) -> None:
        if self._owns_session:
            await self._session.close()

    # -- typed helpers -------------------------------------------------------
    async def send_signal(self, signal: Signal, asset: str, expiry: str,
                          entry_at: float, now: Optional[float] = None,
                          payout: int = 0, martingale_steps: int = 0,
                          entry_price: Optional[float] = None) -> dict:
        return await self.send(format_signal(
            signal, asset, expiry, entry_at, now=now, payout=payout,
            martingale_steps=martingale_steps, entry_price=entry_price))

    async def send_confirmation(self, asset: str, direction: str, outcome: str,
                                wins: int, losses: int, win_rate: float,
                                expiry_at: Optional[float] = None,
                                streak: str = "") -> dict:
        return await self.send(format_confirmation(
            asset, direction, outcome, wins, losses, win_rate,
            expiry_at=expiry_at, streak=streak))

    async def send_text(self, text: str) -> dict:
        return await self.send(text)
