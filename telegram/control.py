"""Telegram command handling, so the bot can be driven from the phone.

Uses long polling (``getUpdates``) against the same bot token the signals are
sent with — no webhook, no extra dependency, no inbound port. Only messages
from the configured chat are obeyed; anything else is ignored, because the bot
token is a shared secret and a stranger who guesses the chat id must not be able
to pause trading or read the stats.

Commands: /status /stats /assets /hours /pause /resume /stop /help.

**A command from before the bot started is never obeyed**, and that is enforced by
the message's own timestamp rather than only by the update offset. Telegram keeps
updates for 24 hours, so a ``/stop`` typed yesterday is still sitting there at
startup; the offset is advanced past the backlog to skip it, but if that first
``getUpdates`` fails — a network blip at exactly the wrong moment — the offset
stays at zero and the whole backlog is delivered. Acting on it would shut the bot
down on a day-old instruction. Comparing ``message.date`` against the process
start cannot be defeated that way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import aiohttp

log = logging.getLogger("pocket.control")

HELP_TEXT = "\n".join([
    "🤖 <b>Commands</b>",
    "/status - bot, feed and warm-up state",
    "/stats - record and win rate",
    "/assets - win rate per market",
    "/hours - win rate per hour",
    "/pause - stop sending signals",
    "/resume - resume signals",
    "/stop - shut the bot down",
])

_POLL_TIMEOUT = 25

# How far before this process started a command may have been sent and still be
# obeyed. Not zero, because ``message.date`` is Telegram's clock and not ours, and
# a few seconds of skew must not silently disable remote control; and small
# enough that "something queued while the bot was off" is still skipped.
_STALE_SLACK = 60.0


class BotController:
    """Shared switchboard between the command poller and the trading loop."""

    def __init__(self) -> None:
        self.paused = False
        self.stop_requested = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def request_stop(self) -> None:
        self.stop_requested = True


class TelegramControl:
    """Long-polls Telegram for commands and applies them to ``controller``."""

    def __init__(self, bot_token: str, chat_id: str, controller: BotController,
                 send: Callable[[str], Awaitable[dict]],
                 status: Callable[[], str]) -> None:
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.chat_id = str(chat_id)
        self.controller = controller
        self._send = send
        self._status = status
        self._offset = 0
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._enabled = True
        # Everything sent before this instant is history, whatever the offset
        # says. See the module docstring for why the offset alone is not enough.
        self._started_at = time.time()

    async def start(self) -> None:
        if self._session is None:
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            self._session = aiohttp.ClientSession(connector=connector)
        # Skip anything queued while the bot was off, so a stale "/pause" from
        # yesterday cannot surprise the operator today. This is the fast path; the
        # timestamp check in ``_handle`` is what makes it safe when it fails.
        await self._drain_backlog()
        self._task = asyncio.create_task(self._run(), name="telegram-control")

    async def _drain_backlog(self) -> None:
        try:
            updates = await self._get_updates(timeout=0)
        except Exception as exc:
            # Not fatal, and not silent: the offset stays where it was, so the
            # next poll re-reads the backlog — which is exactly the case the
            # timestamp check exists to make harmless.
            log.debug("could not drain the command backlog, so old updates will be "
                      "re-read (and dropped by age): %s", exc)
            return
        for update in updates:
            self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)

    async def _get_updates(self, timeout: int) -> list[dict]:
        assert self._session is not None
        url = f"{self.base_url}/getUpdates"
        payload = {"offset": self._offset, "timeout": timeout,
                   "allowed_updates": ["message"]}
        async with self._session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout + 15)) as resp:
            data = await resp.json(content_type=None)
        if not data.get("ok"):
            raise RuntimeError(str(data.get("description", data)))
        return data.get("result", [])

    async def _run(self) -> None:
        while not self.controller.stop_requested:
            try:
                updates = await self._get_updates(timeout=_POLL_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = str(exc)
                if "webhook" in message.lower() or "409" in message:
                    log.error("getUpdates refused (a webhook is set?): %s — "
                              "commands disabled", message)
                    self._enabled = False
                    return
                log.debug("getUpdates failed: %s", message)
                await asyncio.sleep(5)
                continue

            for update in updates:
                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                await self._handle(update)

    def _is_stale(self, message: dict) -> bool:
        """Whether this message predates the process, and must not be obeyed.

        A message with no readable ``date`` is treated as stale, which fails in the
        direction that costs least: the bot keeps trading and the operator can see
        in the log why the commands stopped arriving. Obeying it would mean a
        day-old ``/stop`` could kill a session — the failure the backlog drain
        exists to prevent, so the check must not depend on the drain working.
        """
        try:
            sent = float(message["date"])
        except (KeyError, TypeError, ValueError):
            log.warning("update with no usable date, ignoring the command in it: %r",
                        message.get("text"))
            return True
        return sent < self._started_at - _STALE_SLACK

    async def _handle(self, update: dict) -> None:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("id", "")) != self.chat_id:
            log.warning("ignoring command from unauthorised chat %s", chat.get("id"))
            return
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        if self._is_stale(message):
            log.info("ignoring a command sent before this session started: %r "
                     "(it was queued while the bot was off)", text)
            return
        command = text.split()[0].split("@")[0].lower()
        log.info("command received: %s", command)
        reply = self._dispatch(command)
        if reply:
            await self._send(reply)

    def _dispatch(self, command: str) -> Optional[str]:
        if command in ("/help", "/start"):
            return HELP_TEXT
        if command == "/status":
            return self._status()
        if command == "/stats":
            return self._stats_text()
        if command == "/pause":
            self.controller.pause()
            return "⏸ Paused — no signals until /resume."
        if command == "/resume":
            self.controller.resume()
            return "▶️ Resumed — signals active."
        if command == "/stop":
            self.controller.request_stop()
            return "🛑 Shutting down."
        if command == "/assets":
            return f"<b>Per market</b>\n<code>{self._escape(self._table('asset'))}</code>"
        if command == "/hours":
            return f"<b>Per hour</b>\n<code>{self._escape(self._table('hour'))}</code>"
        return None

    # Tables come from StatsTracker via the status callback's tracker.
    def _table(self, kind: str) -> str:
        tracker = getattr(self, "_tracker", None)
        if tracker is None:
            return "Stats unavailable."
        return tracker.asset_table() if kind == "asset" else tracker.hourly_table()

    def _stats_text(self) -> str:
        tracker = getattr(self, "_tracker", None)
        if tracker is None:
            return "Stats unavailable."
        return "<b>Record</b>\n" + self._escape(tracker.summary_text())

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def attach_stats(self, tracker) -> None:
        """Give the poller access to the stats tracker for /stats and tables."""
        self._tracker = tracker

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
