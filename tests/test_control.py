"""The command poller, driven without a network.

``_handle`` is the whole of the decision — who may command, which commands exist,
and which messages are history. Everything above it is transport, so everything
worth testing is reachable by handing it a dict.

The case that matters most is the stale one. Telegram keeps updates for 24 hours
and the drain that skips the backlog is a single ``getUpdates`` that can fail, so
"a command from before the bot started is never obeyed" has to hold without it.
"""

import asyncio
import time
import unittest

from telegram.control import BotController, TelegramControl


class FakeSender:
    def __init__(self):
        self.sent = []

    async def __call__(self, text: str) -> dict:
        self.sent.append(text)
        return {"ok": True}


def update(text: str, sent_at: float, chat_id: str = "42",
           update_id: int = 1) -> dict:
    return {"update_id": update_id,
            "message": {"date": int(sent_at), "text": text,
                        "chat": {"id": chat_id}}}


class ControlCase(unittest.TestCase):
    def setUp(self):
        self.sender = FakeSender()
        self.controller = BotController()
        self.control = TelegramControl(
            "token", "42", self.controller, send=self.sender, status=lambda: "STATUS")

    def handle(self, upd: dict) -> None:
        asyncio.run(self.control._handle(upd))


class TestOnlyTheConfiguredChatIsObeyed(ControlCase):
    def test_a_stranger_cannot_stop_the_bot(self):
        # The token is a shared secret and a guessed chat id must not be enough to
        # pause trading or read the record.
        self.handle(update("/stop", time.time(), chat_id="999"))

        self.assertFalse(self.controller.stop_requested)
        self.assertEqual(self.sender.sent, [])

    def test_the_configured_chat_can(self):
        self.handle(update("/stop", time.time()))

        self.assertTrue(self.controller.stop_requested)
        self.assertEqual(self.sender.sent, ["🛑 Shutting down."])


class TestACommandFromBeforeThisSessionIsNeverObeyed(ControlCase):
    """The offset skips the backlog when it can; this holds even when it cannot."""

    def test_a_command_from_yesterday_does_not_stop_the_bot(self):
        self.handle(update("/stop", time.time() - 24 * 3600))

        self.assertFalse(self.controller.stop_requested,
                         "Telegram serves 24h of updates, and yesterday's /stop is "
                         "still in there")
        self.assertEqual(self.sender.sent, [])

    def test_a_command_sent_while_the_bot_was_off_is_ignored(self):
        # Ten minutes before startup: queued, not intended for this session.
        self.handle(update("/pause", time.time() - 600))

        self.assertFalse(self.controller.paused)

    def test_a_command_sent_moments_before_startup_still_lands(self):
        # ``message.date`` is Telegram's clock, not ours, so a few seconds of skew
        # must not silently disable remote control.
        self.handle(update("/pause", time.time() - 5))

        self.assertTrue(self.controller.paused)

    def test_a_message_with_no_readable_date_is_refused_not_guessed(self):
        self.handle({"update_id": 7, "message": {"text": "/stop",
                                                 "chat": {"id": "42"}}})

        self.assertFalse(self.controller.stop_requested,
                         "failing closed keeps the bot trading and the log says why")

    def test_a_plain_message_is_not_a_command(self):
        self.handle(update("good morning", time.time()))

        self.assertEqual(self.sender.sent, [])


class TestTheCommandsThemselves(ControlCase):
    def test_status_is_answered_from_the_callback(self):
        self.handle(update("/status", time.time()))

        self.assertEqual(self.sender.sent, ["STATUS"])

    def test_a_command_addressed_to_the_bot_is_the_same_command(self):
        # Telegram appends @botname when a group has several bots in it.
        self.handle(update("/pause@my_bot", time.time()))

        self.assertTrue(self.controller.paused)

    def test_an_unknown_command_is_answered_with_nothing(self):
        self.handle(update("/nonsense", time.time()))

        self.assertEqual(self.sender.sent, [])

    def test_help_lists_the_commands(self):
        self.handle(update("/help", time.time()))

        self.assertIn("/status", self.sender.sent[0])


class TestStalenessIsMeasuredFromConstruction(ControlCase):
    def test_the_cutoff_is_the_moment_the_poller_was_built(self):
        # Constructed now, so a message sent a minute before it is stale and one
        # sent now is not — the boundary is the object's own creation time.
        self.assertAlmostEqual(self.control._started_at, time.time(), delta=5.0)

        before = self.control._started_at - 61
        after = self.control._started_at + 1
        self.assertTrue(self.control._is_stale({"date": before}))
        self.assertFalse(self.control._is_stale({"date": after}))


if __name__ == "__main__":
    unittest.main()
