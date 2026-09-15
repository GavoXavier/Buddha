"""Tests for the live feed's connection state.

``is_connected`` is what lets the supervisor notice a dropped socket at once
instead of waiting out the tick-silence window. Getting it wrong in either
direction is costly in a way that is hard to see: too eager and the bot
reconnects sessions that were working, too lax and it spends most of every flap
cycle in a session that cannot deliver a tick.

The client is also asked for with its own reconnect switched off, since a
second reconnector for the same seat is not a safety net: it is a race for one
session that the supervisor then has to lose.
"""

import asyncio
import unittest
from unittest import mock

from data import SimulatedFeed
from data.base import DataFeed
from data.pocket_option import PocketOptionFeed


class FakeTask:
    """A background loop task, as far as ``done()`` is concerned.

    Not an ``asyncio.Task``: ``done()`` is the entire interface the feed reads,
    and a real task would tie the test to an event loop it does not need.
    """

    def __init__(self, done=False):
        self._done = done

    def done(self):
        return self._done


class FakeEngineIO:
    def __init__(self, state="connected", write_done=False, read_done=False,
                 write_loop_task="present", read_loop_task="present"):
        self.state = state
        # "present" builds a live task; None stands for a cleared attribute.
        self.write_loop_task = None if write_loop_task is None else FakeTask(write_done)
        self.read_loop_task = None if read_loop_task is None else FakeTask(read_done)


class FakeSocketIO:
    def __init__(self, connected=True, eio_state="connected"):
        self.connected = connected
        self.eio = FakeEngineIO(eio_state)


class FakeClient:
    def __init__(self, sio):
        self.sio = sio


def feed_with(client) -> PocketOptionFeed:
    feed = PocketOptionFeed("session", 1, is_demo=True)
    feed._client = client
    return feed


class TestPocketOptionConnectionState(unittest.TestCase):
    def test_a_disconnected_client_is_not_connected(self):
        self.assertFalse(feed_with(None).is_connected,
                         "no client at all is certainly not connected")

    def test_a_live_socket_is_connected(self):
        self.assertTrue(feed_with(FakeClient(FakeSocketIO())).is_connected)

    def test_a_dead_engine_io_socket_is_not_connected(self):
        # The transport dies first; the namespaces above it are torn down
        # afterwards, so for a while ``connected`` still says True.
        client = FakeClient(FakeSocketIO(connected=True, eio_state="disconnected"))
        self.assertFalse(feed_with(client).is_connected)

    def test_dead_namespaces_are_not_connected(self):
        client = FakeClient(FakeSocketIO(connected=False))
        self.assertFalse(feed_with(client).is_connected)

    def test_a_dead_write_loop_is_not_connected(self):
        # The case that was missed live on 2026-09-15: at 21:46:01 the write
        # loop gave up after thirty seconds with nothing to send, and the
        # socket stayed open and "connected" while nothing could go over it.
        # The tick watchdog did catch it at 21:47:01, but only after 93 seconds
        # of silence — and the ping that would have kept the socket alive was
        # one of the things that could not be sent.
        client = FakeClient(FakeSocketIO(eio_state="connected"))
        client.sio.eio.write_loop_task = FakeTask(done=True)

        self.assertFalse(feed_with(client).is_connected,
                         "a transport that cannot send is not a connection")

    def test_a_dead_read_loop_is_not_connected(self):
        client = FakeClient(FakeSocketIO(eio_state="connected"))
        client.sio.eio.read_loop_task = FakeTask(done=True)

        self.assertFalse(feed_with(client).is_connected)

    def test_a_cleared_loop_task_is_not_evidence_of_death(self):
        # Engine.IO sets both attributes to None when it resets, so an absent
        # task says nothing either way; only a finished one does.
        client = FakeClient(FakeSocketIO(eio_state="connected"))
        client.sio.eio.write_loop_task = None
        client.sio.eio.read_loop_task = None

        self.assertTrue(feed_with(client).is_connected)

    def test_a_finished_task_on_a_disconnected_socket_still_reads_dead(self):
        # The two checks must not cancel each other out, whichever runs first.
        client = FakeClient(FakeSocketIO(eio_state="disconnected"))
        client.sio.eio.write_loop_task = FakeTask(done=True)

        self.assertFalse(feed_with(client).is_connected)

    def test_an_unexpected_sdk_shape_fails_open(self):
        # Whatever the SDK looks like tomorrow, a shape this code does not
        # recognise must not start reconnecting sessions that are working. The
        # tick watchdog is still there underneath.
        class Odd:
            pass

        self.assertTrue(feed_with(Odd()).is_connected)
        self.assertTrue(feed_with(FakeClient(Odd())).is_connected)

    def test_the_simulator_is_always_connected(self):
        # It has no transport to lose, so the watchdog's connection tripwire
        # must never fire for it.
        self.assertTrue(SimulatedFeed(["EURUSD_otc"]).is_connected)
        self.assertTrue(DataFeed.is_connected.fget(SimulatedFeed(["X"])))


class Recorder:
    """The SDK's ``on`` registry, reduced to "remember what was subscribed"."""

    def __init__(self):
        self.handlers = {}

    def __getattr__(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler
        return register


class FakeSDKClient:
    """A ``PocketOptionClient`` that only remembers how it was asked for."""

    instances: list["FakeSDKClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.on = Recorder()
        self.url = None
        self.disconnected = False
        FakeSDKClient.instances.append(self)

    async def connect(self, url, **kwargs):
        self.url = url

    async def wait_for_authorization(self, timeout=None):
        return True

    async def disconnect(self):
        self.disconnected = True


class TestTheSdkDoesNotReconnectOnItsOwn(unittest.TestCase):
    """The supervisor is the only thing that reconnects.

    Left on, the SDK's reconnect rebuilt a dropped socket by itself — a fresh
    session on the same SSID — while the watchdog was tearing that session down
    and building another. For a moment the server held two sessions for one
    account, and the one just built was the one it refused: observed as
    "ConnectionError: One or more namespaces failed to connect" at 21:47:20 on
    2026-09-15, immediately after a flap.
    """

    def setUp(self):
        FakeSDKClient.instances.clear()
        # Both are imported inside PocketOptionFeed.connect, so patching them
        # on the module they are imported from is enough.
        for target, replacement in (
            ("pocket_option.PocketOptionClient", FakeSDKClient),
            ("pocket_option.contrib.default_init.default_init",
             lambda *args, **kwargs: None),
        ):
            patch = mock.patch(target, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def connect(self) -> FakeSDKClient:
        async def scenario():
            feed = PocketOptionFeed("session", 42, is_demo=True)
            await feed.connect()
            client = FakeSDKClient.instances[-1]
            await feed.close()
            return client

        return asyncio.run(scenario())

    def test_the_client_is_built_with_reconnection_off(self):
        self.assertIs(self.connect().kwargs.get("reconnection"), False,
                      "the SDK must not race the supervisor for the account")

    def test_it_still_connects_and_closes(self):
        client = self.connect()

        self.assertTrue(client.url, "the client was never told to connect")
        self.assertTrue(client.disconnected, "close() must drop the socket")


if __name__ == "__main__":
    unittest.main()
