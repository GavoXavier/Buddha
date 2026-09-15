"""Tests for the wake lock the supervisor holds while the bot runs.

The lock is the only thing standing between "the bot is running" and "the
machine slept and the socket died with the process still up". It is also the
one piece of the supervisor that touches the host directly, so what it asks for
is worth pinning precisely: the wrong flag keeps the screen lit, and a
silently-refused request leaves the bot asleep at the wheel with nobody told.
"""

import sys
import unittest

from main import WakeLock

# Documented in SetThreadExecutionState's remarks; not named here because the
# point of the flag is that we never pass it.
_ES_DISPLAY_REQUIRED = 0x00000002


class FakeExecutionState:
    """Stands in for kernel32.SetThreadExecutionState."""

    def __init__(self, result: int = 1):
        self.calls: list[int] = []
        self.result = result
        self.restype = None
        self.argtypes = None

    def __call__(self, flags: int) -> int:
        self.calls.append(flags)
        return self.result


def with_fake_api() -> tuple[WakeLock, FakeExecutionState]:
    """A WakeLock wired to a fake API, plus the fake to inspect.

    ``_set`` is the single seam between this class and the host, so replacing
    it is the whole substitution — there is no behaviour left unexercised.
    """
    lock = WakeLock()
    fake = FakeExecutionState()
    lock._set = fake
    return lock, fake


class TestWakeLock(unittest.TestCase):
    def test_it_asks_for_the_system_and_not_the_display(self):
        lock, fake = with_fake_api()

        self.assertTrue(lock.acquire())

        self.assertEqual(fake.calls,
                         [WakeLock.ES_CONTINUOUS | WakeLock.ES_SYSTEM_REQUIRED])
        self.assertEqual(fake.calls[0] & _ES_DISPLAY_REQUIRED, 0,
                         "the screen is allowed to sleep; only the system may not")

    def test_the_request_is_continuous_rather_than_one_shot(self):
        # Without ES_CONTINUOUS the lock expires immediately and the machine
        # sleeps on its usual timer, which would look like it worked.
        lock, fake = with_fake_api()
        lock.acquire()

        self.assertEqual(fake.calls[0] & WakeLock.ES_CONTINUOUS, WakeLock.ES_CONTINUOUS)

    def test_a_refused_request_is_reported_not_assumed(self):
        # The API returns 0 on failure. A bot that believes it holds a lock it
        # was refused is worse off than one that says so and gets watched.
        lock, fake = with_fake_api()
        fake.result = 0

        self.assertFalse(lock.acquire())

    def test_release_drops_the_request(self):
        # ES_CONTINUOUS on its own is the documented way to clear it; passing
        # the system-required flag again would re-take the lock.
        lock, fake = with_fake_api()
        lock.acquire()
        lock.release()

        self.assertEqual(fake.calls[-1], WakeLock.ES_CONTINUOUS)
        self.assertEqual(fake.calls[-1] & WakeLock.ES_SYSTEM_REQUIRED, 0)

    def test_without_the_api_it_is_a_no_op(self):
        lock = WakeLock()
        lock._set = None

        self.assertFalse(lock.supported)
        self.assertFalse(lock.acquire(), "no API is not a refusal, but it is not a lock")
        lock.release()          # must not raise

    def test_supported_tracks_whether_the_api_is_there(self):
        lock, _ = with_fake_api()
        self.assertTrue(lock.supported)

    def test_the_flag_values_are_the_documented_ones(self):
        self.assertEqual(WakeLock.ES_CONTINUOUS, 0x80000000)
        self.assertEqual(WakeLock.ES_SYSTEM_REQUIRED, 0x00000001)


class TestWakeLockOnWindows(unittest.TestCase):
    """The ctypes lookup itself — the part that fails silently if miswired."""

    @unittest.skipUnless(sys.platform == "win32", "the API is Windows-only")
    def test_the_real_api_is_found(self):
        self.assertTrue(WakeLock().supported,
                        "kernel32.SetThreadExecutionState was not resolved")

    @unittest.skipUnless(sys.platform == "win32", "the API is Windows-only")
    def test_the_real_api_grants_and_clears_the_lock(self):
        # A wrong restype or argtypes makes the call fail rather than raise, so
        # the only honest check is to actually take the lock and give it back.
        lock = WakeLock()
        self.assertTrue(lock.acquire(), "the host refused the wake lock")
        lock.release()


if __name__ == "__main__":
    unittest.main()
