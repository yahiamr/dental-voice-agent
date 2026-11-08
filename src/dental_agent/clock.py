"""The time source for all code. Stage: 1.

What this file is:
    A small wrapper around the current time, in Europe/Madrid.

What it contains:
    - Clock.now(), which returns the real time.
    - FixedClock, which returns a set time, and advance(seconds), which moves it forward.

How to understand it:
    Booking rules depend on the time, such as the notice period and the booking window. If code
    called datetime.now() directly, tests would pass or fail depending on the day they ran.
    Tests start at Monday 9 November 2026, 10:00, and can move time to test the 12-second
    deadline.

Used by:
    booking/, guards/validate.py, prompts/build.py, graph/nodes.py, session.py
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# The clinic is in Spain, so every date and time in the system uses this timezone.
MADRID = ZoneInfo("Europe/Madrid")


class Clock:
    """The real clock, used in normal runs."""

    def now(self) -> datetime:
        """Return the current time in Madrid."""
        return datetime.now(MADRID)


class FixedClock(Clock):
    """A clock for tests. It shows a set time and moves only when told to."""

    def __init__(self, start: datetime) -> None:
        # The time this clock reports until advance() is called.
        self._now = start

    def now(self) -> datetime:
        """Return the set time."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move time forward, so a test can pass a deadline without waiting."""
        self._now += timedelta(seconds=seconds)
