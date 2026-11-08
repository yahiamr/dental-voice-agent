"""Tests for the clock: real time uses Madrid, and test time can be moved by hand."""

from datetime import datetime, timedelta

from dental_agent.clock import MADRID, Clock, FixedClock


def test_clock_now_is_in_madrid_time():
    # A change to any other timezone would make every booking rule run at the wrong hour.
    assert Clock().now().tzinfo is MADRID


def test_fixed_clock_returns_the_time_it_was_given():
    start = datetime(2026, 11, 9, 10, 0, tzinfo=MADRID)
    assert FixedClock(start).now() == start


def test_fixed_clock_advance_moves_time_forward():
    # The turn deadline tests depend on being able to move time without waiting.
    start = datetime(2026, 11, 9, 10, 0, tzinfo=MADRID)
    clock = FixedClock(start)
    clock.advance(13)
    assert clock.now() == start + timedelta(seconds=13)
