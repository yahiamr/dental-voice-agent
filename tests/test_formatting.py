"""Tests for spoken values: stored dates and prices must come out as natural words."""

from datetime import datetime

from dental_agent.clock import MADRID
from dental_agent.formatting import spoken_price, spoken_slot


def test_spoken_slot_gives_weekday_date_and_time():
    start = datetime(2026, 11, 10, 10, 0, tzinfo=MADRID)
    assert spoken_slot(start) == "Tuesday 10 November at 10:00"


def test_spoken_slot_drops_the_leading_zero_of_the_day():
    # "Monday 02 November" sounds wrong when read aloud.
    start = datetime(2026, 11, 2, 9, 30, tzinfo=MADRID)
    assert spoken_slot(start) == "Monday 2 November at 09:30"


def test_spoken_price_names_the_currency():
    assert spoken_price(30) == "30 euros"
