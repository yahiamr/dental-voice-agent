"""Stored values into spoken words. Stage: 1.

What this file is:
    Turns one stored value into the phrase a person would say.

What it contains:
    - spoken_slot(), which gives text such as "Tuesday 10 November at 10:00".
    - spoken_price(), which gives a price as words to say.

How to understand it:
    This is plain Python date and number formatting, with no LLM. The spoken text it creates
    must always come out the same, because reply_check searches for that exact string to prove a
    claim is true.

Used by:
    sentences.py, and the code that creates offers and appointments
"""

from datetime import datetime


def spoken_slot(start_at: datetime) -> str:
    """Return a time as it is said aloud, such as "Tuesday 10 November at 10:00"."""
    # start_at.day is used because %d would give "02", which sounds wrong when read aloud.
    return f"{start_at:%A} {start_at.day} {start_at:%B} at {start_at:%H:%M}"


def spoken_price(price_eur: int) -> str:
    """Return a price as it is said aloud, such as "30 euros"."""
    return f"{price_eur} euros"
