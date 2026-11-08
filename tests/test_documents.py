"""Tests for the clinic documents the search reads. Stage: 5.

What this file is:
    Proves the patient handbook never contradicts the facts the booking tools use.

What it contains:
    - Every price in the handbook's price table matches clinic.yaml.
    - The opening hours, appointment length, booking window and notice period match.
    - Every phone number is the clinic line, the urgent line or 112.
    - No dosing word appears, so a reply read from the handbook is never replaced by the medical
      line for a word the handbook itself put there.

How to understand it:
    clinic.yaml is the source of truth, and the tools answer from it. The handbook is invented
    prose, so a typo there would give a caller two different answers depending on which one the
    model used. These tests catch that before the index is built.

Used by:
    pytest
"""

import re
from pathlib import Path

import pytest

HANDBOOK = Path(__file__).parent.parent / "data" / "docs" / "patient-handbook.md"


@pytest.fixture
def raw() -> str:
    """The handbook's lines as written, without its header comment, for the price table."""
    return re.sub(r"<!--.*?-->", "", HANDBOOK.read_text(), flags=re.DOTALL)


@pytest.fixture
def handbook(raw) -> str:
    """The handbook as one line of prose, so a sentence split across lines still matches."""
    return " ".join(raw.split())


def test_every_price_in_the_table_matches_the_clinic_file(raw, clinic):
    rows = re.findall(r"^\| ([^|]+?) \| (\d+) euros", raw, flags=re.MULTILINE)
    prices = {name.strip().lower(): int(price) for name, price in rows}
    expected = {service.name.lower(): service.price_eur for service in clinic.services}
    assert prices == expected


def test_the_opening_hours_match(handbook, clinic):
    morning, afternoon = clinic.hours.morning, clinic.hours.afternoon
    assert (
        f"from {morning[0]} to {morning[1]} and from {afternoon[0]} to {afternoon[1]}" in handbook
    )
    assert "Monday to Friday" in handbook


def test_the_appointment_length_matches(handbook, clinic):
    # Only lengths of an appointment, not walking times or how late a patient may be.
    patterns = [r"(\d+)-minute appointment", r"[Aa]ppointments (?:are|last) (\d+) minutes"]
    lengths = {int(value) for pattern in patterns for value in re.findall(pattern, handbook)}
    assert lengths == {clinic.slot_minutes}


def test_the_booking_window_matches(handbook, clinic):
    window = clinic.booking_window
    hours = {1: "one", 2: "two", 3: "three"}[window.min_hours_ahead]
    assert f"from {hours} hours ahead up to {window.max_days_ahead} days ahead" in handbook


def test_the_notice_period_matches(handbook, clinic):
    periods = {int(value) for value in re.findall(r"(\d+) hours'? (?:notice|away)", handbook)}
    assert periods == {clinic.notice_hours}


def test_every_phone_number_is_one_the_clinic_uses(handbook, clinic):
    numbers = set(re.findall(r"\b\d{3} \d{3} \d{3}\b", handbook))
    assert numbers == {clinic.clinic_line, clinic.urgent_line}
    assert "112" in handbook


def test_no_dosing_word_appears(handbook, clinic):
    words = set(re.findall(r"[a-z]+", handbook.lower()))
    assert words.isdisjoint(word.lower() for word in clinic.dosing_words)
