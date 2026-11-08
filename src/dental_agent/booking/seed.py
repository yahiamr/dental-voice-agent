"""Creates the sample data. Stage: 1.

What this file is:
    Fills the calendar with fictional patients and appointments.

What it contains:
    - seed(calendar, clock, clinic, reset=False).

How to understand it:
    A fixed random seed gives the same data on every run: 60 days of slots, 6 existing sample
    patients, 2 new sample patients and 8 filler patients, with about 40% of slots taken. The
    first sample patient has one check-up, which is handy for trying a call. Nora Blanco is new
    with a first visit ahead, and Raúl Pastor is new with a first visit last week, held in one
    past slot, so his next call marks him existing. It refuses to overwrite data unless reset
    is true.

Uses:
    booking/calendar.py, clock.py, clinic/data.py

Used by:
    cli.py, tests/conftest.py
"""

import random
from datetime import date, datetime, time, timedelta

import holidays

from dental_agent.booking.calendar import Calendar
from dental_agent.clinic.data import Clinic
from dental_agent.clock import Clock

# A fixed seed gives the same data on every run, so instructions and tests can rely on it.
RANDOM_SEED = 2026
SEED_DAYS = 60  # How far ahead slots are created.
TAKEN_SHARE = 0.40  # The share of slots that start out booked.

# Fictional patients. These numbers are placeholders, so nobody should dial them.
# The first six are existing patients.
SAMPLE_PATIENTS = [
    ("+34600000001", "Lucía Moreno"),
    ("+34600000002", "Javier Ortega"),
    ("+34600000003", "Carmen Vidal"),
    ("+34600000004", "Diego Serrano"),
    ("+34600000005", "Elena Prieto"),
    ("+34600000006", "Marco Gil"),
]
# Two new patients, to show the attendance flag. Nora's first visit is still ahead. Raúl's was
# last week and was not cancelled, so his next call marks him existing.
NEW_WITH_FIRST_VISIT_AHEAD = ("+34600000007", "Nora Blanco")
NEW_WITH_FIRST_VISIT_PAST = ("+34600000008", "Raúl Pastor")
PAST_VISIT_DAYS = 7  # How long ago Raúl's first visit was.
FILLER_NAMES = [
    "Sara Campos",
    "Hugo Nieto",
    "Paula Rey",
    "Iván Soler",
    "Marta Crespo",
    "Óscar Vega",
    "Nuria Lago",
    "Raúl Pardo",
]

# Sample patients 2 to 6 get this many appointments, so "0 to 2" is covered.
SAMPLE_APPOINTMENTS = [1, 0, 2, 0, 1]

WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def seed(calendar: Calendar, clock: Clock, clinic: Clinic, reset: bool = False) -> None:
    """Fill the calendar with sample data, refusing to overwrite data unless reset is true."""
    if calendar.slot_count() and not reset:
        raise SystemExit("The calendar already has data. Use `dental-agent seed --reset`.")
    if reset:
        calendar.clear()
    # A private generator, so nothing else in the program can change these results.
    rng = random.Random(RANDOM_SEED)
    now = clock.now()
    created_at = now.isoformat()

    slot_rows = _build_slots(now, clinic)
    slot_ids = [row[0] for row in slot_rows]

    # Patient IDs start at 1. Sample patients come first, and filler patients follow them.
    # Everyone seeded is existing, apart from the two new sample patients added at the end.
    patient_rows = [
        (index, phone, name, "existing", created_at)
        for index, (phone, name) in enumerate(SAMPLE_PATIENTS, start=1)
    ]
    first_filler_id = len(patient_rows) + 1
    for offset, name in enumerate(FILLER_NAMES):
        phone = f"+34600000{101 + offset}"
        patient_rows.append((first_filler_id + offset, phone, name, "existing", created_at))

    # Decide which slots are taken. Shuffling a copy keeps slot_rows in time order.
    shuffled = list(slot_ids)
    rng.shuffle(shuffled)
    taken = shuffled[: round(len(shuffled) * TAKEN_SHARE)]

    # Slots start three days out for the first patient's check-up, outside the notice period.
    start_by_id = {row[0]: row[1] for row in slot_rows}
    earliest_for_sample = (now + timedelta(days=3)).isoformat()
    later = [slot_id for slot_id in sorted(taken) if start_by_id[slot_id] >= earliest_for_sample]

    appointment_rows = [(1, later[0], "check_up", "active", created_at)]
    taken_by_sample = {later[0]}
    services = [item.id for item in clinic.services if not item.first_visit]

    # Give sample patients 2 to 6 their planned number of appointments.
    next_index = 1
    for patient_id, count in enumerate(SAMPLE_APPOINTMENTS, start=2):
        for _ in range(count):
            slot_id = later[next_index]
            next_index += 1
            taken_by_sample.add(slot_id)
            appointment_rows.append(
                (patient_id, slot_id, rng.choice(services), "active", created_at)
            )

    # Filler patients hold all the other taken slots.
    filler_ids = [row[0] for row in patient_rows[len(SAMPLE_PATIENTS) :]]
    for slot_id in taken:
        if slot_id not in taken_by_sample:
            patient_id = rng.choice(filler_ids)
            appointment_rows.append(
                (patient_id, slot_id, rng.choice(services), "active", created_at)
            )

    # The two new patients come last, so the rows above do not change.
    first_visit = next(item.id for item in clinic.services if item.first_visit)
    nora_id, raul_id = len(patient_rows) + 1, len(patient_rows) + 2
    patient_rows.append((nora_id, *NEW_WITH_FIRST_VISIT_AHEAD, "new", created_at))
    patient_rows.append((raul_id, *NEW_WITH_FIRST_VISIT_PAST, "new", created_at))
    free_later = [
        slot_id
        for slot_id in slot_ids
        if slot_id not in taken and start_by_id[slot_id] >= earliest_for_sample
    ]
    appointment_rows.append((nora_id, free_later[0], first_visit, "active", created_at))
    # Raúl's visit needs a slot in the past. Past slots are never searched, so it cannot be
    # offered to anyone.
    past_start = now.replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(
        days=PAST_VISIT_DAYS
    )
    past_id = len(slot_rows) + 1
    past_end = past_start + timedelta(minutes=clinic.slot_minutes)
    slot_rows.append((past_id, past_start.isoformat(), past_end.isoformat()))
    appointment_rows.append((raul_id, past_id, first_visit, "active", created_at))

    calendar.seed_rows(patient_rows, slot_rows, appointment_rows)


def _build_slots(now: datetime, clinic: Clinic) -> list[tuple[int, str, str]]:
    """Create one row per bookable time, on open days that are not public holidays."""
    closed = holidays.country_holidays(
        clinic.holidays.country,
        subdiv=clinic.holidays.subdivision,
        years=[now.year, now.year + 1],
    )
    rows: list[tuple[int, str, str]] = []
    for offset in range(SEED_DAYS):
        day = now.date() + timedelta(days=offset)
        if not _is_open(day, clinic, closed):
            continue
        for start_text, end_text in (clinic.hours.morning, clinic.hours.afternoon):
            start = datetime.combine(day, time.fromisoformat(start_text), tzinfo=now.tzinfo)
            end = datetime.combine(day, time.fromisoformat(end_text), tzinfo=now.tzinfo)
            # Step through the session, keeping only slots that end before it closes.
            while start + timedelta(minutes=clinic.slot_minutes) <= end:
                finish = start + timedelta(minutes=clinic.slot_minutes)
                rows.append((len(rows) + 1, start.isoformat(), finish.isoformat()))
                start = finish
    return rows


def _is_open(day: date, clinic: Clinic, closed: holidays.HolidayBase) -> bool:
    """Return True if the clinic opens on this day."""
    return WEEKDAY_NAMES[day.weekday()] in clinic.open_days and day not in closed
