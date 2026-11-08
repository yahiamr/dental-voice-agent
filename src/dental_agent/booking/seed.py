"""Creates the sample data. Stage: 1.

What this file is:
    Fills the calendar with fictional patients and appointments.

What it contains:
    - seed(calendar, clock, clinic, reset=False).

How to understand it:
    A fixed random seed gives the same data on every run: 60 days of slots, 7 existing sample
    patients, 1 new sample patient and 8 filler patients. Every patient has exactly one
    appointment, and none is in the past, so a call never meets old bookings. The first sample
    patient has one check-up, which is handy for trying a call. Nora Blanco is new with a first
    visit ahead. Raúl Pastor is existing. The attendance rule, where a past first visit makes a
    new patient existing, is tested with its own data in tests/, because the seed has no past
    appointments. It refuses to overwrite data unless reset is true.

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
EARLIEST_DAYS = 3  # Seeded appointments start this far out, outside the notice period.

# Fictional patients. These numbers are placeholders, so nobody should dial them.
# The first seven are existing patients.
SAMPLE_PATIENTS = [
    ("+34600000001", "Lucía Moreno"),
    ("+34600000002", "Javier Ortega"),
    ("+34600000003", "Carmen Vidal"),
    ("+34600000004", "Diego Serrano"),
    ("+34600000005", "Elena Prieto"),
    ("+34600000006", "Marco Gil"),
]
# A new patient whose first visit is still ahead, so she can book nothing else yet.
NEW_WITH_FIRST_VISIT_AHEAD = ("+34600000007", "Nora Blanco")
# An existing patient, listed after Nora so the sample numbers keep their order.
EXISTING_LISTED_LAST = ("+34600000008", "Raúl Pastor")
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

    # Patient IDs start at 1, in this order: the six sample patients, the filler patients, then
    # Nora and Raúl. Only Nora is new.
    people = [(phone, name, "existing") for phone, name in SAMPLE_PATIENTS]
    people += [(f"+34600000{101 + offset}", name, "existing") for offset, name in
               enumerate(FILLER_NAMES)]  # fmt: skip
    people.append((*NEW_WITH_FIRST_VISIT_AHEAD, "new"))
    people.append((*EXISTING_LISTED_LAST, "existing"))
    patient_rows = [
        (index, phone, name, status, created_at)
        for index, (phone, name, status) in enumerate(people, start=1)
    ]

    # One appointment each, in distinct future slots spread over the 60 days. The eight sample
    # patients get the earliest, in the order they are listed, so a call soon finds theirs. The
    # first patient's check-up is the earliest of all.
    earliest = (now + timedelta(days=EARLIEST_DAYS)).isoformat()
    future = [row[0] for row in slot_rows if row[1] >= earliest]
    slots_in_order = [future[0], *sorted(rng.sample(future[1:], len(patient_rows) - 1))]
    fillers = len(FILLER_NAMES)
    samples = patient_rows[: len(SAMPLE_PATIENTS)] + patient_rows[len(SAMPLE_PATIENTS) + fillers :]
    order = samples + patient_rows[len(SAMPLE_PATIENTS) : len(SAMPLE_PATIENTS) + fillers]
    services = [item.id for item in clinic.services if not item.first_visit]
    first_visit = next(item.id for item in clinic.services if item.first_visit)
    appointment_rows = []
    for (patient_id, _phone, _name, status, _created), slot_id in zip(
        order, slots_in_order, strict=True
    ):
        if patient_id == 1:
            service = "check_up"
        elif status == "new":
            service = first_visit
        else:
            service = rng.choice(services)
        appointment_rows.append((patient_id, slot_id, service, "active", created_at))

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
