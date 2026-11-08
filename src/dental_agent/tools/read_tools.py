"""Tools that only read. Stage: 1 and 5.

What this file is:
    How the model looks things up.

What it contains:
    - find_slots: up to 2 earliest free slots. A new search replaces earlier offers. It refuses
      a service this caller cannot book, so such a time is never offered.
    - lookup_appointment: this caller's active appointments, up to 5.
    - get_clinic_facts: services, prices, hours or rules from clinic.yaml.
    - search_clinic_info: searches the clinic documents, from Stage 5.

How to understand it:
    These can run freely because they change nothing in the calendar. Slot and appointment IDs
    count up during a call and never repeat, so an old ID cannot point to a different slot.

Uses:
    booking/calendar.py, clinic/, formatting.py, models.py

Used by:
    tools/__init__.py
"""

from datetime import timedelta

from dental_agent.clinic.retriever import SearchUnavailable
from dental_agent.deps import GraphDeps
from dental_agent.formatting import spoken_price, spoken_slot
from dental_agent.guards.validate import caller_standing, service_error
from dental_agent.models import Appointment, ErrorCode, Offer, ToolOutcome, ToolResult
from dental_agent.state import CallState
from dental_agent.tools.schemas import (
    FindSlotsInput,
    GetClinicFactsInput,
    LookupAppointmentInput,
    SearchClinicInfoInput,
    failure,
)

MAX_OFFERS = 2  # Two choices are easy to follow by ear.
MAX_APPOINTMENTS = 5  # A limit, so a spoken list stays short.


def find_slots(args: FindSlotsInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Search free slots, and make the results the current offers."""
    clinic = deps.clinic
    if clinic.service(args.service) is None:
        return failure(ErrorCode.UNKNOWN_SERVICE, clinic)

    # Never offer a time for a service this caller cannot book. The record decides.
    status, booked = caller_standing(deps.calendar, state.caller_phone, clinic)
    refusal = service_error(args.service, status, booked, clinic)
    if refusal:
        return failure(refusal, clinic)

    # Reject dates that cannot be booked, before reading the calendar.
    now = deps.clock.now()
    last_day = (now + timedelta(days=clinic.booking_window.max_days_ahead)).date()
    bad_order = args.date_to < args.date_from
    in_past = args.date_to < now.date()
    too_far = args.date_to > last_day
    if bad_order or in_past or too_far:
        return failure(ErrorCode.INVALID_WINDOW, clinic)

    # "any" means no time limit, and the other two map to the opening hours.
    hours = {"morning": clinic.hours.morning, "afternoon": clinic.hours.afternoon}
    earliest = now + timedelta(hours=clinic.booking_window.min_hours_ahead)
    slots = deps.calendar.free_slots(
        args.date_from,
        args.date_to,
        args.weekdays,
        hours.get(args.time_window),
        MAX_OFFERS,
        earliest,
    )
    if not slots:
        # Old offers are cleared, so the model cannot pick a time from an earlier search.
        return failure(ErrorCode.NO_SLOTS, clinic, updates={"offers": []})

    # IDs count up through the call, so a stale ID can never name a different slot.
    number = state.next_slot_number
    offers = [
        Offer(
            slot_id=f"s{number + index}",
            db_slot_id=slot.db_slot_id,
            service=args.service,
            spoken=spoken_slot(slot.start_at),
        )
        for index, slot in enumerate(slots)
    ]
    # The model sees only the short ID and the words, never the database ID.
    data = [{"slot_id": offer.slot_id, "spoken": offer.spoken} for offer in offers]
    updates = {"offers": offers, "next_slot_number": number + len(offers)}
    return ToolOutcome(result=ToolResult(ok=True, data=data), updates=updates)


def with_call_ids(rows: list[Appointment], first_number: int) -> list[Appointment]:
    """Give calendar rows their per-call IDs and spoken text."""
    return [
        row.model_copy(
            update={
                "appointment_id": f"a{first_number + index}",
                "spoken": spoken_slot(row.start_at),
            }
        )
        for index, row in enumerate(rows)
    ]


def lookup_appointment(
    args: LookupAppointmentInput, state: CallState, deps: GraphDeps
) -> ToolOutcome:
    """List the active appointments for the number on this call."""
    # The phone comes from state, so the model cannot ask about another number.
    rows = deps.calendar.appointments_for(state.caller_phone, MAX_APPOINTMENTS)
    if not rows:
        return failure(ErrorCode.NO_APPOINTMENT, deps.clinic)
    found = with_call_ids(rows, state.next_appointment_number)
    data = [
        {
            "appointment_id": item.appointment_id,
            "patient_name": item.patient_name,
            "service": item.service,
            "spoken": item.spoken,
        }
        for item in found
    ]
    updates = {
        "found_appointments": found,
        "next_appointment_number": state.next_appointment_number + len(found),
    }
    return ToolOutcome(result=ToolResult(ok=True, data=data), updates=updates)


def get_clinic_facts(args: GetClinicFactsInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Answer from clinic.yaml, so the model never has to remember a price or an hour."""
    clinic = deps.clinic
    prices = [
        {"service": item.name, "price": spoken_price(item.price_eur)} for item in clinic.services
    ]
    facts = {
        "services": [{"id": item.id, "name": item.name} for item in clinic.services],
        "prices": prices,
        "hours": {
            "morning": list(clinic.hours.morning),
            "afternoon": list(clinic.hours.afternoon),
            "open_days": clinic.open_days,
            "closed": "public holidays in the Community of Madrid",
        },
        "rules": {
            "notice_hours": clinic.notice_hours,
            "min_hours_ahead": clinic.booking_window.min_hours_ahead,
            "max_days_ahead": clinic.booking_window.max_days_ahead,
            "new_patients": "New patients can book only the first-visit consultation.",
        },
    }
    if args.topic not in facts:
        return failure(ErrorCode.UNKNOWN_TOPIC, clinic)
    return ToolOutcome(result=ToolResult(ok=True, data=facts[args.topic]))


def search_clinic_info(
    args: SearchClinicInfoInput, state: CallState, deps: GraphDeps
) -> ToolOutcome:
    """Search the clinic's documents. Booking never depends on this working."""
    try:
        results = deps.retriever.search(args.question)
    except SearchUnavailable:
        return failure(ErrorCode.SEARCH_UNAVAILABLE, deps.clinic)
    data = [{"text": item.text, "source": item.source} for item in results]
    return ToolOutcome(result=ToolResult(ok=True, data=data))
