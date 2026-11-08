"""Tools that only read. Stage: 1, 3 and 5.

What this file is:
    How the model looks things up.

What it contains:
    - find_slots: up to 2 earliest free slots. A new search replaces earlier offers. It refuses
      a service this caller cannot book, so such a time is never offered. With appointment_id,
      it searches new times for that found appointment's own service, so a new patient can move
      their first visit, and refuses an appointment inside the notice period. Its result tells
      the model to prepare a time the caller already asked for, without asking first: with
      prepare_booking for a booking search, and prepare_reschedule for a move search. With time
      (HH:MM), it looks up that exact time, and if it is not free returns the nearest free
      times that day and says so, because the earliest-first search never saw later times.
    - lookup_appointment: this caller's active appointments still to come, up to 5.
    - get_clinic_facts: services, prices, hours or rules from clinic.yaml.
    - search_clinic_info: searches the patient handbook, and returns up to 3 pieces, each with
      its heading path, its text cut at MAX_PIECE_CHARS, and its source, with a message telling
      the model to answer only from them. An empty search says to give the clinic number.

How to understand it:
    These can run freely because they change nothing in the calendar. Slot and appointment IDs
    count up during a call and never repeat, so an old ID cannot point to a different slot.

Uses:
    booking/calendar.py, clinic/, formatting.py, models.py

Used by:
    tools/__init__.py
"""

from datetime import datetime, timedelta

from dental_agent.clinic.retriever import SearchUnavailable
from dental_agent.deps import GraphDeps
from dental_agent.formatting import spoken_price, spoken_slot
from dental_agent.guards.validate import caller_standing, notice_blocks, service_error
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
DAY_SEARCH_LIMIT = 500  # Enough free slots to find the nearest to an exact time on any day.
MAX_PIECE_CHARS = 1000  # A search piece longer than this is cut, so results stay short.

# Sent with every search result, because the model decides here what to say. A search always
# returns its closest pieces, even when none of them answers the question.
SEARCH_MESSAGE = (
    "Answer only from what these pieces say. If they do not answer the question, say you do "
    "not have that information and give the clinic number. Prices, hours and booking rules "
    "come from get_clinic_facts and the booking tools, which win if they differ."
)
# Sent when the exact time asked for is not free. No prepare, because the caller did not ask
# for these times.
NOT_FREE_MESSAGE = (
    "{time} is not free on the days asked for, or is not a time the clinic books. The nearest "
    "free times are on {day}: these. Say so in one sentence, and offer them."
)
NOTHING_FOUND_MESSAGE = (
    "Nothing in the clinic documents matches. Say you do not have that information and give "
    "the clinic number."
)

# Sent with every search result, because the model decides here. With the rule only in the
# system prompt, live calls still asked "Would you like me to book that?" after finding the exact
# time the caller asked for, so the caller had to say yes twice. A prepare with no saved service
# or no name fails and can use up the tool rounds, so the booking message names both. A search
# for a move gets its own message, which never names prepare_booking, so the model cannot book a
# second appointment instead of moving the first.
BOOK_FOUND_MESSAGE = (
    "If the caller already asked for one of these exact times and their name is known, call "
    "prepare_booking for it now, with save_details in the same response if the service is not "
    "saved yet. Do not ask whether to book it, because the system asks. Any text you send with "
    'it is spoken, so keep it to one short sentence such as "Yes, that time is free." '
    "Otherwise offer these times."
)
MOVE_FOUND_MESSAGE = (
    "If the caller already asked for one of these exact times, call prepare_reschedule now with "
    "this appointment_id and that slot_id. Do not ask whether to move it, because the system "
    'asks. Any text you send with it is spoken, so keep it to one short sentence such as "Yes, '
    'that time is free." Otherwise offer these times.'
)


def find_slots(args: FindSlotsInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Search free slots, and make the results the current offers."""
    clinic = deps.clinic
    service_id = args.service
    if args.appointment_id is not None:
        # A search to move an appointment is for that appointment's own service. The caller
        # already has it, so the rule on what they may book does not apply.
        moving = state.found(args.appointment_id)
        if moving is None:
            return failure(ErrorCode.NOT_OWNER, clinic)
        # Never offer times for a move that prepare_reschedule would refuse.
        actions = deps.calendar.actions_for_call(state.conversation_id)
        if notice_blocks(moving, actions, deps.clock.now(), clinic):
            return failure(ErrorCode.POLICY_BLOCKED, clinic)
        service_id = moving.service
    elif clinic.service(service_id) is None:
        return failure(ErrorCode.UNKNOWN_SERVICE, clinic)
    else:
        # Never offer a time for a service this caller cannot book. The record decides.
        status, booked = caller_standing(deps.calendar, state.caller_phone, clinic)
        refusal = service_error(service_id, status, booked, clinic)
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
    exact_free = None  # Whether an exact time asked for is free. None when none was asked.
    if args.time is not None:
        slots, exact_free = _exact_time(deps, args, earliest)
    else:
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
            service=service_id,
            spoken=spoken_slot(slot.start_at),
        )
        for index, slot in enumerate(slots)
    ]
    # The model sees only the short ID and the words, never the database ID.
    data = [{"slot_id": offer.slot_id, "spoken": offer.spoken} for offer in offers]
    updates = {"offers": offers, "next_slot_number": number + len(offers)}
    message = BOOK_FOUND_MESSAGE if args.appointment_id is None else MOVE_FOUND_MESSAGE
    if exact_free is True:
        message = f"{args.time} is free on the days shown. " + message
    elif exact_free is False:
        day = spoken_slot(slots[0].start_at).rsplit(" at ", 1)[0]
        message = NOT_FREE_MESSAGE.format(time=args.time, day=day)
    result = ToolResult(ok=True, data=data, message=message)
    return ToolOutcome(result=result, updates=updates)


def _exact_time(deps: GraphDeps, args: FindSlotsInput, earliest: datetime) -> tuple[list, bool]:
    """Return the slots at exactly args.time, or the nearest free ones, and whether it was free.

    The earliest-first search only ever saw the first two times of a day, so a caller asking
    about 10:00 heard it was taken while it was free. An exact time is looked up on its own.
    If it is not free, or is not a time the clinic books, the nearest free times on the first
    day that has any are returned instead, so the caller still gets a real choice.
    """
    start = datetime.strptime(args.time, "%H:%M")
    exact_window = (args.time, (start + timedelta(minutes=1)).strftime("%H:%M"))
    calendar = deps.calendar
    exact = calendar.free_slots(
        args.date_from, args.date_to, args.weekdays, exact_window, MAX_OFFERS, earliest
    )
    if exact:
        return exact, True
    free = calendar.free_slots(
        args.date_from, args.date_to, args.weekdays, None, DAY_SEARCH_LIMIT, earliest
    )
    if not free:
        return [], False
    first_day = free[0].start_at.date()
    target = start.hour * 60 + start.minute

    def distance(slot) -> tuple[int, datetime]:
        minutes = slot.start_at.hour * 60 + slot.start_at.minute
        return abs(minutes - target), slot.start_at

    same_day = [slot for slot in free if slot.start_at.date() == first_day]
    nearest = sorted(same_day, key=distance)[:MAX_OFFERS]
    return sorted(nearest, key=lambda slot: slot.start_at), False


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
    # The phone comes from state, so the model cannot ask about another number. Past
    # appointments are left out, because they cannot be changed.
    rows = deps.calendar.appointments_for(
        state.caller_phone, MAX_APPOINTMENTS, after=deps.clock.now()
    )
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
    # Traced, so an answer can be checked afterwards against what search actually returned.
    deps.trace.event(
        "search",
        question=args.question,
        pieces=[item.piece_id for item in results],
        scores=[round(item.score, 4) for item in results],
    )
    data = [
        {"heading": item.heading, "text": _cut(item.text), "source": item.source}
        for item in results
    ]
    message = SEARCH_MESSAGE if data else NOTHING_FOUND_MESSAGE
    # Kept for the next turn's Current Facts, so an answer from them is not taken back.
    kept = [{"heading": item["heading"], "text": item["text"]} for item in data]
    result = ToolResult(ok=True, data=data, message=message)
    return ToolOutcome(result=result, updates={"last_search": kept})


def _cut(text: str) -> str:
    """Return the text, cut at a word near MAX_PIECE_CHARS, with an ellipsis if it was cut."""
    if len(text) <= MAX_PIECE_CHARS:
        return text
    return text[:MAX_PIECE_CHARS].rsplit(" ", 1)[0] + "…"
