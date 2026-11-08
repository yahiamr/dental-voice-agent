"""Checks an action is allowed. Stage: 1 and 3.

What this file is:
    The business rules for a booking, move or cancel.

What it contains:
    - validate_action(action, context, calendar, now, clinic) -> ErrorCode | None.
    - Seven ordered checks: appointment found and active, details set, the number decides the
      service, slot on offer, slot free, slot in the booking window, notice period.
    - service_error(service_id, on_record, clinic): a new number books only the first visit,
      and a number on record books anything else. find_slots uses it too.
    - notice_blocks(appointment, actions, now, clinic): the notice rule in one place. find_slots
      and the prompt's Current Facts use it too, so they agree with the move and cancel checks.

How to understand it:
    It runs twice: at prepare, so the caller hears only valid options, and in
    write_to_calendar, because a slot can be taken between the two. It returns the first failure
    as a typed error, which tools/ turns into a message for the model.

    A move or cancel skips checks 2 and 3: it changes an appointment the caller already has, so
    the booking details and the first-visit rule do not apply. The notice period and the
    appointment's status are read from the calendar, not from the lookup, so a stale lookup
    cannot get round them. An appointment moved since the lookup is refused, because the caller
    would hear its old time. An appointment booked or moved in this call is exempt from the
    notice period, so a caller can fix a mistake at once.

Uses:
    models.py

Used by:
    tools/action_tools.py, tools/read_tools.py, prompts/build.py, graph/nodes.py (write_to_calendar)
"""

from datetime import datetime, timedelta

from dental_agent.booking.calendar import Calendar
from dental_agent.clinic.data import Clinic
from dental_agent.models import (
    ActionContext,
    ActionRecord,
    Appointment,
    ErrorCode,
    Offer,
    PendingAction,
)


def validate_action(
    action: PendingAction,
    context: ActionContext,
    calendar: Calendar,
    now: datetime,
    clinic: Clinic,
) -> ErrorCode | None:
    """Return the first rule this action breaks, or None if it is allowed."""
    if action.kind != "book":
        return _change_error(action, context, calendar, now, clinic)
    details = context.details

    # Check 2: a booking needs a name and a service. The patient type comes from the number.
    # Check 1 applies only to a move or cancel.
    if details.missing_for_booking():
        return ErrorCode.MISSING_FIELD

    # Check 3: the number decides which services can be booked. One number is one patient,
    # so a number on record is an existing patient, and anything said on the call cannot
    # change that. It also runs in find_slots, so a caller is never offered a time they cannot
    # book.
    status, booked = caller_standing(calendar, context.caller_phone, clinic)
    error = service_error(details.service, status, booked, clinic)
    if error:
        return error

    # Check 4: the slot must be one we offered, for the service being booked.
    # This stops the model booking a real slot it was never given.
    if not _was_offered(action.offer, context.offers, details.service):
        return ErrorCode.SLOT_NOT_OFFERED

    # Checks 5 and 6: the slot is still free and still inside the booking window.
    # Check 7, the notice period, applies only to a move or cancel.
    return _slot_error(action.offer, calendar, now, clinic)


def _change_error(
    action: PendingAction,
    context: ActionContext,
    calendar: Calendar,
    now: datetime,
    clinic: Clinic,
) -> ErrorCode | None:
    """Return the first rule a move or cancel breaks. Checks 2 and 3 are for bookings only."""
    target = action.appointment
    # Check 1: the appointment must be one lookup_appointment gave this caller, and still
    # active. found_appointments holds only this number's appointments, so this is ownership.
    if target is None or not any(
        item.appointment_id == target.appointment_id and item.db_id == target.db_id
        for item in context.found_appointments
    ):
        return ErrorCode.NOT_OWNER
    # Read the row again: it may have changed since the lookup, and its time is the truth.
    current = calendar.get_appointment(target.db_id)
    if current is None or current.status != "active":
        return ErrorCode.NOT_OWNER
    # A row moved since the lookup would be read back at its old time, so it is refused too.
    if current.db_slot_id != target.db_slot_id:
        return ErrorCode.NOT_OWNER

    if action.kind == "move":
        # Check 4: the new slot must be on offer for the appointment's own service.
        if not _was_offered(action.offer, context.offers, current.service):
            return ErrorCode.SLOT_NOT_OFFERED
        # Checks 5 and 6.
        error = _slot_error(action.offer, calendar, now, clinic)
        if error:
            return error

    # Check 7: no change inside the notice period, unless this call booked or moved it.
    if notice_blocks(current, calendar.actions_for_call(context.conversation_id), now, clinic):
        return ErrorCode.POLICY_BLOCKED
    return None


def notice_blocks(
    appointment: Appointment, actions: list[ActionRecord], now: datetime, clinic: Clinic
) -> bool:
    """Return True if the notice period stops this appointment changing on this line.

    It does when the appointment starts too soon and this call did not book or move it. An
    appointment this call made is exempt, so the caller can fix a mistake at once.
    """
    too_soon = appointment.start_at < now + timedelta(hours=clinic.notice_hours)
    made_here = any(
        item.appointment_id == appointment.db_id and item.kind in ("book", "move")
        for item in actions
    )
    return too_soon and not made_here


def _slot_error(
    offer: Offer, calendar: Calendar, now: datetime, clinic: Clinic
) -> ErrorCode | None:
    """Checks 5 and 6: the slot must still be free and still inside the booking window."""
    if not calendar.slot_is_free(offer.db_slot_id):
        return ErrorCode.SLOT_TAKEN
    # Time may have passed since the offer, so the window is checked again.
    slot = calendar.get_slot(offer.db_slot_id)
    window = clinic.booking_window
    earliest = now + timedelta(hours=window.min_hours_ahead)
    latest = now + timedelta(days=window.max_days_ahead)
    if slot is None or not earliest <= slot.start_at <= latest:
        return ErrorCode.SLOT_TAKEN
    return None


def first_visit_id(clinic: Clinic) -> str:
    """Return the ID of the first-visit consultation."""
    return next(item.id for item in clinic.services if item.first_visit)


def caller_standing(calendar: Calendar, phone: str, clinic: Clinic) -> tuple[str | None, bool]:
    """Return the record's status (None for a new caller) and whether a first visit is booked."""
    status = calendar.patient_status(phone)
    booked = status == "new" and calendar.has_active_service(phone, first_visit_id(clinic))
    return status, booked


def service_error(
    service_id: str | None, status: str | None, first_visit_booked: bool, clinic: Clinic
) -> ErrorCode | None:
    """Return why this caller cannot book this service, or None if they can.

    A patient is new until their first visit is attended. A new patient, on record or not, can
    book only the first-visit consultation, and only one at a time. An existing patient can
    book anything else.
    """
    service = clinic.service(service_id)
    first_visit = bool(service and service.first_visit)
    if status == "existing":
        return ErrorCode.EXISTING_PATIENT if first_visit else None
    if not first_visit:
        return ErrorCode.FIRST_VISIT_REQUIRED
    if first_visit_booked:
        return ErrorCode.FIRST_VISIT_BOOKED
    return None


def _was_offered(offer: Offer | None, offers: list[Offer], service: str | None) -> bool:
    """Return True if this exact slot is on offer for this service."""
    if offer is None:
        return False
    return any(
        item.slot_id == offer.slot_id
        and item.db_slot_id == offer.db_slot_id
        and item.service == service
        for item in offers
    )
